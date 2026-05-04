"""Analyze the booklist and emit a standalone HTML page with Plotly charts.

Outputs a single page (default: personal-website/books.html) styled to match
Ryan's Pixelarity Story template. Re-run after editing the source xlsx.
"""

from __future__ import annotations

import argparse
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent  # personal-website/scripts/
SITE = ROOT.parent                       # personal-website/
DEFAULT_INPUT = SITE.parent / "Claude-personal" / "books" / "data" / "booklist.xlsx"
DEFAULT_OUTPUT = SITE / "books.html"
GOODREADS_URL = "https://docs.google.com/spreadsheets/d/1mII4DnEvJ5OjeU0u0IXgfmgKPfY6ymeP/edit?gid=2023378342#gid=2023378342"

# Site palette (from personal-website/assets/css/main.css)
ACCENT = "#47D3E5"
ACCENT_DARK = "#1ebdd1"
INK = "#1c1c1c"
MUTED = "#7f888f"

# Letter grade order, best to worst
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]

# Authors excluded from the "most-read authors" chart because their counts
# are dominated by long series — distorts the leaderboard.
SERIES_AUTHORS = {
    "Orson Scott Card",
    "J.K. Rowling",
    "Eliezer Yudkowsky",
    "Cixin Liu",
    "Liu Cixin",  # same person, logged two different ways
    "James S.A. Corey",
    "Hilary Mantel",
    "Kim Stanley Robinson",
}

# Titles excluded from the my-vs-Goodreads scatter — typically obscure books
# whose GR average is based on a tiny sample and isn't comparable.
SCATTER_EXCLUDED_TITLES = {
    "The Dying of the Light",  # GR avg of 5.0 from too-few raters
}


def _apply_layout(fig: go.Figure, *, height: int = 420) -> go.Figure:
    """Common Plotly layout — transparent paper so site bg shows through."""
    fig.update_layout(
        font=dict(family='"Source Sans Pro", Helvetica, sans-serif', color=INK, size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=50, b=60),
        height=height,
        title=dict(font=dict(size=18)),
        xaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0.2)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", linecolor="rgba(0,0,0,0.2)"),
        hoverlabel=dict(font_family='"Source Sans Pro", Helvetica, sans-serif'),
    )
    return fig


def load_data(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_excel(xlsx_path, sheet_name="Booklist")
    books = books.rename(columns={
        "Letter Grade Converted to 5.0 scale": "MyRating",
        "Goodreads avg rating": "GRRating",
        "Year Read": "Year",
        "Read count": "ReadCount",
        "Letter Grade": "Grade",
    })
    books["Year"] = books["Year"].astype(int)

    dnf = pd.read_excel(xlsx_path, sheet_name="DNF", header=2)
    # Sheet sometimes has stray non-empty cells in unrelated columns below the
    # actual list — anchor on Book column being populated.
    dnf = dnf[dnf["Book"].notna()].reset_index(drop=True)
    return books, dnf


# ---------- chart builders ----------

def chart_books_per_year(books: pd.DataFrame) -> go.Figure:
    counts = books.groupby("Year").size().reindex(range(books["Year"].min(), books["Year"].max() + 1), fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>%{y} books<extra></extra>",
    ))
    fig.update_layout(title="Books read per year", xaxis_title=None, yaxis_title="Books")
    return _apply_layout(fig)


def chart_grade_distribution(books: pd.DataFrame) -> go.Figure:
    present = [g for g in GRADE_ORDER if g in books["Grade"].values]
    counts = books["Grade"].value_counts().reindex(present)
    pct = counts / counts.sum() * 100
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=[f"{p:.1f}%" for p in pct], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} books (%{customdata:.1f}%)<extra></extra>",
        customdata=pct.values,
    ))
    fig.update_layout(
        title="Grade distribution",
        xaxis=dict(categoryorder="array", categoryarray=present, title=None),
        yaxis_title="Books",
    )
    return _apply_layout(fig)


def chart_genre_breakdown(books: pd.DataFrame) -> go.Figure:
    counts = books["Genre"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=counts.values, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} books<extra></extra>",
    ))
    fig.update_layout(title="Books by genre", xaxis_title="Books", yaxis_title=None)
    return _apply_layout(fig, height=520)


def chart_top_authors(books: pd.DataFrame, top_n: int = 15) -> go.Figure:
    filtered = books[~books["Author"].isin(SERIES_AUTHORS)]
    counts = filtered["Author"].value_counts().head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=counts.values, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} books<extra></extra>",
    ))
    fig.update_layout(title=f"Most-read authors (top {top_n})", xaxis_title="Books", yaxis_title=None)
    return _apply_layout(fig, height=520)


def chart_my_vs_goodreads(books: pd.DataFrame) -> go.Figure:
    df = books.dropna(subset=["GRRating", "MyRating"]).copy()
    df["Title_clean"] = df["Title"].map(clean_title)
    df = df[~df["Title_clean"].isin(SCATTER_EXCLUDED_TITLES)]
    # R² of the simple linear fit between my rating and the Goodreads average
    r2 = float(np.corrcoef(df["GRRating"], df["MyRating"])[0, 1] ** 2)
    fig = px.scatter(
        df, x="GRRating", y="MyRating", color="Genre",
        hover_data={"Title": True, "Author": True, "Grade": True, "GRRating": ":.2f", "MyRating": False, "Genre": False},
        labels={"GRRating": "Goodreads average", "MyRating": "My rating (5.0 scale)"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="rgba(0,0,0,0.3)")))
    # Reference line: y = x
    fig.add_shape(type="line", x0=2.5, x1=5, y0=2.5, y1=5,
                  line=dict(color="rgba(0,0,0,0.25)", dash="dash", width=1))
    fig.add_annotation(x=4.9, y=5, text="agree with Goodreads", showarrow=False,
                       font=dict(size=11, color=MUTED), xanchor="right", yanchor="bottom")
    # R² in the upper-left, in paper coords
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"R² = {r2:.3f}<br><span style='font-size:11px;color:{MUTED}'>n = {len(df)}</span>",
        showarrow=False, align="left",
        font=dict(size=14, color=INK),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
        borderpad=6, xanchor="left", yanchor="top",
    )
    fig.update_layout(title="My rating vs. Goodreads average")
    return _apply_layout(fig, height=560)


def chart_avg_grade_by_genre(books: pd.DataFrame, min_books: int = 5) -> go.Figure:
    g = books.groupby("Genre").agg(avg=("MyRating", "mean"), n=("MyRating", "size")).reset_index()
    g = g[g["n"] >= min_books].sort_values("avg")
    fig = go.Figure(go.Bar(
        x=g["avg"], y=g["Genre"], orientation="h",
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=[f"{a:.2f}  (n={n})" for a, n in zip(g["avg"], g["n"])], textposition="outside",
        hovertemplate="<b>%{y}</b><br>avg %{x:.2f} across %{customdata} books<extra></extra>",
        customdata=g["n"],
    ))
    fig.update_layout(
        title=f"Average rating by genre (genres with ≥ {min_books} books)",
        xaxis=dict(title="Average rating (5.0 scale)", range=[2.5, 5.2]),
        yaxis_title=None,
    )
    return _apply_layout(fig, height=520)


def chart_cumulative(books: pd.DataFrame) -> go.Figure:
    df = books.sort_values("Date Read").copy()
    df["cum"] = range(1, len(df) + 1)
    fig = go.Figure(go.Scatter(
        x=df["Date Read"], y=df["cum"], mode="lines",
        line=dict(color=ACCENT_DARK, width=2.5),
        fill="tozeroy", fillcolor="rgba(71, 211, 229, 0.18)",
        hovertemplate="%{x|%b %Y}<br><b>%{y}</b> books read<extra></extra>",
    ))
    fig.update_layout(title="Cumulative books over time", xaxis_title=None, yaxis_title="Total books read")
    return _apply_layout(fig)


def chart_year_genre_heatmap(books: pd.DataFrame, year_min: int | None = None) -> go.Figure:
    if year_min is None:
        year_min = int(books["Year"].min())
    df = books[books["Year"] >= year_min]
    pivot = df.pivot_table(index="Genre", columns="Year", values="Title", aggfunc="count", fill_value=0)
    # Order genres by total descending (so the busiest are at the top)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "rgba(71,211,229,0.05)"], [1, ACCENT_DARK]],
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} books<extra></extra>",
        colorbar=dict(title="Books", thickness=12),
    ))
    fig.update_layout(
        title=f"What I read, when (genre × year, since {year_min})",
        xaxis=dict(title=None, dtick=1),
        yaxis_title=None,
    )
    return _apply_layout(fig, height=560)


# ---------- table builders (HTML) ----------

def _table(rows: list[list[str]], headers: list[str]) -> str:
    head_html = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f'<table class="books-table"><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>'


_RECOUNT_RE = re.compile(r"\s*\((?:\d+x|\d+(?:st|nd|rd|th)\s*read)\)\s*$", re.IGNORECASE)


def clean_title(t: str) -> str:
    return _RECOUNT_RE.sub("", str(t)).strip()


def table_favorites(books: pd.DataFrame) -> str:
    df = books[books["Grade"] == "A+"].copy()
    df["Title_clean"] = df["Title"].map(clean_title)
    # Dedupe re-read favorites; keep the most recent read for the year column.
    df = df.sort_values("Date Read", ascending=False)
    df = df.drop_duplicates(subset=["Title_clean", "Author"], keep="first")
    rows = [[r["Title_clean"], r["Author"], int(r["Year"]), r["Grade"]] for _, r in df.iterrows()]
    return _table(rows, ["Title", "Author", "Year read", "Grade"])


def table_rereads(books: pd.DataFrame) -> str:
    df = books[books["ReadCount"] > 1].copy()
    df["Title_clean"] = df["Title"].map(clean_title)
    # A re-read book may appear as multiple rows (one per read). Keep only the
    # row with the highest ReadCount per (title, author) — that's the most
    # recent read and the canonical times-read total.
    df = df.sort_values("ReadCount", ascending=False)
    df = df.drop_duplicates(subset=["Title_clean", "Author"], keep="first")
    df = df.sort_values(["ReadCount", "Date Read"], ascending=[False, False])
    rows = [[r["Title_clean"], r["Author"], int(r["ReadCount"]), r["Year"]] for _, r in df.iterrows()]
    return _table(rows, ["Title", "Author", "Times read", "Most recent"])


def table_top_diffs(books: pd.DataFrame, kind: str, n: int = 12) -> str:
    df = books.dropna(subset=["GRRating"]).copy()
    df["diff"] = df["MyRating"] - df["GRRating"]
    df["Title_clean"] = df["Title"].map(clean_title)
    # Re-read books can appear as multiple rows with the same diff — collapse.
    df = df.drop_duplicates(subset=["Title_clean", "Author"])
    df = df.sort_values("diff", ascending=(kind == "low"))
    df = df.head(n)
    rows = [
        [r["Title_clean"], r["Author"], r["Grade"], f'{r["GRRating"]:.2f}', f'{r["diff"]:+.2f}']
        for _, r in df.iterrows()
    ]
    return _table(rows, ["Title", "Author", "My grade", "Goodreads avg", "Diff"])


def table_dnf(dnf: pd.DataFrame) -> str:
    df = dnf.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%b %Y").fillna("")
    rows = [[r.get("Book", ""), r.get("Author", ""), r.get("Date", ""), r.get("Reason", "")]
            for _, r in df.iterrows()]
    return _table(rows, ["Book", "Author", "Date", "Reason"])


# ---------- summary numbers ----------

def headline_stats(books: pd.DataFrame) -> dict[str, str]:
    total = len(books)
    avg = books["MyRating"].mean()
    years_active = books["Year"].nunique()
    unique_authors = books["Author"].nunique()
    rereads = (books["ReadCount"] > 1).sum()
    return {
        "Total books": f"{total:,}",
        "Average rating": f"{avg:.2f}",
        "Years tracked": str(years_active),
        "Total authors": f"{unique_authors:,}",
        "Books re-read": str(rereads),
    }


# ---------- page assembly ----------

def fig_div(fig: go.Figure) -> str:
    """Render a Plotly figure as an embeddable <div> (no full HTML, no JS bundle)."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


# ---------- patch mode ----------
# books.html is the canonical source for prose. We don't regenerate it from
# scratch on every run; instead we replace just the data-driven blocks
# (charts, tables, headline stats) between HTML comment markers like
#   <!-- chart:per_year -->...<!-- /chart:per_year -->
# Anything outside the markers — banner copy, section h3s, .blurb text — is
# Ryan's to edit by hand and survives every regeneration.

# (marker name in books.html, fragment key) — order matters only for logging.
MARKERS: list[tuple[str, str]] = [
    ("stats", "stats"),
    ("chart:per_year", "per_year"),
    ("chart:cumulative", "cumulative"),
    ("table:favorites", "favorites"),
    ("chart:grades", "grades"),
    ("chart:genres", "genres"),
    ("chart:avg_by_genre", "avg_by_genre"),
    ("chart:authors", "authors"),
    ("chart:my_vs_gr", "my_vs_gr"),
    ("chart:heatmap", "heatmap"),
    ("table:rereads", "rereads"),
    ("table:diffs_high", "diffs_high"),
    ("table:diffs_low", "diffs_low"),
    ("table:dnf", "dnf"),
]


def build_fragments(books: pd.DataFrame, dnf: pd.DataFrame) -> dict[str, str]:
    """All the data-driven HTML blocks, keyed by marker name."""
    figs = {
        "per_year": chart_books_per_year(books),
        "cumulative": chart_cumulative(books),
        "grades": chart_grade_distribution(books),
        "genres": chart_genre_breakdown(books),
        "avg_by_genre": chart_avg_grade_by_genre(books),
        "authors": chart_top_authors(books),
        "my_vs_gr": chart_my_vs_goodreads(books),
        "heatmap": chart_year_genre_heatmap(books),
    }
    stats = headline_stats(books)
    stat_cards = "".join(
        f'<li><span class="stat-num">{escape(v)}</span><span class="stat-label">{escape(k)}</span></li>'
        for k, v in stats.items()
    )
    return {
        "stats": f'<ul class="books-stats">{stat_cards}</ul>',
        **{k: fig_div(v) for k, v in figs.items()},
        "favorites": table_favorites(books),
        "rereads": table_rereads(books),
        "diffs_high": table_top_diffs(books, "high"),
        "diffs_low": table_top_diffs(books, "low"),
        "dnf": table_dnf(dnf),
    }


def patch_block(html: str, marker: str, fragment: str) -> str:
    """Replace whatever is between <!-- {marker} --> and <!-- /{marker} --> with fragment."""
    pattern = re.compile(
        r"(<!-- " + re.escape(marker) + r" -->\n)(.*?)(\n[ \t]*<!-- /" + re.escape(marker) + r" -->)",
        re.DOTALL,
    )
    new_html, n = pattern.subn(lambda m: m.group(1) + fragment + m.group(3), html)
    if n == 0:
        raise RuntimeError(f"marker {marker!r} not found in books.html")
    if n > 1:
        raise RuntimeError(f"marker {marker!r} appears {n} times — must be unique")
    return new_html


def patch_page(existing_html: str, books: pd.DataFrame, dnf: pd.DataFrame) -> str:
    fragments = build_fragments(books, dnf)
    html = existing_html
    for marker, key in MARKERS:
        html = patch_block(html, marker, fragments[key])
    return html


def has_markers(html: str) -> bool:
    """True if the page has at least the first marker we expect."""
    return "<!-- chart:per_year -->" in html


# ---------- full-page render (fallback only) ----------

def render_page(books: pd.DataFrame, dnf: pd.DataFrame) -> str:
    figs = {
        "per_year": chart_books_per_year(books),
        "grades": chart_grade_distribution(books),
        "genres": chart_genre_breakdown(books),
        "authors": chart_top_authors(books),
        "my_vs_gr": chart_my_vs_goodreads(books),
        "avg_by_genre": chart_avg_grade_by_genre(books),
        "cumulative": chart_cumulative(books),
        "heatmap": chart_year_genre_heatmap(books),
    }
    stats = headline_stats(books)
    stat_cards = "".join(
        f'<li><span class="stat-num">{escape(v)}</span><span class="stat-label">{escape(k)}</span></li>'
        for k, v in stats.items()
    )

    return f"""<!DOCTYPE HTML>
<html>
<head>
<title>Ryan Duncombe — Books</title>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
<link rel="stylesheet" href="assets/css/main.css" />
<noscript><link rel="stylesheet" href="assets/css/noscript.css" /></noscript>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
#wrapper > section.banner .content {{
    text-align: center;
    padding: 2rem 2.5rem 1rem;
}}
#wrapper > section.banner .content h1 {{ margin-top: 0; margin-bottom: 1rem; }}
#wrapper > section.banner .content .actions {{ justify-content: center; }}

.books-stats {{
    list-style: none; padding: 0; margin: 2rem 0 0;
    display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: center;
}}
.books-stats li {{
    padding: 1.25rem 1.75rem; min-width: 9rem;
    background: rgba(71, 211, 229, 0.08);
    border: 1px solid rgba(71, 211, 229, 0.35);
    border-radius: 4px; text-align: center;
}}
.books-stats .stat-num {{
    display: block; font-size: 2rem; font-weight: 700; color: {ACCENT_DARK};
    line-height: 1.1;
}}
.books-stats .stat-label {{
    display: block; font-size: 0.8rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: {MUTED}; margin-top: 0.4rem;
}}

.chart-section {{ padding-bottom: 1rem; }}
.chart-section .inner {{ max-width: 1100px; margin: 0 auto; padding: 3rem 2rem; }}
.chart-section + .chart-section .inner {{ padding-top: 0.5rem; }}
.chart-section h3 {{ margin-bottom: 0.25rem; }}
.chart-section .blurb {{ color: {MUTED}; margin-bottom: 1.25rem; }}

.books-table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
.books-table th, .books-table td {{
    padding: 0.55rem 0.75rem; border-bottom: 1px solid rgba(0,0,0,0.08);
    text-align: left; font-size: 0.95rem;
}}
.books-table th {{
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    font-size: 0.75rem; color: {MUTED}; border-bottom: 2px solid rgba(0,0,0,0.15);
}}
.books-table tbody tr:hover {{ background: rgba(71, 211, 229, 0.06); }}

.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body class="is-preload">
<div id="wrapper" class="divided">

  <section class="banner style2 orient-center content-align-center image-position-center onload-image-fade-in fullscreen">
    <div class="content">
      <h1>Books</h1>
      <p>A running log of nearly every book I've read since 2003 — with vibey letter grades, comparisons to Goodreads, and a few patterns I've found along the way.</p>
      <!-- stats -->
      <ul class="books-stats">{stat_cards}</ul>
      <!-- /stats -->
      <ul class="actions" style="justify-content:center; margin-top:2rem;">
        <li><a href="personal.html" class="button">Back to Personal</a></li>
        <li><a href="{escape(GOODREADS_URL)}" class="button">Raw spreadsheet</a></li>
      </ul>
    </div>
    <div class="image">
      <img src="images/books.jpg" alt="" style="object-position: center;" />
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>How much I read each year</h3>
      <p class="blurb">Reading dipped during grad school (2014–2021) and bounced back hard after.</p>
      <!-- chart:per_year -->
      {fig_div(figs["per_year"])}
      <!-- /chart:per_year -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Cumulative books over time</h3>
      <p class="blurb">Every book I've logged, plotted as a running total.</p>
      <!-- chart:cumulative -->
      {fig_div(figs["cumulative"])}
      <!-- /chart:cumulative -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Favorites bookshelf</h3>
      <p class="blurb">Every A+ book — the ones I'd press into a stranger's hands.</p>
      <!-- table:favorites -->
      {table_favorites(books)}
      <!-- /table:favorites -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>How I grade books</h3>
      <p class="blurb">My letter-grade distribution. I'm a soft grader — most books land between B− and A−.</p>
      <!-- chart:grades -->
      {fig_div(figs["grades"])}
      <!-- /chart:grades -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>What I read</h3>
      <p class="blurb">Genre breakdown across all logged books.</p>
      <!-- chart:genres -->
      {fig_div(figs["genres"])}
      <!-- /chart:genres -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Average rating by genre</h3>
      <p class="blurb">Within genres I've read at least 5 books from — which categories actually deliver for me.</p>
      <!-- chart:avg_by_genre -->
      {fig_div(figs["avg_by_genre"])}
      <!-- /chart:avg_by_genre -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Most-read authors</h3>
      <p class="blurb">The authors I keep coming back to. Excludes a few writers whose counts are inflated by long series (Card, Rowling, Yudkowsky, Liu, Corey, Mantel, Robinson).</p>
      <!-- chart:authors -->
      {fig_div(figs["authors"])}
      <!-- /chart:authors -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>My ratings vs. Goodreads</h3>
      <p class="blurb">Hover over a point to see the book. Books above the dashed line — I liked them more than the average Goodreads reader; below, less.</p>
      <!-- chart:my_vs_gr -->
      {fig_div(figs["my_vs_gr"])}
      <!-- /chart:my_vs_gr -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Genre × year</h3>
      <p class="blurb">A heatmap of what I was into when.</p>
      <!-- chart:heatmap -->
      {fig_div(figs["heatmap"])}
      <!-- /chart:heatmap -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Books I've re-read</h3>
      <p class="blurb">If I went back to it, it earned a spot here.</p>
      <!-- table:rereads -->
      {table_rereads(books)}
      <!-- /table:rereads -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <div class="two-col">
        <div>
          <h3>Where I rated higher than Goodreads</h3>
          <p class="blurb">My most contrarian loves (top 12).</p>
          <!-- table:diffs_high -->
          {table_top_diffs(books, "high")}
          <!-- /table:diffs_high -->
        </div>
        <div>
          <h3>Where I rated lower than Goodreads</h3>
          <p class="blurb">Beloved books that didn't land for me (top 12).</p>
          <!-- table:diffs_low -->
          {table_top_diffs(books, "low")}
          <!-- /table:diffs_low -->
        </div>
      </div>
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Books I didn't finish</h3>
      <p class="blurb">I don't often DNF, but when I do, I (sometimes) log it.</p>
      <!-- table:dnf -->
      {table_dnf(dnf)}
      <!-- /table:dnf -->
    </div>
  </section>

  <footer class="wrapper style1 align-center">
    <div class="inner">
      <ul class="icons">
        <li><a href="https://www.linkedin.com/in/ryan-duncombe/" class="icon brands style2 fa-linkedin-in"><span class="label">LinkedIn</span></a></li>
        <li><a href="mailto:ryduncombe@gmail.com" class="icon style2 fa-envelope"><span class="label">Email</span></a></li>
      </ul>
      <p><a href="personal.html">Back to Personal</a> &nbsp;&mdash;&nbsp; &copy; Ryan Duncombe.</p>
    </div>
  </footer>

</div>

<script src="assets/js/jquery.min.js"></script>
<script src="assets/js/jquery.scrollex.min.js"></script>
<script src="assets/js/jquery.scrolly.min.js"></script>
<script src="assets/js/browser.min.js"></script>
<script src="assets/js/breakpoints.min.js"></script>
<script src="assets/js/util.js"></script>
<script src="assets/js/main.js"></script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to booklist.xlsx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write books.html")
    args = parser.parse_args()

    books, dnf = load_data(args.input)

    if args.output.exists() and has_markers(args.output.read_text(encoding="utf-8")):
        existing = args.output.read_text(encoding="utf-8")
        html = patch_page(existing, books, dnf)
        mode = "patched"
    else:
        html = render_page(books, dnf)
        mode = "rendered fresh"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{mode} {args.output}  ({args.output.stat().st_size / 1024:.1f} KB, {len(books)} books)")


if __name__ == "__main__":
    main()
