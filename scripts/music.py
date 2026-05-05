"""Analyze the Last.fm listening history and emit a standalone HTML page.

Outputs personal-website/music.html. Re-run after the source CSV updates.
Prose lives in music.html (between sections, outside marker comments) and is
hand-edited; this script only patches the data-driven blocks between markers.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent  # personal-website/scripts/
SITE = ROOT.parent                       # personal-website/
DEFAULT_INPUT = SITE.parent / "Claude-personal" / "music" / "enriched_history.csv"
DEFAULT_OUTPUT = SITE / "music.html"
LASTFM_URL = "https://www.last.fm/user/rkd92"

# Site palette (from personal-website/assets/css/main.css)
ACCENT = "#47D3E5"
ACCENT_DARK = "#1ebdd1"
INK = "#1c1c1c"
MUTED = "#7f888f"

# How many top items to show in long-tail charts.
TOP_ARTISTS = 20
TOP_TRACKS = 20
TOP_COUNTRIES = 15
TOP_FINE_GENRES = 12  # for the fine-genre-over-time chart

# Coarse genre buckets, ordered most-specific first. For each tag in a track,
# the first bucket whose keyword appears as a substring of the tag wins; the
# track's coarse genre is the highest-priority bucket any of its tags hit.
GENRE_MAP: list[tuple[str, list[str]]] = [
    ("Classical", [
        "classical", "baroque", "romantic classical", "opera", "orchestral",
        "symphony", "chamber music", "choral", "modern classical",
        "renaissance", "early music", "gregorian", "cantata", "concerto",
    ]),
    ("Soundtrack", ["film score", "soundtrack", "video game"]),
    ("Metal", ["metal", "grindcore", "metalcore", "deathcore", "djent", "mathcore"]),
    ("Punk", ["punk", "hardcore punk", "post-hardcore", "emo", "screamo"]),
    ("Hip-Hop", ["hip hop", "hip-hop", "rap", "trap", "boom bap"]),
    ("Reggae", ["reggae", "dub", "ska", "dancehall"]),
    ("Jazz", ["jazz", "bebop", "swing", "free jazz", "fusion"]),
    ("Blues", ["blues"]),
    ("Country", ["country", "bluegrass", "honky tonk"]),
    ("Folk", ["folk", "americana", "singer-songwriter"]),
    ("Soul/R&B", ["soul", "r&b", "funk", "motown", "disco"]),
    ("Latin", ["latin", "salsa", "bossa nova", "samba", "tango", "reggaeton"]),
    ("World", [
        "afrobeat", "celtic", "nordic", "raga", "indian classical",
        "middle eastern", "asian", "african", "balkan",
    ]),
    ("Ambient", ["ambient","minimalism"]),
    ("Electronic", [
        "electronic", "electronica", "techno", "house",  "idm",
        "dubstep", "drum and bass", "synth", "edm", "downtempo", "trance",
    ]),
    ("Classic Rock", [
        "classic rock", "psychedelic rock", "progressive rock", "art rock",
        "krautrock", "southern rock", "glam rock", "arena rock"
    ]),
    ("Indie/Alt Rock", [
        "indie rock", "alternative rock", "indie", "alternative", "post-rock",
        "shoegaze", "math rock", "noise rock",
    ]),
    ("Pop", ["pop", "art pop", "pop rock", "power pop", "indie pop", "synth-pop", "dream pop"]),
    ("Rock", ["rock", "soft rock", "hard rock"]),  # final catchall
    ("Experimental", ["experimental", "avant-garde", "noise", "drone"]),
]
COARSE_OTHER = "Other"

# Tags excluded from the FINE genre charts. Last.fm/MusicBrainz tags include
# a lot of noise — places, decades, artist descriptors, languages. Add to this
# set whenever you spot a tag that pollutes the chart.
GENRE_BLACKLIST: set[str] = {
    # Geography
    "american", "british", "german", "english", "swedish", "european",
    "united states", "norwegian", "finnish", "icelandic", "danish", "french",
    "italian", "spanish", "japanese", "polish", "russian", "australian",
    "canadian", "irish", "scottish", "dutch", "greek", "hungarian", "czech",
    "austrian", "brazilian", "mexican", "argentinian", "chilean",
    "santa barbara", "isla vista", "ucsb", "los angeles", "new york",
    "berlin", "london", "tokyo", "paris", "chicago",
    # Decade tags handled by regex below
    # Descriptors that aren't genres
    "instrumental", "composer", "pianist", "guitarist", "vocalist", "violinist",
    "soundtrack",
    # MusicBrainz noise
    "vyrzukhisuc-artiest",
}
_DECADE_RE = re.compile(r"^\d+0s$")  # 60s, 70s, 1990s, 2010s
_YEAR_RE = re.compile(r"^\d{4}$")


def is_blacklisted(tag: str) -> bool:
    return tag in GENRE_BLACKLIST or bool(_DECADE_RE.match(tag) or _YEAR_RE.match(tag))


def coarse_genre(tags: list[str]) -> str:
    for label, keywords in GENRE_MAP:
        for tag in tags:
            for kw in keywords:
                if kw in tag:
                    return label
    return COARSE_OTHER


# ---------- shared layout ----------

def _apply_layout(fig: go.Figure, *, height: int = 420) -> go.Figure:
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


def fig_div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})


# ---------- data loading ----------

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["dt"] = pd.to_datetime(df["utc_time"], format="%d %b %Y, %H:%M", errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.to_period("M").dt.to_timestamp()

    # Convert UTC timestamps to approximate local time.
    # Pre-2022: America/Chicago (Chicago years); 2022+: America/Los_Angeles (Pacifica).
    # Travel periods will still be slightly off — unavoidable without GPS data.
    dt_utc = pd.to_datetime(df["uts"], unit="s", utc=True)
    pre = df["year"] < 2022
    local_chi = dt_utc[pre].dt.tz_convert("America/Chicago")
    local_la  = dt_utc[~pre].dt.tz_convert("America/Los_Angeles")
    df["hour"] = pd.concat([local_chi.dt.hour, local_la.dt.hour]).sort_index()
    df["dow"]  = pd.concat([local_chi.dt.dayofweek, local_la.dt.dayofweek]).sort_index()

    # Tokenize genres: split on '|', strip, lowercase. NaN → empty list.
    def tokenize(g):
        if not isinstance(g, str):
            return []
        return [t.strip().lower() for t in g.split("|") if t.strip()]
    df["tags"] = df["genres"].map(tokenize)
    df["coarse"] = df["tags"].map(lambda ts: coarse_genre(ts) if ts else COARSE_OTHER)
    return df


# ---------- chart builders ----------

def chart_plays_per_year(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("year").size().reindex(range(df["year"].min(), df["year"].max() + 1), fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>%{y:,} plays<extra></extra>",
    ))
    fig.update_layout(title="Plays per year", xaxis_title=None, yaxis_title="Plays")
    return _apply_layout(fig)


def chart_cumulative(df: pd.DataFrame) -> go.Figure:
    s = df.sort_values("dt")
    cum = pd.Series(range(1, len(s) + 1), index=s["dt"])
    # Downsample to daily for plot performance — same shape, fewer points.
    daily = cum.resample("D").max().ffill()
    fig = go.Figure(go.Scatter(
        x=daily.index, y=daily.values, mode="lines",
        line=dict(color=ACCENT_DARK, width=2.5),
        fill="tozeroy", fillcolor="rgba(71, 211, 229, 0.18)",
        hovertemplate="%{x|%b %Y}<br><b>%{y:,}</b> plays<extra></extra>",
    ))
    fig.update_layout(title="Cumulative plays over time", xaxis_title=None, yaxis_title="Total plays")
    return _apply_layout(fig)


def chart_listening_hour(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("hour").size().reindex(range(24), fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        hovertemplate="<b>%{x}:00</b><br>%{y:,} plays<extra></extra>",
    ))
    fig.update_layout(
        title="What time of day do I listen?",
        xaxis=dict(title="Hour of day (local time)", dtick=1),
        yaxis_title="Plays",
    )
    return _apply_layout(fig)


def chart_hour_dow_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="dow", columns="hour", values="track", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=days,
        colorscale=[[0, "rgba(71,211,229,0.05)"], [1, ACCENT_DARK]],
        hovertemplate="<b>%{y} %{x}:00</b><br>%{z:,} plays<extra></extra>",
        colorbar=dict(title="Plays", thickness=12),
    ))
    fig.update_layout(
        title="When I listen (hour × day of week)",
        xaxis=dict(title="Hour of day (local time)", dtick=1),
        yaxis=dict(title=None, autorange="reversed"),
    )
    return _apply_layout(fig, height=420)


def chart_top_artists_alltime(df: pd.DataFrame) -> go.Figure:
    counts = df["artist"].value_counts().head(TOP_ARTISTS).sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=[f"{n:,}" for n in counts.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} plays<extra></extra>",
    ))
    fig.update_layout(title=f"Top {TOP_ARTISTS} artists, all-time", xaxis_title="Plays", yaxis_title=None)
    return _apply_layout(fig, height=560)


def chart_top_tracks_alltime(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["track", "artist"]).size().nlargest(TOP_TRACKS).sort_values()
    labels = [f"{t} — {a}" for (t, a) in grp.index]
    fig = go.Figure(go.Bar(
        x=grp.values, y=labels, orientation="h",
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=[f"{n:,}" for n in grp.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} plays<extra></extra>",
    ))
    fig.update_layout(title=f"Top {TOP_TRACKS} tracks, all-time", xaxis_title="Plays", yaxis_title=None)
    return _apply_layout(fig, height=560)


def _by_year_dropdown(df: pd.DataFrame, *, group_cols: list[str], top_n: int,
                      title_template: str) -> go.Figure:
    """Build a single chart with a year dropdown that swaps which trace is visible."""
    years = sorted(df["year"].unique())
    default_idx = len(years) - 1  # latest year visible by default
    fig = go.Figure()
    buttons = []
    for i, year in enumerate(years):
        ydata = df[df["year"] == year]
        grp = ydata.groupby(group_cols).size().nlargest(top_n).sort_values()
        if len(group_cols) == 1:
            ylabels = list(grp.index)
        else:
            ylabels = [f"{a} — {b}" for (a, b) in grp.index]
        fig.add_trace(go.Bar(
            x=grp.values, y=ylabels, orientation="h",
            marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
            text=[f"{n:,}" for n in grp.values], textposition="outside",
            visible=(i == default_idx), name=str(year),
            hovertemplate="<b>%{y}</b><br>%{x:,} plays<extra></extra>",
        ))
        visibility = [j == i for j in range(len(years))]
        buttons.append(dict(
            label=str(year), method="update",
            args=[{"visible": visibility}, {"title": title_template.format(year=year)}],
        ))
    fig.update_layout(
        title=title_template.format(year=years[default_idx]),
        xaxis_title="Plays", yaxis_title=None,
        updatemenus=[dict(
            active=default_idx, buttons=buttons, direction="down",
            x=1.0, xanchor="right", y=1.18, yanchor="top",
            bgcolor="white", bordercolor="rgba(0,0,0,0.15)",
        )],
    )
    return _apply_layout(fig, height=540)


def chart_top_artists_by_year(df: pd.DataFrame) -> go.Figure:
    return _by_year_dropdown(df, group_cols=["artist"], top_n=10,
                              title_template="Top artists in {year}")


def chart_top_tracks_by_year(df: pd.DataFrame) -> go.Figure:
    return _by_year_dropdown(df, group_cols=["track", "artist"], top_n=10,
                              title_template="Top tracks in {year}")


def _stacked_area_genre(df: pd.DataFrame, genre_col: str, *, title: str,
                        top_n: int | None = None) -> go.Figure:
    counts = df.groupby(["year", genre_col]).size().reset_index(name="plays")
    if top_n is not None:
        # Keep only the top_n genres by total plays; bucket the rest into "Other".
        totals = counts.groupby(genre_col)["plays"].sum().nlargest(top_n).index
        counts.loc[~counts[genre_col].isin(totals), genre_col] = "other"
    pivot = counts.pivot_table(index="year", columns=genre_col, values="plays", aggfunc="sum", fill_value=0)
    # Order genres by total descending so the legend reads "biggest first".
    order = pivot.sum().sort_values(ascending=False).index.tolist()
    pivot = pivot[order]
    fig = go.Figure()
    for name in order:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[name], name=name,
            mode="lines", stackgroup="one",
            hovertemplate="<b>%{x}</b><br>" + name + ": %{y:,} plays<extra></extra>",
        ))
    fig.update_layout(title=title, xaxis_title=None, yaxis_title="Plays",
                      legend=dict(font=dict(size=11)))
    return _apply_layout(fig, height=520)


def chart_coarse_genre_over_time(df: pd.DataFrame) -> go.Figure:
    return _stacked_area_genre(df, "coarse", title="Genres over time (coarse)")


def chart_fine_genre_over_time(df: pd.DataFrame) -> go.Figure:
    """Each play contributes to every (non-blacklisted) tag it has."""
    rows = []
    for year, tags in zip(df["year"], df["tags"]):
        for tag in tags:
            if not is_blacklisted(tag):
                rows.append((year, tag))
    if not rows:
        return _apply_layout(go.Figure(), height=520)
    expanded = pd.DataFrame(rows, columns=["year", "tag"])
    return _stacked_area_genre(expanded, "tag",
                               title=f"Top {TOP_FINE_GENRES} fine genres over time",
                               top_n=TOP_FINE_GENRES)


def chart_discovery(df: pd.DataFrame) -> go.Figure:
    first_listen = df.groupby("artist")["dt"].min()
    new_per_year = first_listen.dt.year.value_counts().sort_index()
    new_per_year = new_per_year.reindex(
        range(new_per_year.index.min(), new_per_year.index.max() + 1), fill_value=0,
    )
    fig = go.Figure(go.Bar(
        x=new_per_year.index, y=new_per_year.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>%{y:,} new artists<extra></extra>",
    ))
    fig.update_layout(title="New artists discovered each year",
                      xaxis_title=None, yaxis_title="New artists")
    return _apply_layout(fig)


def chart_era(df: pd.DataFrame) -> go.Figure:
    s = df.dropna(subset=["release_year"]).copy()
    s["decade"] = (s["release_year"] // 10 * 10).astype(int)
    counts = s.groupby("decade").size()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 10, 10), fill_value=0)
    labels = [f"{d}s" for d in counts.index]
    fig = go.Figure(go.Bar(
        x=labels, y=counts.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>%{y:,} plays<extra></extra>",
    ))
    fig.update_layout(title="What era is the music from?",
                      xaxis_title="Release decade", yaxis_title="Plays")
    return _apply_layout(fig)


def chart_artist_countries(df: pd.DataFrame) -> go.Figure:
    counts = df["artist_country"].dropna().value_counts().head(TOP_COUNTRIES).sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        text=[f"{n:,}" for n in counts.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} plays<extra></extra>",
    ))
    fig.update_layout(title=f"Top {TOP_COUNTRIES} artist countries (ISO codes)",
                      xaxis_title="Plays", yaxis_title=None)
    return _apply_layout(fig, height=520)


def chart_artist_diversity(df: pd.DataFrame) -> go.Figure:
    diversity = df.groupby("year")["artist"].nunique()
    fig = go.Figure(go.Scatter(
        x=diversity.index, y=diversity.values, mode="lines+markers",
        line=dict(color=ACCENT_DARK, width=2.5),
        marker=dict(size=8, color=ACCENT, line=dict(color=ACCENT_DARK, width=1)),
        hovertemplate="<b>%{x}</b><br>%{y:,} unique artists<extra></extra>",
    ))
    fig.update_layout(title="Unique artists each year",
                      xaxis_title=None, yaxis_title="Unique artists")
    return _apply_layout(fig)


def chart_playcount_buckets(df: pd.DataFrame) -> go.Figure:
    song_plays = df.groupby(["artist", "track"]).size()
    total_plays = song_plays.sum()
    max_plays = int(song_plays.max())

    bins   = [0, 1, 5, 20, 100, float("inf")]
    labels = ["Heard once", "2–5×", "6–20×", "21–100×", "100+×"]
    buckets = pd.cut(song_plays, bins=bins, labels=labels, right=True)

    song_counts  = buckets.value_counts().reindex(labels, fill_value=0)
    listen_pcts  = song_plays.groupby(buckets).sum().reindex(labels, fill_value=0) / total_plays * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Unique songs", x=labels, y=song_counts.values,
        marker_color=ACCENT, marker_line_color=ACCENT_DARK, marker_line_width=1,
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>%{y:,} songs (%{customdata:.1f}% of listens)<extra></extra>",
        customdata=listen_pcts.values,
    ))
    fig.add_trace(go.Scatter(
        name="% of total listens", x=labels, y=listen_pcts.values,
        mode="lines+markers", yaxis="y2",
        line=dict(color=ACCENT_DARK, width=2.5),
        marker=dict(size=8, color=ACCENT_DARK),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% of all listens<extra></extra>",
    ))
    fig.update_layout(
        title="How often do I replay songs?",
        xaxis_title=None,
        yaxis=dict(title="Unique songs", gridcolor="rgba(0,0,0,0.08)", linecolor="rgba(0,0,0,0.2)"),
        yaxis2=dict(title="% of total listens", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False, linecolor="rgba(0,0,0,0.2)"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        barmode="group",
    )
    return _apply_layout(fig)


def chart_pareto(df: pd.DataFrame) -> go.Figure:
    song_plays = df.groupby(["artist", "track"]).size().sort_values(ascending=False)
    total_plays = song_plays.sum()
    total_songs = len(song_plays)

    cum_plays_pct = (song_plays.cumsum() / total_plays * 100).values
    songs_pct = np.linspace(0, 100, total_songs, endpoint=False)

    # Downsample to ~2000 points for performance without losing curve shape.
    if total_songs > 2000:
        idx = np.unique(np.round(np.linspace(0, total_songs - 1, 2000)).astype(int))
        songs_pct_plot = songs_pct[idx]
        cum_plays_pct_plot = cum_plays_pct[idx]
    else:
        songs_pct_plot, cum_plays_pct_plot = songs_pct, cum_plays_pct

    # Find x% of songs that account for 80% of listens.
    cutoff_idx = int(np.searchsorted(cum_plays_pct, 80))
    cutoff_pct = songs_pct[min(cutoff_idx, total_songs - 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=songs_pct_plot, y=cum_plays_pct_plot, mode="lines",
        line=dict(color=ACCENT_DARK, width=2.5),
        fill="tozeroy", fillcolor="rgba(71, 211, 229, 0.18)",
        hovertemplate="Top %{x:.1f}% of songs<br><b>%{y:.1f}%</b> of listens<extra></extra>",
        name="Cumulative listens",
    ))
    # Reference: 80% line
    fig.add_hline(y=80, line_dash="dash", line_color=MUTED, line_width=1,
                  annotation_text=f"80% of listens → top {cutoff_pct:.1f}% of songs",
                  annotation_position="top right",
                  annotation_font=dict(size=12, color=MUTED))
    fig.update_layout(
        title="Pareto: how concentrated are my listens?",
        xaxis=dict(title="Top X% of songs (by play count)", ticksuffix="%",
                   showgrid=False, linecolor="rgba(0,0,0,0.2)"),
        yaxis=dict(title="Cumulative % of total listens", ticksuffix="%",
                   gridcolor="rgba(0,0,0,0.08)", linecolor="rgba(0,0,0,0.2)"),
        showlegend=False,
    )
    return _apply_layout(fig)


# ---------- headline stats ----------

def headline_stats(df: pd.DataFrame) -> dict[str, str]:
    return {
        "Total plays": f"{len(df):,}",
        "Years tracked": str(df["year"].nunique()),
        "Unique artists": f"{df['artist'].nunique():,}",
        "Unique tracks": f"{df.groupby(['artist','track']).ngroups:,}",
        "Top artist": df["artist"].value_counts().idxmax(),
    }


# ---------- patch mode ----------

MARKERS: list[tuple[str, str]] = [
    ("stats", "stats"),
    ("chart:plays_per_year", "plays_per_year"),
    ("chart:cumulative", "cumulative"),
    ("chart:top_artists_alltime", "top_artists_alltime"),
    ("chart:top_artists_by_year", "top_artists_by_year"),
    ("chart:top_tracks_alltime", "top_tracks_alltime"),
    ("chart:top_tracks_by_year", "top_tracks_by_year"),
    ("chart:coarse_genre", "coarse_genre"),
    ("chart:fine_genre", "fine_genre"),
    ("chart:discovery", "discovery"),
    ("chart:listening_hour", "listening_hour"),
    ("chart:hour_dow", "hour_dow"),
    ("chart:era", "era"),
    ("chart:countries", "countries"),
    ("chart:diversity", "diversity"),
    ("chart:playcount_buckets", "playcount_buckets"),
    ("chart:pareto", "pareto"),
]


def build_fragments(df: pd.DataFrame) -> dict[str, str]:
    figs = {
        "plays_per_year": chart_plays_per_year(df),
        "cumulative": chart_cumulative(df),
        "top_artists_alltime": chart_top_artists_alltime(df),
        "top_artists_by_year": chart_top_artists_by_year(df),
        "top_tracks_alltime": chart_top_tracks_alltime(df),
        "top_tracks_by_year": chart_top_tracks_by_year(df),
        "coarse_genre": chart_coarse_genre_over_time(df),
        "fine_genre": chart_fine_genre_over_time(df),
        "discovery": chart_discovery(df),
        "listening_hour": chart_listening_hour(df),
        "hour_dow": chart_hour_dow_heatmap(df),
        "era": chart_era(df),
        "countries": chart_artist_countries(df),
        "diversity": chart_artist_diversity(df),
        "playcount_buckets": chart_playcount_buckets(df),
        "pareto": chart_pareto(df),
    }
    stats = headline_stats(df)
    stat_cards = "".join(
        f'<li><span class="stat-num">{escape(v)}</span><span class="stat-label">{escape(k)}</span></li>'
        for k, v in stats.items()
    )
    return {"stats": f'<ul class="music-stats">{stat_cards}</ul>',
            **{k: fig_div(v) for k, v in figs.items()}}


def patch_block(html: str, marker: str, fragment: str) -> str:
    pattern = re.compile(
        r"(<!-- " + re.escape(marker) + r" -->\n)(.*?)(\n[ \t]*<!-- /" + re.escape(marker) + r" -->)",
        re.DOTALL,
    )
    new_html, n = pattern.subn(lambda m: m.group(1) + fragment + m.group(3), html)
    if n == 0:
        raise RuntimeError(f"marker {marker!r} not found in music.html")
    if n > 1:
        raise RuntimeError(f"marker {marker!r} appears {n} times — must be unique")
    return new_html


def patch_page(existing_html: str, df: pd.DataFrame) -> str:
    fragments = build_fragments(df)
    html = existing_html
    for marker, key in MARKERS:
        html = patch_block(html, marker, fragments[key])
    return html


def has_markers(html: str) -> bool:
    return "<!-- chart:plays_per_year -->" in html


# ---------- full-page render (fallback only) ----------

def render_page(df: pd.DataFrame) -> str:
    fragments = build_fragments(df)
    return f"""<!DOCTYPE HTML>
<html>
<head>
<title>Ryan Duncombe — Music</title>
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

.music-stats {{
    list-style: none; padding: 0; margin: 2rem 0 0;
    display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: center;
}}
.music-stats li {{
    padding: 1.25rem 1.75rem; min-width: 9rem;
    background: rgba(71, 211, 229, 0.08);
    border: 1px solid rgba(71, 211, 229, 0.35);
    border-radius: 4px; text-align: center;
}}
.music-stats .stat-num {{
    display: block; font-size: 2rem; font-weight: 700; color: {ACCENT_DARK};
    line-height: 1.1;
}}
.music-stats .stat-label {{
    display: block; font-size: 0.8rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: {MUTED}; margin-top: 0.4rem;
}}

.chart-section {{ padding-bottom: 1rem; }}
.chart-section .inner {{ max-width: 1100px; margin: 0 auto; padding: 3rem 2rem; }}
.chart-section + .chart-section .inner {{ padding-top: 0.5rem; }}
.chart-section h3 {{ margin-bottom: 0.25rem; }}
.chart-section .blurb {{ color: {MUTED}; margin-bottom: 1.25rem; }}
</style>
</head>
<body class="is-preload">
<div id="wrapper" class="divided">

  <section class="banner style2 orient-center content-align-center image-position-center onload-image-fade-in fullscreen">
    <div class="content">
      <h1>Music</h1>
      <!-- stats -->
      {fragments["stats"]}
      <!-- /stats -->
      <ul class="actions" style="justify-content:center; margin-top:2rem;">
        <li><a href="personal.html" class="button">Back to Personal</a></li>
        <li><a href="{escape(LASTFM_URL)}" class="button">My Last.fm</a></li>
      </ul>
    </div>
    <div class="image">
      <img src="images/music.JPEG" alt="" style="object-position: center;" />
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Plays per year</h3>
      <!-- chart:plays_per_year -->
      {fragments["plays_per_year"]}
      <!-- /chart:plays_per_year -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Cumulative plays over time</h3>
      <!-- chart:cumulative -->
      {fragments["cumulative"]}
      <!-- /chart:cumulative -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Top artists, all-time</h3>
      <!-- chart:top_artists_alltime -->
      {fragments["top_artists_alltime"]}
      <!-- /chart:top_artists_alltime -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Top artists by year</h3>
      <!-- chart:top_artists_by_year -->
      {fragments["top_artists_by_year"]}
      <!-- /chart:top_artists_by_year -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Top tracks, all-time</h3>
      <!-- chart:top_tracks_alltime -->
      {fragments["top_tracks_alltime"]}
      <!-- /chart:top_tracks_alltime -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Top tracks by year</h3>
      <!-- chart:top_tracks_by_year -->
      {fragments["top_tracks_by_year"]}
      <!-- /chart:top_tracks_by_year -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Genres over time (coarse)</h3>
      <!-- chart:coarse_genre -->
      {fragments["coarse_genre"]}
      <!-- /chart:coarse_genre -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Genres over time (fine)</h3>
      <!-- chart:fine_genre -->
      {fragments["fine_genre"]}
      <!-- /chart:fine_genre -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>New artists each year</h3>
      <!-- chart:discovery -->
      {fragments["discovery"]}
      <!-- /chart:discovery -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>What time of day do I listen?</h3>
      <!-- chart:listening_hour -->
      {fragments["listening_hour"]}
      <!-- /chart:listening_hour -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>When I listen (hour × day)</h3>
      <!-- chart:hour_dow -->
      {fragments["hour_dow"]}
      <!-- /chart:hour_dow -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>What era is the music from?</h3>
      <!-- chart:era -->
      {fragments["era"]}
      <!-- /chart:era -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Where the artists are from</h3>
      <!-- chart:countries -->
      {fragments["countries"]}
      <!-- /chart:countries -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Artist diversity, year by year</h3>
      <!-- chart:diversity -->
      {fragments["diversity"]}
      <!-- /chart:diversity -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>How often do I replay songs?</h3>
      <!-- chart:playcount_buckets -->
      {fragments["playcount_buckets"]}
      <!-- /chart:playcount_buckets -->
    </div>
  </section>

  <section class="wrapper style1 align-center chart-section">
    <div class="inner">
      <h3>Pareto: how concentrated are my listens?</h3>
      <!-- chart:pareto -->
      {fragments["pareto"]}
      <!-- /chart:pareto -->
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to enriched_history.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write music.html")
    args = parser.parse_args()

    df = load_data(args.input)

    if args.output.exists() and has_markers(args.output.read_text(encoding="utf-8")):
        existing = args.output.read_text(encoding="utf-8")
        html = patch_page(existing, df)
        mode = "patched"
    else:
        html = render_page(df)
        mode = "rendered fresh"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{mode} {args.output}  ({args.output.stat().st_size / 1024:.1f} KB, {len(df):,} plays)")


if __name__ == "__main__":
    main()
