from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_FILES = {
    "2023-24": "PL-season-2324.csv",
    "2024-25": "PL-season-2425.csv",
}

REQUIRED_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
}

METRIC_OPTIONS = {
    "Goals": "Goals",
    "Shots": "Shots",
    "ShotsOT": "Shots on Target",
    "Corners": "Corners",
}


def _parse_dates(date_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_series, format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            date_series.loc[missing_mask], format="%d/%m/%y", errors="coerce"
        )
    if parsed.isna().any():
        raise ValueError("Some match dates could not be parsed.")
    return parsed


@st.cache_data(show_spinner=False)
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent
    frames: list[pd.DataFrame] = []

    for season, filename in DATA_FILES.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        season_df = pd.read_csv(path)
        missing_cols = REQUIRED_COLUMNS.difference(season_df.columns)
        if missing_cols:
            raise ValueError(
                f"{filename} is missing required columns: {sorted(missing_cols)}"
            )
        season_df["Season"] = season
        frames.append(season_df)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = _parse_dates(df["Date"])

    home = df.assign(
        Team=df["HomeTeam"],
        Goals=df["FTHG"],
        Points=df["FTR"].map({"H": 3, "D": 1, "A": 0}),
        Shots=df["HS"],
        ShotsOT=df["HST"],
        Corners=df["HC"],
    )[
        [
            "Date",
            "Season",
            "Team",
            "Goals",
            "Points",
            "Shots",
            "ShotsOT",
            "Corners",
        ]
    ]

    away = df.assign(
        Team=df["AwayTeam"],
        Goals=df["FTAG"],
        Points=df["FTR"].map({"A": 3, "D": 1, "H": 0}),
        Shots=df["AS"],
        ShotsOT=df["AST"],
        Corners=df["AC"],
    )[
        [
            "Date",
            "Season",
            "Team",
            "Goals",
            "Points",
            "Shots",
            "ShotsOT",
            "Corners",
        ]
    ]

    team_matches = (
        pd.concat([home, away]).sort_values(["Season", "Team", "Date"]).reset_index(drop=True)
    )
    team_matches["GameNum"] = team_matches.groupby(["Season", "Team"]).cumcount() + 1

    home_pts = (
        df.assign(Team=df["HomeTeam"], Points=df["FTR"].map({"H": 3, "D": 1, "A": 0}))
        .groupby(["Season", "Team"], as_index=False)["Points"]
        .sum()
        .rename(columns={"Points": "HomePoints"})
    )
    away_pts = (
        df.assign(Team=df["AwayTeam"], Points=df["FTR"].map({"A": 3, "D": 1, "H": 0}))
        .groupby(["Season", "Team"], as_index=False)["Points"]
        .sum()
        .rename(columns={"Points": "AwayPoints"})
    )
    team_ha = home_pts.merge(away_pts, on=["Season", "Team"])

    matches = df[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    matches["TotalGoals"] = matches["FTHG"] + matches["FTAG"]
    matches["MatchLabel"] = (
        matches["HomeTeam"]
        + " "
        + matches["FTHG"].astype(str)
        + "-"
        + matches["FTAG"].astype(str)
        + " "
        + matches["AwayTeam"]
    )

    return team_matches, team_ha, matches


def make_home_away_chart(team_ha: pd.DataFrame, selected_team: str | None) -> alt.Chart:
    axis_max = int(max(team_ha["HomePoints"].max(), team_ha["AwayPoints"].max()) + 2)
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, axis_max], "y": [0, axis_max]}))
        .mark_line(strokeDash=[5, 3], color="gray", opacity=0.4)
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, axis_max])), y="y:Q")
    )

    if not selected_team:
        points = (
            alt.Chart(team_ha)
            .mark_circle(size=95, opacity=0.82)
            .encode(
                x=alt.X("HomePoints:Q", title="Home Points"),
                y=alt.Y("AwayPoints:Q", title="Away Points"),
                color=alt.Color("Season:N", title="Season"),
                tooltip=["Team:N", "Season:N", "HomePoints:Q", "AwayPoints:Q"],
            )
        )
        return alt.layer(diagonal, points).properties(
            title="Scene 1: Home vs Away Points",
            height=350,
        )

    selected_df = team_ha[team_ha["Team"] == selected_team]
    other_df = team_ha[team_ha["Team"] != selected_team]

    muted = (
        alt.Chart(other_df)
        .mark_circle(size=70, color="lightgray", opacity=0.35)
        .encode(
            x=alt.X("HomePoints:Q", title="Home Points"),
            y=alt.Y("AwayPoints:Q", title="Away Points"),
            tooltip=["Team:N", "Season:N", "HomePoints:Q", "AwayPoints:Q"],
        )
    )
    highlight = (
        alt.Chart(selected_df)
        .mark_circle(size=140, opacity=0.95)
        .encode(
            x=alt.X("HomePoints:Q", title="Home Points"),
            y=alt.Y("AwayPoints:Q", title="Away Points"),
            color=alt.Color("Season:N", title="Season"),
            tooltip=["Team:N", "Season:N", "HomePoints:Q", "AwayPoints:Q"],
        )
    )

    return alt.layer(diagonal, muted, highlight).properties(
        title=f"Scene 1: Home vs Away Points ({selected_team} highlighted)",
        height=350,
    )


def make_trend_chart(team_matches: pd.DataFrame, selected_team: str, metric: str) -> alt.Chart:
    team_df = team_matches[team_matches["Team"] == selected_team].copy()
    team_df = team_df.sort_values(["Season", "GameNum"])
    team_df["RollingAvg"] = team_df.groupby("Season")[metric].transform(
        lambda values: values.rolling(window=5, min_periods=1).mean()
    )

    return (
        alt.Chart(team_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("GameNum:Q", title="Game Number"),
            y=alt.Y(
                "RollingAvg:Q",
                title=f"5-Game Rolling Avg ({METRIC_OPTIONS[metric]})",
            ),
            color=alt.Color("Season:N", title="Season"),
            tooltip=[
                alt.Tooltip("Season:N"),
                alt.Tooltip("GameNum:Q", title="Game"),
                alt.Tooltip("RollingAvg:Q", title="Rolling Avg", format=".2f"),
            ],
        )
        .properties(
            title=f"Scene 2: How {selected_team}'s attacking output evolves",
            height=320,
        )
    )


def make_extreme_match_chart(matches: pd.DataFrame) -> alt.Chart:
    brush = alt.selection_interval()

    scatter = (
        alt.Chart(matches)
        .mark_circle(size=58)
        .encode(
            x=alt.X("FTHG:Q", title="Home Goals", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("FTAG:Q", title="Away Goals", axis=alt.Axis(tickMinStep=1)),
            color=alt.condition(brush, alt.Color("Season:N", title="Season"), alt.value("lightgray")),
            opacity=alt.condition(brush, alt.value(0.9), alt.value(0.3)),
            tooltip=["HomeTeam:N", "AwayTeam:N", "FTHG:Q", "FTAG:Q", "Season:N"],
        )
        .add_params(brush)
        .properties(
            title="Scene 3A: Match scorelines (brush a region)",
            width=340,
            height=300,
        )
    )

    detail = (
        alt.Chart(matches)
        .transform_filter(brush)
        .transform_window(rank="rank()", sort=[{"field": "TotalGoals", "order": "descending"}])
        .transform_filter("datum.rank <= 15")
        .mark_bar()
        .encode(
            x=alt.X("TotalGoals:Q", title="Total Goals"),
            y=alt.Y("MatchLabel:N", sort="-x", title=""),
            color=alt.Color("Season:N", title="Season"),
            tooltip=["HomeTeam:N", "AwayTeam:N", "FTHG:Q", "FTAG:Q", "TotalGoals:Q"],
        )
        .properties(
            title="Scene 3B: Top 15 brushed matches by total goals",
            width=470,
            height=300,
        )
    )

    return alt.hconcat(scatter, detail).resolve_scale(color="independent")


def make_linked_dashboard(
    team_matches: pd.DataFrame, team_ha: pd.DataFrame, matches: pd.DataFrame
) -> alt.Chart:
    team_sel = alt.selection_point(fields=["Team"], empty=False)
    brush = alt.selection_interval()
    metric_param = alt.param(
        name="metric",
        value="Goals",
        bind=alt.binding_select(
            options=list(METRIC_OPTIONS.keys()),
            labels=list(METRIC_OPTIONS.values()),
            name="Metric: ",
        ),
    )

    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 45], "y": [0, 45]}))
        .mark_line(strokeDash=[5, 3], color="gray", opacity=0.35)
        .encode(x="x:Q", y="y:Q")
    )

    q3 = (
        alt.layer(
            diag,
            alt.Chart(team_ha)
            .mark_circle(size=90)
            .encode(
                x=alt.X("HomePoints:Q", title="Home Points"),
                y=alt.Y("AwayPoints:Q", title="Away Points"),
                color=alt.condition(team_sel, alt.Color("Season:N"), alt.value("lightgray")),
                opacity=alt.condition(team_sel, alt.value(0.9), alt.value(0.45)),
                tooltip=["Team:N", "Season:N", "HomePoints:Q", "AwayPoints:Q"],
            )
            .add_params(team_sel),
        )
        .properties(
            title="Q3: Home vs Away Points (click a team)",
            width=340,
            height=290,
        )
    )

    q2 = (
        alt.Chart(team_matches)
        .transform_filter(team_sel)
        .transform_fold(list(METRIC_OPTIONS.keys()), as_=["Metric", "Value"])
        .transform_filter("datum.Metric === metric")
        .transform_window(
            RollingAvg="mean(Value)",
            frame=[-4, 0],
            sort=[{"field": "GameNum"}],
            groupby=["Season", "Team", "Metric"],
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("GameNum:Q", title="Game Number"),
            y=alt.Y("RollingAvg:Q", title="5-Game Rolling Average"),
            color=alt.Color("Season:N"),
            detail="Team:N",
            tooltip=[
                alt.Tooltip("Team:N"),
                alt.Tooltip("Season:N"),
                alt.Tooltip("GameNum:Q", title="Game"),
                alt.Tooltip("RollingAvg:Q", title="Rolling Avg", format=".2f"),
            ],
        )
        .add_params(metric_param)
        .properties(
            title="Q2: Team attacking trend over time",
            width=450,
            height=290,
        )
    )

    q4_scatter = (
        alt.Chart(matches)
        .mark_circle(size=55)
        .encode(
            x=alt.X("FTHG:Q", title="Home Goals", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("FTAG:Q", title="Away Goals", axis=alt.Axis(tickMinStep=1)),
            color=alt.condition(brush, alt.Color("Season:N"), alt.value("lightgray")),
            opacity=alt.condition(brush, alt.value(0.85), alt.value(0.25)),
            tooltip=["HomeTeam:N", "AwayTeam:N", "FTHG:Q", "FTAG:Q", "Season:N"],
        )
        .add_params(brush)
        .properties(
            title="Q4: Match outcomes (brush points)",
            width=340,
            height=290,
        )
    )

    q4_detail = (
        alt.Chart(matches)
        .transform_filter(brush)
        .transform_window(rank="rank()", sort=[{"field": "TotalGoals", "order": "descending"}])
        .transform_filter("datum.rank <= 15")
        .mark_bar()
        .encode(
            x=alt.X("TotalGoals:Q", title="Total Goals"),
            y=alt.Y("MatchLabel:N", sort="-x", title=""),
            color=alt.Color("Season:N"),
            tooltip=["HomeTeam:N", "AwayTeam:N", "FTHG:Q", "FTAG:Q", "TotalGoals:Q"],
        )
        .properties(
            title="Q4 Detail: Top 15 brushed matches",
            width=450,
            height=290,
        )
    )

    return (
        alt.vconcat(
            alt.hconcat(q3, q2).resolve_scale(color="independent"),
            alt.hconcat(q4_scatter, q4_detail).resolve_scale(color="independent"),
        )
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=12, fontWeight="normal")
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
    )


def main() -> None:
    st.set_page_config(
        page_title="HW4: Football Narrative Visualization",
        layout="wide",
    )

    st.title("Do teams turn attacking pressure into points consistently?")
    st.markdown(
        """
This story compares two Premier League seasons (2023-24 and 2024-25) to answer a
central question: **which teams convert attacking volume into reliable points, and
how does that change by venue and over time?**
"""
    )

    try:
        team_matches, team_ha, matches = load_and_prepare_data()
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to load data: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{len(matches):,}")
    c2.metric("Team-Game Rows", f"{len(team_matches):,}")
    c3.metric("Team-Season Rows", f"{len(team_ha):,}")

    teams = sorted(team_ha["Team"].unique().tolist())
    default_team_index = teams.index("Man City") if "Man City" in teams else 0

    st.header("Scene 1 - Home edge vs away resilience")
    selected_team = st.selectbox(
        "Follow one team through the story:",
        teams,
        index=default_team_index,
    )
    st.altair_chart(
        make_home_away_chart(team_ha, selected_team=selected_team),
        use_container_width=True,
    )

    selected_points = team_ha[team_ha["Team"] == selected_team].copy()
    selected_points["Gap"] = selected_points["AwayPoints"] - selected_points["HomePoints"]
    avg_gap = selected_points["Gap"].mean()
    if avg_gap > 0:
        venue_sentence = "wins points slightly better away from home"
    elif avg_gap < 0:
        venue_sentence = "relies more on home performances for points"
    else:
        venue_sentence = "is almost perfectly balanced between home and away points"
    st.caption(
        f"{selected_team} {venue_sentence} across the two seasons "
        f"(average away-home gap: {avg_gap:.2f} points)."
    )

    st.header("Scene 2 - Are attacking trends stable or streaky?")
    metric = st.selectbox(
        "Metric to track over the season:",
        list(METRIC_OPTIONS.keys()),
        format_func=lambda key: METRIC_OPTIONS[key],
    )
    st.altair_chart(
        make_trend_chart(team_matches, selected_team=selected_team, metric=metric),
        use_container_width=True,
    )
    st.caption(
        "A 5-game rolling average filters single-match noise while preserving "
        "short-run tactical swings."
    )

    st.header("Scene 3 - Which scorelines drive the extremes?")
    st.markdown(
        "Brush the scatter to isolate a region of scorelines, then inspect the ranked matches on the right."
    )
    st.altair_chart(make_extreme_match_chart(matches), use_container_width=True)

    st.header("Synthesis")
    st.markdown(
        """
Attacking output and points are related, but not linearly. Venue effects, finishing quality,
and game state management create separation between teams with similar shot volume.
This is why we need both aggregate views (points splits), trend views (rolling form),
and match-level outlier analysis (extreme scorelines) to understand consistency.
"""
    )

    st.header("Reader-Driven Exploration")
    st.markdown(
        """
Use the linked dashboard below to run your own checks:
- Click a team in Q3 to filter Q2.
- Change the metric in Q2.
- Brush scorelines in Q4 to inspect the highest-scoring matches in that region.
"""
    )
    st.altair_chart(
        make_linked_dashboard(team_matches=team_matches, team_ha=team_ha, matches=matches),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
