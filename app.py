
import re
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NBA Matchup Simulator", layout="wide")
st.title("🏀 NBA Matchup Simulator")
st.caption(
    "ESPN schedule, odds and injuries · Monte Carlo game predictions "
    "· First basket simulator"
)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =====================================================================
# DATA LAYER
# =====================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_json(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def to_american(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100
    try:
        return int(float(s.replace("+", "")))
    except (ValueError, TypeError):
        return None


def parse_odds(comp, home_abbr):
    o = (comp.get("odds") or [{}])[0]
    out = {
        "provider": (o.get("provider") or {}).get("name", "n/a"),
        "total": o.get("overUnder"),
        "home_spread": None,
        "home_ml": None,
        "away_ml": None,
        "home_spread_odds": -110,
        "away_spread_odds": -110,
        "over_odds": -110,
        "under_odds": -110,
    }

    details = (o.get("details") or "").strip()
    m = re.match(r"^([A-Z]{2,4})\s+([+-]?\d+(\.\d+)?)$", details)

    if details.upper() in ("EVEN", "PK"):
        out["home_spread"] = 0.0
    elif m:
        val = float(m.group(2))
        out["home_spread"] = val if m.group(1) == home_abbr else -val
    elif o.get("spread") is not None:
        out["home_spread"] = float(o["spread"])

    h = o.get("homeTeamOdds") or {}
    a = o.get("awayTeamOdds") or {}

    out["home_ml"] = to_american(h.get("moneyLine"))
    out["away_ml"] = to_american(a.get("moneyLine"))

    ml = o.get("moneyline") or {}
    if out["home_ml"] is None:
        out["home_ml"] = to_american(
            ((ml.get("home") or {}).get("close") or {}).get("odds")
        )
    if out["away_ml"] is None:
        out["away_ml"] = to_american(
            ((ml.get("away") or {}).get("close") or {}).get("odds")
        )

    out["home_spread_odds"] = to_american(h.get("spreadOdds")) or -110
    out["away_spread_odds"] = to_american(a.get("spreadOdds")) or -110
    out["over_odds"] = to_american(o.get("overOdds")) or -110
    out["under_odds"] = to_american(o.get("underOdds")) or -110

    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_games(date_str):
    data = get_json(f"{BASE}/scoreboard", {"dates": date_str})
    games = []

    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        home = next(
            c for c in comp["competitors"] if c["homeAway"] == "home"
        )
        away = next(
            c for c in comp["competitors"] if c["homeAway"] == "away"
        )

        games.append({
            "id": ev["id"],
            "home": home["team"]["displayName"],
            "away": away["team"]["displayName"],
            "start": ev.get("date"),
            "state": ev["status"]["type"]["state"],
            "odds": parse_odds(
                comp, home["team"].get("abbreviation", "")
            ),
        })

    return games


@st.cache_data(ttl=600, show_spinner=False)
def load_injuries():
    data = get_json(f"{BASE}/injuries")
    out = {}

    for t in data.get("injuries", []):
        rows = []
        for i in t.get("injuries", []):
            ath = i.get("athlete") or {}
            rows.append({
                "Player": ath.get("displayName", "?"),
                "Pos": (ath.get("position") or {}).get("abbreviation", ""),
                "Status": i.get("status", ""),
                "Note": (i.get("shortComment") or "")[:140],
            })
        out[t.get("displayName")] = rows

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_form():
    try:
        data = get_json(STANDINGS_URL)
    except Exception:
        return {}

    out = {}

    def walk(node):
        for e in (node.get("standings") or {}).get("entries", []):
            stats = {
                x.get("name"): x.get("value")
                for x in e.get("stats", [])
            }
            gp = stats.get("gamesPlayed") or (
                (stats.get("wins") or 0) + (stats.get("losses") or 0)
            )
            pf = stats.get("avgPointsFor")
            pa = stats.get("avgPointsAgainst")

            if gp and pf is not None and pa is not None:
                out[e["team"]["displayName"]] = {
                    "gp": float(gp),
                    "pf": float(pf),
                    "pa": float(pa),
                }

        for child in node.get("children", []):
            walk(child)

    walk(data)
    return out


# =====================================================================
# TEAM STRENGTH PRIORS
# =====================================================================

LEAGUE_AVG = 114.0

PRIORS = {
    "Atlanta Hawks": (1.5, 1.5),
    "Boston Celtics": (4.5, -3.5),
    "Brooklyn Nets": (-4.0, 2.5),
    "Charlotte Hornets": (-1.0, 0.5),
    "Chicago Bulls": (-1.5, 2.0),
    "Cleveland Cavaliers": (2.5, -1.5),
    "Dallas Mavericks": (1.0, 1.0),
    "Denver Nuggets": (4.0, 0.5),
    "Detroit Pistons": (3.5, -2.0),
    "Golden State Warriors": (0.5, 0.5),
    "Houston Rockets": (1.0, -1.5),
    "Indiana Pacers": (1.5, 3.5),
    "LA Clippers": (-0.5, -0.5),
    "Los Angeles Lakers": (2.0, 1.0),
    "Memphis Grizzlies": (-1.5, 1.5),
    "Miami Heat": (0.5, -1.0),
    "Milwaukee Bucks": (0.5, 2.5),
    "Minnesota Timberwolves": (2.5, -0.5),
    "New Orleans Pelicans": (-0.5, 2.0),
    "New York Knicks": (3.0, -1.5),
    "Oklahoma City Thunder": (5.0, -4.5),
    "Orlando Magic": (0.0, -2.0),
    "Philadelphia 76ers": (1.5, -0.5),
    "Phoenix Suns": (1.0, -0.5),
    "Portland Trail Blazers": (-1.0, 0.5),
    "Sacramento Kings": (0.5, 2.5),
    "San Antonio Spurs": (3.0, -2.5),
    "Toronto Raptors": (0.0, 1.0),
    "Utah Jazz": (-1.0, 3.5),
    "Washington Wizards": (-3.5, 4.5),
}


def team_strength(team, form):
    off_p, def_p = PRIORS.get(team, (0.0, 0.0))
    prior_off = LEAGUE_AVG + off_p
    prior_allowed = LEAGUE_AVG + def_p
    f = form.get(team)

    if not f:
        return prior_off, prior_allowed, 0

    w = f["gp"] / (f["gp"] + 25.0)
    return (
        w * f["pf"] + (1 - w) * prior_off,
        w * f["pa"] + (1 - w) * prior_allowed,
        f["gp"],
    )


# =====================================================================
# BETTING MATH
# =====================================================================

def dec(odds):
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def implied(odds):
    return (
        100 / (odds + 100)
        if odds > 0
        else abs(odds) / (abs(odds) + 100)
    )


def novig(a, b):
    ia, ib = implied(a), implied(b)
    return ia / (ia + ib), ib / (ia + ib)


def ev100(p_win, p_push, odds):
    return 100 * (
        p_win * (dec(odds) - 1) - (1 - p_win - p_push)
    )


def fair_american(p):
    p = min(max(p, 0.001), 0.999)
    if p >= 0.5:
        return f"{-round(100 * p / (1 - p)):+d}"
    return f"+{round(100 * (1 - p) / p)}"


# =====================================================================
# MODEL SETTINGS
# =====================================================================

SIMS = 10_000
HCA = 2.0
MARGIN_SD = 12.0
TOTAL_SD = 18.0
LEAN_MIN, LEAN_STRONG = 0.02, 0.04

ROLE_PTS = {
    "Star": 6.0,
    "Starter": 3.0,
    "Rotation": 1.2,
    "Deep bench": 0.0,
}
STATUS_W = {
    "out": 1.0,
    "doubtful": 0.85,
    "questionable": 0.5,
    "day-to-day": 0.3,
    "probable": 0.1,
}


# =====================================================================
# GAME SELECTION
# =====================================================================

date = st.date_input("Game date", dt.date.today())

try:
    games = load_games(date.strftime("%Y%m%d"))
    injuries = load_injuries()
    form = load_team_form()
    live_ok = True
except Exception as e:
    games, injuries, form, live_ok = [], {}, {}, False
    st.error(f"Could not reach ESPN ({e}). Use manual matchup.")

all_teams = sorted(PRIORS.keys())
labels = [f"{g['away']} @ {g['home']}" for g in games]
labels.append("✏️ Manual matchup")

choice = st.selectbox("Select game", labels)

blank = {
    "provider": "manual",
    "total": None,
    "home_spread": None,
    "home_ml": None,
    "away_ml": None,
    "home_spread_odds": -110,
    "away_spread_odds": -110,
    "over_odds": -110,
    "under_odds": -110,
}

if choice == "✏️ Manual matchup":
    c1, c2 = st.columns(2)
    home_team = c1.selectbox(
        "Home team",
        all_teams,
        index=all_teams.index("Philadelphia 76ers"),
    )
    away_team = c2.selectbox(
        "Away team",
        all_teams,
        index=all_teams.index("Boston Celtics"),
    )
    odds, gid = dict(blank), "manual"
else:
    g = games[labels.index(choice)]
    home_team = g["home"]
    away_team = g["away"]
    odds = dict(g["odds"])
    gid = g["id"]

    if g["state"] != "pre":
        st.warning(
            "Game has started or finished. Pregame lines may be unavailable."
        )

if not games and live_ok:
    st.info(
        "No games found for this date. Pick another date or use manual matchup."
    )

# Allow manual entry of missing sportsbook lines.
if any(
    odds[k] is None
    for k in ("home_ml", "away_ml", "home_spread", "total")
):
    with st.expander(
        "Enter missing sportsbook lines",
        expanded=True,
    ):
        a, b, c = st.columns(3)

        odds["home_ml"] = a.number_input(
            f"{home_team} moneyline",
            value=int(odds["home_ml"] or -110),
            step=5,
            key=f"hml_{gid}",
        )
        odds["away_ml"] = a.number_input(
            f"{away_team} moneyline",
            value=int(odds["away_ml"] or -110),
            step=5,
            key=f"aml_{gid}",
        )
        odds["home_spread"] = b.number_input(
            f"{home_team} spread",
            value=float(
                odds["home_spread"]
                if odds["home_spread"] is not None
                else 0.0
            ),
            step=0.5,
            key=f"sp_{gid}",
        )
        odds["total"] = c.number_input(
            "Game total",
            value=float(
                odds["total"] if odds["total"] is not None else 224.5
            ),
            step=0.5,
            key=f"tot_{gid}",
        )


# =====================================================================
# INJURIES
# =====================================================================

def injury_panel(team, key):
    rows = injuries.get(team, [])

    if not rows:
        st.caption(
            "No injuries reported."
            if live_ok
            else "Injury data unavailable."
        )
        return 0.0

    df = pd.DataFrame(rows)
    df["Role"] = "Rotation"

    edited = st.data_editor(
        df,
        key=key,
        hide_index=True,
        disabled=["Player", "Pos", "Status", "Note"],
        column_config={
            "Role": st.column_config.SelectboxColumn(
                options=list(ROLE_PTS),
                required=True,
            )
        },
    )

    total = min(
        sum(
            ROLE_PTS[r] * STATUS_W.get(str(s).lower(), 0.3)
            for r, s in zip(edited["Role"], edited["Status"])
        ),
        15.0,
    )

    st.caption(f"Estimated impact: **-{total:.1f} points**")
    return total


st.subheader("🩹 Injuries")
st.caption(
    "ESPN injury reports. Adjust player roles to estimate their impact."
)

i1, i2 = st.columns(2)

with i1:
    st.markdown(f"**🏠 {home_team}**")
    home_inj = injury_panel(home_team, f"hinj_{gid}")
    home_b2b = st.checkbox("Back-to-back", key=f"hb_{gid}")

with i2:
    st.markdown(f"**✈️ {away_team}**")
    away_inj = injury_panel(away_team, f"ainj_{gid}")
    away_b2b = st.checkbox("Back-to-back", key=f"ab_{gid}")


# =====================================================================
# GAME SIMULATION
# =====================================================================

st.markdown("---")
st.header("📊 Game Prediction")

if st.button("🔥 Simulate Game", type="primary", key=f"game_{gid}"):
    if home_team == away_team:
        st.error("A team can't play itself.")
        st.stop()

    h_off, h_allowed, _ = team_strength(home_team, form)
    a_off, a_allowed, _ = team_strength(away_team, form)

    home_pts = (h_off + a_allowed) / 2 + HCA / 2
    away_pts = (a_off + h_allowed) / 2 - HCA / 2

    home_pts -= home_inj / 2
    away_pts += home_inj / 2
    away_pts -= away_inj / 2
    home_pts += away_inj / 2

    if home_b2b:
        home_pts -= 0.75
        away_pts += 0.75

    if away_b2b:
        away_pts -= 0.75
        home_pts += 0.75

    rng = np.random.default_rng()

    margin = np.rint(
        rng.normal(home_pts - away_pts, MARGIN_SD, SIMS)
    )
    total = np.rint(
        rng.normal(home_pts + away_pts, TOTAL_SD, SIMS)
    )

    ties = margin == 0
    margin[ties] = rng.choice([-1, 1], ties.sum())

    line = odds["home_spread"]
    tot_line = odds["total"]

    p_home = float(np.mean(margin > 0))
    ch = float(np.mean(margin + line > 0))
    ca = float(np.mean(margin + line < 0))
    ov = float(np.mean(total > tot_line))
    un = float(np.mean(total < tot_line))

    nv_ml = novig(odds["home_ml"], odds["away_ml"])
    nv_sp = novig(odds["home_spread_odds"], odds["away_spread_odds"])
    nv_t = novig(odds["over_odds"], odds["under_odds"])

    def lean(edge):
        if edge >= LEAN_STRONG:
            return "🟢 Strong lean"
        if edge >= LEAN_MIN:
            return "🟡 Lean"
        return "—"

    def row(market, side, line_txt, book_odds, model_p, push, book_p):
        edge = model_p - book_p
        return {
            "Market": market,
            "Side": side,
            "Line": line_txt,
            "Model %": f"{model_p * 100:.1f}%",
            "Model odds": fair_american(model_p),
            "Sportsbook odds": f"{book_odds:+d}",
            "Book % (no-vig)": f"{book_p * 100:.1f}%",
            "Edge": f"{edge * 100:+.1f}%",
            "EV/$100": f"{ev100(model_p, push, book_odds):.1f}",
            "Model leans": lean(edge),
            "_edge": edge,
        }

    push_s = max(0.0, 1 - ch - ca)
    push_t = max(0.0, 1 - ov - un)

    rows = [
        row("Moneyline", home_team, "", odds["home_ml"], p_home, 0, nv_ml[0]),
        row("Moneyline", away_team, "", odds["away_ml"], 1 - p_home, 0, nv_ml[1]),
        row("Spread", home_team, f"{line:+.1f}", odds["home_spread_odds"], ch, push_s, nv_sp[0]),
        row("Spread", away_team, f"{-line:+.1f}", odds["away_spread_odds"], ca, push_s, nv_sp[1]),
        row("Total", "Over", str(tot_line), odds["over_odds"], ov, push_t, nv_t[0]),
        row("Total", "Under", str(tot_line), odds["under_odds"], un, push_t, nv_t[1]),
    ]

    st.subheader(f"{away_team} @ {home_team}")
    st.caption(f"Sportsbook: {odds['provider']} · {SIMS:,} simulations")
    st.markdown(
        f"#### Projected score: {away_team} **{away_pts:.0f}** – "
        f"{home_team} **{home_pts:.0f}**"
    )

    best = max(rows, key=lambda r: r["_edge"])

    if best["_edge"] >= LEAN_MIN:
        label = f"{best['Side']} {best['Line']}".strip()
        st.success(
            f"Best model edge: {best['Market']} — {label}. "
            f"Model probability: {best['Model %']}; "
            f"no-vig market probability: {best['Book % (no-vig)']}."
        )
    else:
        st.info("No clear positive edge based on these model assumptions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Home win probability", f"{p_home:.1%}")
        st.metric("Away win probability", f"{1-p_home:.1%}")

    with col2:
        st.metric("Home cover probability", f"{ch:.1%}")
        st.metric("Away cover probability", f"{ca:.1%}")

    with col3:
        st.metric("Over probability", f"{ov:.1%}")
        st.metric("Under probability", f"{un:.1%}")

    with st.expander("See full betting model results"):
        st.dataframe(
            pd.DataFrame(rows).drop(columns="_edge"),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Probabilities and estimated betting value are model outputs, "
        "not guarantees. The model uses simplified team-strength and "
        "score-distribution assumptions."
    )


# =====================================================================
# FIRST BASKET SIMULATOR
# =====================================================================

st.markdown("---")
st.header("🎯 First Basket Simulator")
st.caption(
    "Simulate the first scoring event, including two-pointers, "
    "three-pointers, and made free throws."
)

FIRST_BASKET_SIMS = 20_000


def parse_first_basket_players(text):
    """Read one player per line, optionally followed by | weight."""
    players = []
    weights = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.rsplit("|", 1)

        if len(parts) == 2:
            name = parts[0].strip()
            try:
                weight = float(parts[1].strip())
            except ValueError:
                continue
        else:
            name = line
            weight = 1.0

        if name and np.isfinite(weight) and weight > 0:
            players.append(name)
            weights.append(weight)

    return players, weights


def simulate_first_basket(
    home_players,
    home_weights,
    away_players,
    away_weights,
    home_tip_prob,
    home_score_prob,
    away_score_prob,
    home_three_prob,
    away_three_prob,
    home_ft_share,
    away_ft_share,
    sims=FIRST_BASKET_SIMS,
):
    """
    Simplified opening-possession simulation.

    A scoring event is assigned to a player using that team's player
    weights. The first scoring event can be a 2PT, 3PT, or FT.

    This is not a play-by-play engine and does not use live player stats.
    """
    rng = np.random.default_rng()

    home_weights = np.asarray(home_weights, dtype=float)
    away_weights = np.asarray(away_weights, dtype=float)

    home_weights /= home_weights.sum()
    away_weights /= away_weights.sum()

    possession_home = rng.random(sims) < home_tip_prob

    first_team = np.full(sims, -1, dtype=int)
    first_player = np.full(sims, "", dtype=object)
    first_type = np.full(sims, "", dtype=object)
    first_points = np.zeros(sims, dtype=int)

    home_names = np.asarray(home_players, dtype=object)
    away_names = np.asarray(away_players, dtype=object)

    for _ in range(40):
        active = first_team == -1

        if not active.any():
            break

        home_ids = np.flatnonzero(active & possession_home)
        away_ids = np.flatnonzero(active & ~possession_home)

        home_scored = (
            rng.random(len(home_ids)) < home_score_prob
        )
        away_scored = (
            rng.random(len(away_ids)) < away_score_prob
        )

        scoring_home = home_ids[home_scored]
        scoring_away = away_ids[away_scored]

        def record_scores(
            ids, team_id, names, weights, three_prob, ft_share
        ):
            if len(ids) == 0:
                return

            first_team[ids] = team_id

            player_ids = rng.choice(
                len(names), size=len(ids), p=weights
            )
            first_player[ids] = names[player_ids]

            # Remaining scoring events are two-pointers.
            ft_share = min(max(ft_share, 0.0), 1.0)
            three_prob = min(max(three_prob, 0.0), 1.0)

            probs = [
                (1 - ft_share) * (1 - three_prob),
                (1 - ft_share) * three_prob,
                ft_share,
            ]

            shot = rng.choice(
                ["2PT", "3PT", "FT"],
                size=len(ids),
                p=probs,
            )

            first_type[ids] = shot
            first_points[ids] = np.select(
                [
                    shot == "2PT",
                    shot == "3PT",
                    shot == "FT",
                ],
                [2, 3, 1],
            )

        record_scores(
            scoring_home,
            0,
            home_names,
            home_weights,
            home_three_prob,
            home_ft_share,
        )
        record_scores(
            scoring_away,
            1,
            away_names,
            away_weights,
            away_three_prob,
            away_ft_share,
        )

        # Simplified alternating possessions after each possession.
        possession_home[active] = ~possession_home[active]

    completed = first_team != -1
    completed_count = int(completed.sum())

    if completed_count == 0:
        return None

    player_rows = []

    for team_id, team_name, players in [
        (0, home_team, home_players),
        (1, away_team, away_players),
    ]:
        for player in players:
            count = int(np.sum(
                completed
                & (first_team == team_id)
                & (first_player == player)
            ))

            player_rows.append({
                "Player": player,
                "Team": team_name,
                "First basket count": count,
                "Probability": count / completed_count,
            })

    player_df = pd.DataFrame(player_rows).sort_values(
        "Probability", ascending=False
    ).reset_index(drop=True)

    team_df = pd.DataFrame([
        {
            "Team": home_team,
            "Probability": float(np.mean(first_team[completed] == 0)),
        },
        {
            "Team": away_team,
            "Probability": float(np.mean(first_team[completed] == 1)),
        },
    ])

    type_rows = []

    for shot_type in ["2PT", "3PT", "FT"]:
        count = int(np.sum(
            completed & (first_type == shot_type)
        ))

        type_rows.append({
            "First scoring event": shot_type,
            "Probability": count / completed_count,
        })

    return {
        "players": player_df,
        "teams": team_df,
        "types": pd.DataFrame(type_rows),
        "completed": completed_count,
        "no_score": sims - completed_count,
        "first_points": first_points[completed],
    }


with st.expander("⚙️ First basket settings", expanded=True):
    st.markdown(
        "Enter expected starters or other eligible players, one per line. "
        "Optional format: `Player Name | 3`. A weight of 3 gives that "
        "player three times the selection weight of a player weighted 1 "
        "on the same team."
    )

    hcol, acol = st.columns(2)

    with hcol:
        st.markdown(f"**🏠 {home_team}**")
        home_lineup_text = st.text_area(
            "Home players",
            key=f"fb_home_{gid}",
            placeholder=(
                "Player A | 3\nPlayer B | 2\nPlayer C | 1.5\n"
                "Player D | 1\nPlayer E | 1"
            ),
            height=160,
        )

    with acol:
        st.markdown(f"**✈️ {away_team}**")
        away_lineup_text = st.text_area(
            "Away players",
            key=f"fb_away_{gid}",
            placeholder=(
                "Player F | 3\nPlayer G | 2\nPlayer H | 1.5\n"
                "Player I | 1\nPlayer J | 1"
            ),
            height=160,
        )

    st.markdown("**Opening possession assumptions**")

    c1, c2, c3 = st.columns(3)

    home_tip_prob = c1.slider(
        f"{home_team} tip-off win chance (%)",
        30, 70, 52,
        key=f"fb_tip_{gid}",
    ) / 100

    home_score_prob = c2.slider(
        f"{home_team} possession scoring chance (%)",
        10, 60, 28,
        key=f"fb_hscore_{gid}",
    ) / 100

    away_score_prob = c3.slider(
        f"{away_team} possession scoring chance (%)",
        10, 60, 27,
        key=f"fb_asc_{gid}",
    ) / 100

    st.markdown("**Scoring-event assumptions**")

    c4, c5, c6 = st.columns(3)

    home_three_prob = c4.slider(
        f"{home_team} three-point share (%)",
        0, 70, 35,
        key=f"fb_hthree_{gid}",
    ) / 100

    away_three_prob = c5.slider(
        f"{away_team} three-point share (%)",
        0, 70, 35,
        key=f"fb_athree_{gid}",
    ) / 100

    home_ft_share = c6.slider(
        "Free throws as share of scoring events (%)",
        0, 30, 8,
        key=f"fb_ft_{gid}",
    ) / 100

    away_ft_share = home_ft_share


if st.button(
    "🎯 Simulate First Basket",
    type="primary",
    key=f"fb_run_{gid}",
):
    home_players, home_weights = parse_first_basket_players(
        home_lineup_text
    )
    away_players, away_weights = parse_first_basket_players(
        away_lineup_text
    )

    if not home_players or not away_players:
        st.error(
            "Enter at least one valid player for each team. "
            "Use a positive weight, for example: Player Name | 2"
        )
    elif home_team == away_team:
        st.error("Select two different teams.")
    else:
        result = simulate_first_basket(
            home_players=home_players,
            home_weights=home_weights,
            away_players=away_players,
            away_weights=away_weights,
            home_tip_prob=home_tip_prob,
            home_score_prob=home_score_prob,
            away_score_prob=away_score_prob,
            home_three_prob=home_three_prob,
            away_three_prob=away_three_prob,
            home_ft_share=home_ft_share,
            away_ft_share=away_ft_share,
        )

        if result is None:
            st.warning(
                "No scoring event was recorded. Try higher possession "
                "scoring probabilities or rerun the simulation."
            )
        else:
            st.subheader(
                f"First basket forecast: {away_team} @ {home_team}"
            )
            st.caption(
                f"{FIRST_BASKET_SIMS:,} simulations · "
                f"{result['completed']:,} with a recorded first score"
            )

            top = result["players"].iloc[0]
            team_probs = result["teams"].set_index("Team")["Probability"]

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Top simulated first scorer",
                top["Player"],
                f"{top['Probability']:.2%} probability",
            )
            m2.metric(
                f"{home_team} scores first",
                f"{team_probs[home_team]:.1%}",
            )
            m3.metric(
                f"{away_team} scores first",
                f"{team_probs[away_team]:.1%}",
            )

            st.markdown("#### 🏆 Player rankings")

            players_display = result["players"].copy()
            players_display["Probability"] = (
                players_display["Probability"].map(
                    lambda p: f"{p:.2%}"
                )
            )

            st.dataframe(
                players_display,
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("#### 🏀 First scoring event breakdown")

            types_display = result["types"].copy()
            types_display["Probability"] = (
                types_display["Probability"].map(
                    lambda p: f"{p:.2%}"
                )
            )

            st.dataframe(
                types_display,
                hide_index=True,
                use_container_width=True,
            )

            st.metric(
                "Average points on the first scoring event",
                f"{np.mean(result['first_points']):.2f}",
            )

            st.caption(
                f"No score recorded within 40 simulated possessions: "
                f"{result['no_score']:,}. Those runs are excluded from "
                "the first-scorer percentages."
            )

            st.info(
                "This is an assumption-based Monte Carlo model. Player "
                "weights, tip-off probability, and scoring assumptions "
                "are editable inputs, not verified live player statistics. "
                "The simulation simplifies rebounds, turnovers, fouls, "
                "free-throw sequences, and actual play-by-play."
            )

st.markdown("---")
st.caption(
    "For informational and entertainment purposes. Model estimates "
    "are not guarantees of game results or betting outcomes."
)
