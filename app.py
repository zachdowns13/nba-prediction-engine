
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
# FIRST BASKET SIMULATOR — NBA FEED, STARTERS ONLY
# Requires existing: st, np, pd, requests, BASE, get_json,
# HEADERS, home_team, away_team, gid.
# =====================================================================

NBA_SCOREBOARD_URL = (
    "https://cdn.nba.com/static/json/liveData/scoreboard/"
    "todaysScoreboard_00.json"
)

NBA_BOXSCORE_URL = (
    "https://cdn.nba.com/static/json/liveData/boxscore/"
    "boxscore_{game_id}.json"
)

FIRST_BASKET_SIMS = 20_000

# ESPN/NBA team-name normalization.
TEAM_ALIASES = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "new jersey nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


def team_tricode(name):
    return TEAM_ALIASES.get(str(name).strip().lower())


@st.cache_data(ttl=60, show_spinner=False)
def load_nba_scoreboard():
    """Get the NBA's current-day scoreboard and game IDs."""
    response = requests.get(
        NBA_SCOREBOARD_URL,
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("scoreboard", {}).get("games", [])


@st.cache_data(ttl=60, show_spinner=False)
def load_nba_boxscore(nba_game_id):
    """Fetch an NBA live box score by NBA game ID."""
    response = requests.get(
        NBA_BOXSCORE_URL.format(game_id=str(nba_game_id)),
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("game", {})


def find_nba_game(home, away):
    """Match the selected matchup against the NBA current-day scoreboard."""
    home_code = team_tricode(home)
    away_code = team_tricode(away)

    if not home_code or not away_code:
        raise ValueError(
            "Could not map the selected teams to NBA team abbreviations."
        )

    games = load_nba_scoreboard()

    for game in games:
        nba_home = (game.get("homeTeam") or {}).get("teamTricode")
        nba_away = (game.get("awayTeam") or {}).get("teamTricode")

        if nba_home == home_code and nba_away == away_code:
            return game

    raise ValueError(
        "The selected game was not found on the NBA current-day scoreboard. "
        "This lookup supports today's games only; games on other dates "
        "need a date-specific NBA schedule lookup."
    )


def starters_from_nba_team(team_data):
    """Read only players whose NBA starter field is explicitly set."""
    starters = []
    seen = set()

    for player in team_data.get("players", []):
        flag = str(player.get("starter", "")).strip().lower()

        if flag not in ("1", "true", "yes"):
            continue

        first = player.get("firstName", "")
        last = player.get("familyName", "")
        name = player.get("name") or " ".join(
            part for part in (first, last) if part
        ).strip()

        if not name or name in seen:
            continue

        seen.add(name)
        starters.append({
            "name": name,
            "position": player.get("position", ""),
        })

    return starters


def get_nba_starting_fives(home, away):
    """Return exactly five NBA-identified starters per team."""
    game = find_nba_game(home, away)

    game_id = str(game.get("gameId", ""))

    if not game_id.isdigit() or len(game_id) != 10:
        raise ValueError("The NBA scoreboard returned an invalid game ID.")

    status = game.get("gameStatus", 0)

    # Status 1 = scheduled, 2 = live, 3 = final.
    box = load_nba_boxscore(game_id)

    nba_home = box.get("homeTeam") or {}
    nba_away = box.get("awayTeam") or {}

    home_players = starters_from_nba_team(nba_home)
    away_players = starters_from_nba_team(nba_away)

    if len(home_players) != 5 or len(away_players) != 5:
        raise ValueError(
            "The NBA box score does not yet identify five starters for "
            f"both teams. Found {len(home_players)} for {home} and "
            f"{len(away_players)} for {away}. "
            "The game may be upcoming and the official lineups may not "
            "have been published. No bench players were substituted."
        )

    return game_id, status, home_players, away_players


def simulate_first_basket(
    home_players,
    away_players,
    home,
    away,
    sims=FIRST_BASKET_SIMS,
):
    """Simulate the first scoring event for exactly 10 starters."""
    if len(home_players) != 5 or len(away_players) != 5:
        raise ValueError("Exactly five starters per team are required.")

    rng = np.random.default_rng()

    home_names = [p["name"] for p in home_players]
    away_names = [p["name"] for p in away_players]

    def weights(players):
        # Basic fallback only; replace with player-level stats for a
        # properly calibrated forecast.
        values = []
        for player in players:
            pos = str(player.get("position", "")).upper()
            values.append({
                "C": 1.15,
                "PF": 1.10,
                "SF": 1.05,
                "SG": 1.05,
                "PG": 1.00,
            }.get(pos, 1.00))

        values = np.asarray(values, dtype=float)
        return values / values.sum()

    home_weights = weights(home_players)
    away_weights = weights(away_players)

    # Simplified initial possession and scoring assumptions.
    possession_home = rng.random(sims) < 0.52

    first_team = np.full(sims, -1, dtype=int)
    first_player = np.full(sims, "", dtype=object)
    first_event = np.full(sims, "", dtype=object)
    first_points = np.zeros(sims, dtype=int)

    def record(indices, team_id, names, player_weights):
        if len(indices) == 0:
            return

        first_team[indices] = team_id

        chosen = rng.choice(
            len(names),
            size=len(indices),
            p=player_weights,
        )
        first_player[indices] = np.asarray(names, dtype=object)[chosen]

        event = rng.choice(
            ["2PT", "3PT", "FT"],
            size=len(indices),
            p=[0.60, 0.32, 0.08],
        )

        first_event[indices] = event
        first_points[indices] = np.select(
            [event == "2PT", event == "3PT", event == "FT"],
            [2, 3, 1],
        )

    for _ in range(40):
        active = first_team == -1

        if not active.any():
            break

        home_ids = np.flatnonzero(active & possession_home)
        away_ids = np.flatnonzero(active & ~possession_home)

        home_scored = rng.random(len(home_ids)) < 0.28
        away_scored = rng.random(len(away_ids)) < 0.27

        record(
            home_ids[home_scored], 0, home_names, home_weights
        )
        record(
            away_ids[away_scored], 1, away_names, away_weights
        )

        possession_home[active] = ~possession_home[active]

    completed = first_team != -1
    count = int(completed.sum())

    if count == 0:
        return None

    player_rows = []

    for team_id, team_name, names in [
        (0, home, home_names),
        (1, away, away_names),
    ]:
        for name in names:
            hits = int(np.sum(
                completed
                & (first_team == team_id)
                & (first_player == name)
            ))

            player_rows.append({
                "Player": name,
                "Team": team_name,
                "Simulated first scores": hits,
                "Probability": hits / count,
            })

    players_df = pd.DataFrame(player_rows).sort_values(
        "Probability",
        ascending=False,
    ).reset_index(drop=True)

    teams_df = pd.DataFrame([
        {
            "Team": home,
            "Probability": float(np.mean(first_team[completed] == 0)),
        },
        {
            "Team": away,
            "Probability": float(np.mean(first_team[completed] == 1)),
        },
    ])

    events_df = pd.DataFrame([
        {
            "First scoring event": event,
            "Probability": float(np.mean(first_event[completed] == event)),
        }
        for event in ("2PT", "3PT", "FT")
    ])

    return {
        "players": players_df,
        "teams": teams_df,
        "events": events_df,
        "completed": count,
        "no_score": sims - count,
        "points": first_points[completed],
    }


# =====================================================================
# FIRST BASKET UI
# =====================================================================

st.markdown("---")
st.header("🎯 First Basket Simulator")
st.caption(
    "Uses the NBA box score to identify the starting five for each team. "
    "No sliders and no bench-player substitutions."
)

if st.button(
    "Simulate First Basket",
    type="primary",
    key=f"nba_first_basket_{gid}",
):
    if home_team == away_team:
        st.error("A team cannot play itself.")
    else:
        try:
            with st.spinner("Checking NBA game data and starting lineups..."):
                nba_game_id, game_status, home_players, away_players = (
                    get_nba_starting_fives(home_team, away_team)
                )

                result = simulate_first_basket(
                    home_players,
                    away_players,
                    home_team,
                    away_team,
                )

            if result is None:
                st.warning("No scoring events were recorded in the simulation.")
            else:
                st.success(
                    f"Loaded exactly 10 starters. NBA game ID: {nba_game_id}"
                )

                if game_status == 1:
                    st.info(
                        "The game is scheduled. Confirm that the published "
                        "lineup reflects the expected starters."
                    )

                with st.expander("View the 10 players used"):
                    st.markdown(f"**{home_team}**")
                    st.write(", ".join(p["name"] for p in home_players))
                    st.markdown(f"**{away_team}**")
                    st.write(", ".join(p["name"] for p in away_players))

                top = result["players"].iloc[0]
                team_probs = result["teams"].set_index("Team")["Probability"]

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Top simulated scorer",
                    top["Player"],
                    f"{top['Probability']:.2%}",
                )
                c2.metric(
                    f"{home_team} scores first",
                    f"{team_probs[home_team]:.1%}",
                )
                c3.metric(
                    f"{away_team} scores first",
                    f"{team_probs[away_team]:.1%}",
                )

                st.subheader("First basket probabilities")

                player_display = result["players"].copy()
                player_display["Probability"] = player_display[
                    "Probability"
                ].map(lambda p: f"{p:.2%}")

                st.dataframe(
                    player_display,
                    hide_index=True,
                    use_container_width=True,
                )

                st.subheader("First scoring event")

                event_display = result["events"].copy()
                event_display["Probability"] = event_display[
                    "Probability"
                ].map(lambda p: f"{p:.2%}")

                st.dataframe(
                    event_display,
                    hide_index=True,
                    use_container_width=True,
                )

                st.metric(
                    "Average first-score points",
                    f"{np.mean(result['points']):.2f}",
                )

                st.caption(
                    f"Simulations without a score in 40 possessions: "
                    f"{result['no_score']:,}."
                )

                st.warning(
                    "These probabilities use simplified scoring assumptions "
                    "and position-based weights. They are not calibrated "
                    "using actual player-level first-basket rates."
                )

        except Exception as exc:
            st.error(f"First basket simulator: {exc}")
            st.info(
                "For upcoming games, official starters may not be available "
                "yet. This version checks today's NBA scoreboard only. "
                "A date-specific schedule and projected-lineup provider "
                "are needed for future-date pregame predictions."
            )
