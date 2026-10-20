import re
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NBA Matchup Simulator", layout="wide")
st.title("🏀 NBA Matchup Simulator")
st.caption("Live schedule, odds and injuries from ESPN + Monte Carlo simulation")

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =====================================================================
# DATA LAYER (live, cached)
# =====================================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_json(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def to_american(x):
    """ESPN gives odds as ints, '-250', '+180' or 'EVEN'. Return int or None."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100
    try:
        return int(float(s.replace("+", "")))
    except ValueError:
        return None


def parse_odds(comp, home_abbr):
    o = (comp.get("odds") or [{}])[0]
    out = {"provider": (o.get("provider") or {}).get("name", "n/a"),
           "total": o.get("overUnder"), "home_spread": None,
           "home_ml": None, "away_ml": None,
           "home_spread_odds": -110, "away_spread_odds": -110,
           "over_odds": -110, "under_odds": -110}

    # Spread: derive the HOME spread from the 'details' text (e.g. "BOS -5.5"),
    # which is unambiguous about which team is favored.
    details = (o.get("details") or "").strip()
    m = re.match(r"^([A-Z]{2,4})\s+([+-]?\d+(\.\d+)?)$", details)
    if details.upper() in ("EVEN", "PK"):
        out["home_spread"] = 0.0
    elif m:
        val = float(m.group(2))
        out["home_spread"] = val if m.group(1) == home_abbr else -val
    elif o.get("spread") is not None:
        out["home_spread"] = float(o["spread"])

    h, a = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    out["home_ml"], out["away_ml"] = to_american(h.get("moneyLine")), to_american(a.get("moneyLine"))
    ml = o.get("moneyline") or {}
    if out["home_ml"] is None:
        out["home_ml"] = to_american(((ml.get("home") or {}).get("close") or {}).get("odds"))
    if out["away_ml"] is None:
        out["away_ml"] = to_american(((ml.get("away") or {}).get("close") or {}).get("odds"))
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
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        games.append({
            "id": ev["id"],
            "home": home["team"]["displayName"], "away": away["team"]["displayName"],
            "start": ev.get("date"),
            "state": ev["status"]["type"]["state"],  # pre / in / post
            "odds": parse_odds(comp, home["team"].get("abbreviation", "")),
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
    """Current-season points for/against per game from ESPN standings."""
    try:
        data = get_json(STANDINGS_URL)
    except Exception:
        return {}
    out = {}

    def walk(node):
        for e in (node.get("standings") or {}).get("entries", []):
            s = {x.get("name"): x.get("value") for x in e.get("stats", [])}
            gp = s.get("gamesPlayed") or ((s.get("wins") or 0) + (s.get("losses") or 0))
            if gp and s.get("avgPointsFor") and s.get("avgPointsAgainst"):
                out[e["team"]["displayName"]] = {"gp": gp, "pf": s["avgPointsFor"], "pa": s["avgPointsAgainst"]}
        for c in node.get("children", []):
            walk(c)

    walk(data)
    return out


# =====================================================================
# PRIORS (used early in the season, blended out as games are played)
# These are PLACEHOLDER preseason strengths. Update them yourself.
# off_rtg: + = scores more than average. def_rtg: + = allows more (worse).
# =====================================================================
LEAGUE_AVG = 114.0
PRIORS = {
    "Atlanta Hawks": (1.5, 1.5), "Boston Celtics": (4.5, -3.5), "Brooklyn Nets": (-4.0, 2.5),
    "Charlotte Hornets": (-1.0, 0.5), "Chicago Bulls": (-1.5, 2.0), "Cleveland Cavaliers": (2.5, -1.5),
    "Dallas Mavericks": (1.0, 1.0), "Denver Nuggets": (4.0, 0.5), "Detroit Pistons": (3.5, -2.0),
    "Golden State Warriors": (0.5, 0.5), "Houston Rockets": (1.0, -1.5), "Indiana Pacers": (1.5, 3.5),
    "LA Clippers": (-0.5, -0.5), "Los Angeles Lakers": (2.0, 1.0), "Memphis Grizzlies": (-1.5, 1.5),
    "Miami Heat": (0.5, -1.0), "Milwaukee Bucks": (0.5, 2.5), "Minnesota Timberwolves": (2.5, -0.5),
    "New Orleans Pelicans": (-0.5, 2.0), "New York Knicks": (3.0, -1.5), "Oklahoma City Thunder": (5.0, -4.5),
    "Orlando Magic": (0.0, -2.0), "Philadelphia 76ers": (1.5, -0.5), "Phoenix Suns": (1.0, -0.5),
    "Portland Trail Blazers": (-1.0, 0.5), "Sacramento Kings": (0.5, 2.5), "San Antonio Spurs": (3.0, -2.5),
    "Toronto Raptors": (0.0, 1.0), "Utah Jazz": (-1.0, 3.5), "Washington Wizards": (-3.5, 4.5),
}


def team_strength(team, form):
    off_p, def_p = PRIORS.get(team, (0.0, 0.0))
    prior_off, prior_allowed = LEAGUE_AVG + off_p, LEAGUE_AVG + def_p
    f = form.get(team)
    if not f:
        return prior_off, prior_allowed, 0
    w = f["gp"] / (f["gp"] + 25.0)  # more games played -> trust real results more
    return w * f["pf"] + (1 - w) * prior_off, w * f["pa"] + (1 - w) * prior_allowed, f["gp"]


# =====================================================================
# BETTING MATH
# =====================================================================
def dec(odds):
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def implied(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def novig(a, b):
    ia, ib = implied(a), implied(b)
    return ia / (ia + ib), ib / (ia + ib)


def ev100(p_win, p_push, odds):
    return 100 * (p_win * (dec(odds) - 1) - (1 - p_win - p_push))


def fair_american(p):
    p = min(max(p, 0.001), 0.999)
    return f"{-round(100 * p / (1 - p)):+d}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.header("⚙️ Model Settings")
sims = st.sidebar.select_slider("Simulations", [10_000, 25_000, 50_000, 100_000], value=10_000)
pace = st.sidebar.slider("Projected possessions (pace)", 90.0, 105.0, 98.5, 0.5)
hca = st.sidebar.slider("Home-court advantage (pts)", 0.0, 4.0, 2.0, 0.25)
margin_sd = st.sidebar.slider("Std dev of final margin", 10.0, 15.0, 12.0, 0.5)
total_sd = st.sidebar.slider("Std dev of game total", 14.0, 24.0, 18.0, 0.5)
st.sidebar.caption("NBA margin SD is historically ~12 and total SD ~18. "
                   "The old version used 11.2 per team, which implied a ~15.8 margin SD (too wide).")

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
    st.error(f"Could not reach ESPN ({e}). Falling back to manual entry.")

all_teams = sorted(PRIORS.keys())
labels = ["✏️ Manual matchup"] + [f"{g['away']} @ {g['home']}" for g in games]
choice = st.selectbox("Select game", labels, index=1 if games else 0)

if choice == labels[0]:
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", all_teams, index=all_teams.index("Philadelphia 76ers"))
    away_team = c2.selectbox("Away team", all_teams, index=all_teams.index("Boston Celtics"))
    odds, gid = {"provider": "manual", "total": 224.5, "home_spread": 0.0, "home_ml": -110, "away_ml": -110,
                 "home_spread_odds": -110, "away_spread_odds": -110, "over_odds": -110, "under_odds": -110}, "manual"
else:
    g = games[labels.index(choice) - 1]
    home_team, away_team, odds, gid = g["home"], g["away"], g["odds"], g["id"]
    if g["state"] != "pre":
        st.warning("This game has started or finished — ESPN removes pregame lines once a game begins.")

if not games and live_ok:
    st.info("No games on this date (or the season hasn't started). Pick another date or use manual matchup.")

# =====================================================================
# MARKET LINES (auto-filled from ESPN, editable)
# =====================================================================
st.subheader(f"🎰 Pregame Market Lines  ·  source: {odds['provider']}")
if odds["home_spread"] is None or odds["total"] is None:
    st.caption("ESPN has no posted lines for this game yet — enter them manually.")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**Moneyline**")
    home_ml = st.number_input(f"{home_team} ML", value=int(odds["home_ml"] or -110), step=5, key=f"hml{gid}")
    away_ml = st.number_input(f"{away_team} ML", value=int(odds["away_ml"] or -110), step=5, key=f"aml{gid}")
with m2:
    st.markdown("**Spread** (home line; negative = home favored)")
    home_spread = st.number_input(f"{home_team} spread", value=float(odds["home_spread"] or 0.0), step=0.5, key=f"sp{gid}")
    hs_odds = st.number_input("Home spread odds", value=int(odds["home_spread_odds"]), step=5, key=f"hso{gid}")
    as_odds = st.number_input("Away spread odds", value=int(odds["away_spread_odds"]), step=5, key=f"aso{gid}")
with m3:
    st.markdown("**Total**")
    total_line = st.number_input("Over/Under", value=float(odds["total"] or 224.5), step=0.5, key=f"tot{gid}")
    over_odds = st.number_input("Over odds", value=int(odds["over_odds"]), step=5, key=f"oo{gid}")
    under_odds = st.number_input("Under odds", value=int(odds["under_odds"]), step=5, key=f"uo{gid}")

# =====================================================================
# INJURIES
# =====================================================================
ROLE_PTS = {"Star": 6.0, "Starter": 3.0, "Rotation": 1.2, "Deep bench": 0.0}
STATUS_W = {"out": 1.0, "doubtful": 0.85, "questionable": 0.5, "day-to-day": 0.3, "probable": 0.1}

st.subheader("🩹 Injury Report (live from ESPN)")
st.caption("ESPN doesn't provide player value, so set each player's **Role** — this is the biggest lever on accuracy. "
           "Impact = role points × chance they miss (Out=100%, Doubtful=85%, Questionable=50%, Day-to-Day=30%).")


def injury_panel(team, key):
    rows = injuries.get(team, [])
    if not rows:
        st.caption("No injuries reported." if live_ok else "Injury data unavailable.")
        return 0.0
    df = pd.DataFrame(rows)
    df["Role"] = "Rotation"
    edited = st.data_editor(
        df, key=key, hide_index=True,
        disabled=["Player", "Pos", "Status", "Note"],
        column_config={"Role": st.column_config.SelectboxColumn(options=list(ROLE_PTS), required=True)},
    )
    total = sum(ROLE_PTS[r] * STATUS_W.get(str(s).lower(), 0.3) for r, s in zip(edited["Role"], edited["Status"]))
    total = min(total, 15.0)
    st.caption(f"Estimated impact: **-{total:.1f} pts** of net rating")
    return total


i1, i2 = st.columns(2)
with i1:
    st.markdown(f"**🏠 {home_team}**")
    home_inj = injury_panel(home_team, f"hinj{gid}")
    home_b2b = st.checkbox("On a back-to-back", key=f"hb{gid}")
with i2:
    st.markdown(f"**✈️ {away_team}**")
    away_inj = injury_panel(away_team, f"ainj{gid}")
    away_b2b = st.checkbox("On a back-to-back", key=f"ab{gid}")

st.markdown("---")

# =====================================================================
# SIMULATION
# =====================================================================
if st.button(f"🔥 Run {sims:,} simulations"):
    if home_team == away_team:
        st.error("A team can't play itself.")
        st.stop()

    h_off, h_allowed, h_gp = team_strength(home_team, form)
    a_off, a_allowed, a_gp = team_strength(away_team, form)
    scalar = pace / 98.5

    home_pts = (h_off + a_allowed) / 2 * scalar + hca / 2
    away_pts = (a_off + h_allowed) / 2 * scalar - hca / 2

    # Injuries hurt the team's own offense AND defense, so split the effect.
    home_pts -= home_inj / 2; away_pts += home_inj / 2
    away_pts -= away_inj / 2; home_pts += away_inj / 2
    # Back-to-backs (~1.5 pt net effect)
    if home_b2b: home_pts -= 0.75; away_pts += 0.75
    if away_b2b: away_pts -= 0.75; home_pts += 0.75

    margin_mu, total_mu = home_pts - away_pts, home_pts + away_pts

    rng = np.random.default_rng()
    margin = np.rint(rng.normal(margin_mu, margin_sd, sims))
    total = np.rint(rng.normal(total_mu, total_sd, sims))
    ties = margin == 0                                  # no ties in the NBA (overtime)
    margin[ties] = rng.choice([-1, 1], ties.sum())
    home_sc, away_sc = (total + margin) / 2, (total - margin) / 2

    p_home = float(np.mean(margin > 0))
    p_away = 1 - p_home
    cover_h = float(np.mean(margin + home_spread > 0))
    cover_a = float(np.mean(margin + home_spread < 0))
    push_s = 1 - cover_h - cover_a
    over = float(np.mean(total > total_line))
    under = float(np.mean(total < total_line))
    push_t = 1 - over - under

    st.success(f"Completed {sims:,} simulations "
               f"(team data: {home_team} {h_gp} GP, {away_team} {a_gp} GP; early season leans on priors)")

    r1, r2, r3 = st.columns(3)
    r1.metric("🏆 Projected winner", home_team if p_home > 0.5 else away_team, f"{max(p_home, p_away) * 100:.1f}%")
    r2.metric("📏 Model spread (home)", f"{-np.median(margin):+.1f}", f"market {home_spread:+.1f}", delta_color="off")
    r3.metric("🏀 Model total", f"{np.median(total):.1f}", f"market {total_line}", delta_color="off")

    s1, s2 = st.columns(2)
    s1.metric(f"🏠 {home_team}", f"{home_sc.mean():.0f} PTS", f"{p_home * 100:.1f}% win · fair ML {fair_american(p_home)}", delta_color="off")
    s2.metric(f"✈️ {away_team}", f"{away_sc.mean():.0f} PTS", f"{p_away * 100:.1f}% win · fair ML {fair_american(p_away)}", delta_color="off")

    # ---- Edge table ----
    nv_ml_h, nv_ml_a = novig(home_ml, away_ml)
    nv_sp_h, nv_sp_a = novig(hs_odds, as_odds)
    nv_o, nv_u = novig(over_odds, under_odds)

    def row(market, side, line, o, model, push, mkt):
        return {"Market": market, "Side": side, "Line": line, "Odds": f"{o:+d}",
                "Model %": round(model * 100, 1), "Market % (no-vig)": round(mkt * 100, 1),
                "Edge (pts of prob)": round((model - mkt) * 100, 1),
                "EV per $100": round(ev100(model, push, o), 2)}

    table = pd.DataFrame([
        row("Moneyline", home_team, "", home_ml, p_home, 0, nv_ml_h),
        row("Moneyline", away_team, "", away_ml, p_away, 0, nv_ml_a),
        row("Spread", home_team, f"{home_spread:+.1f}", hs_odds, cover_h, push_s, nv_sp_h),
        row("Spread", away_team, f"{-home_spread:+.1f}", as_odds, cover_a, push_s, nv_sp_a),
        row("Total", "Over", f"{total_line}", over_odds, over, push_t, nv_o),
        row("Total", "Under", f"{total_line}", under_odds, under, push_t, nv_u),
    ])
    st.markdown("### 📊 Model vs. Market")
    st.dataframe(table, hide_index=True)
    st.caption("Market % removes the sportsbook's vig so it's comparable to the model. "
               "Edges above ~5 points almost always mean the model is missing something (usually injuries or "
               "stale ratings) rather than a true edge — the closing line is very hard to beat.")
