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
# FIXED MODEL SETTINGS (no sidebar)
# =====================================================================
SIMS = 10_000
HCA = 2.0          # home-court advantage, points
MARGIN_SD = 12.0   # std dev of final margin
TOTAL_SD = 18.0    # std dev of game total
LEAN_MIN, LEAN_STRONG = 0.02, 0.04   # edge thresholds (probability)

ROLE_PTS = {"Star": 6.0, "Starter": 3.0, "Rotation": 1.2, "Deep bench": 0.0}
STATUS_W = {"out": 1.0, "doubtful": 0.85, "questionable": 0.5, "day-to-day": 0.3, "probable": 0.1}

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
labels = [f"{g['away']} @ {g['home']}" for g in games] + ["✏️ Manual matchup"]
choice = st.selectbox("Select game", labels)

blank = {"provider": "manual", "total": None, "home_spread": None, "home_ml": None, "away_ml": None,
         "home_spread_odds": -110, "away_spread_odds": -110, "over_odds": -110, "under_odds": -110}
if choice == "✏️ Manual matchup":
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", all_teams, index=all_teams.index("Philadelphia 76ers"))
    away_team = c2.selectbox("Away team", all_teams, index=all_teams.index("Boston Celtics"))
    odds, gid = dict(blank), "manual"
else:
    g = games[labels.index(choice)]
    home_team, away_team, odds, gid = g["home"], g["away"], dict(g["odds"]), g["id"]
    if g["state"] != "pre":
        st.warning("Game has started or finished, so ESPN may no longer show pregame lines.")

if not games and live_ok:
    st.info("No games found for this date (or the season hasn't started). Pick another date or use manual matchup.")

# If the book has no lines posted, let the user type them in. Otherwise they're read-only.
if None in (odds["home_ml"], odds["away_ml"], odds["home_spread"], odds["total"]):
    with st.expander("No sportsbook lines found — enter them manually", expanded=True):
        a, b, c = st.columns(3)
        odds["home_ml"] = a.number_input(f"{home_team} ML", value=odds["home_ml"] or -110, step=5, key=f"hml{gid}")
        odds["away_ml"] = a.number_input(f"{away_team} ML", value=odds["away_ml"] or -110, step=5, key=f"aml{gid}")
        odds["home_spread"] = b.number_input(f"{home_team} spread", value=float(odds["home_spread"] or 0.0), step=0.5, key=f"sp{gid}")
        odds["total"] = c.number_input("Total", value=float(odds["total"] or 224.5), step=0.5, key=f"tot{gid}")

# =====================================================================
# INJURIES
# =====================================================================
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
    total = min(sum(ROLE_PTS[r] * STATUS_W.get(str(s).lower(), 0.3)
                    for r, s in zip(edited["Role"], edited["Status"])), 15.0)
    st.caption(f"Estimated impact: **-{total:.1f} pts**")
    return total


st.subheader("🩹 Injuries")
st.caption("Live from ESPN. Set the Role on key players (default is Rotation) so the model knows who actually matters.")
i1, i2 = st.columns(2)
with i1:
    st.markdown(f"**🏠 {home_team}**")
    home_inj = injury_panel(home_team, f"hinj{gid}")
    home_b2b = st.checkbox("Back-to-back", key=f"hb{gid}")
with i2:
    st.markdown(f"**✈️ {away_team}**")
    away_inj = injury_panel(away_team, f"ainj{gid}")
    away_b2b = st.checkbox("Back-to-back", key=f"ab{gid}")

st.markdown("---")

# =====================================================================
# SIMULATE
# =====================================================================
if st.button("🔥 Simulate Game", type="primary"):
    if home_team == away_team:
        st.error("A team can't play itself.")
        st.stop()

    h_off, h_allowed, _ = team_strength(home_team, form)
    a_off, a_allowed, _ = team_strength(away_team, form)
    home_pts = (h_off + a_allowed) / 2 + HCA / 2
    away_pts = (a_off + h_allowed) / 2 - HCA / 2
    home_pts -= home_inj / 2; away_pts += home_inj / 2
    away_pts -= away_inj / 2; home_pts += away_inj / 2
    if home_b2b: home_pts -= 0.75; away_pts += 0.75
    if away_b2b: away_pts -= 0.75; home_pts += 0.75

    rng = np.random.default_rng()
    margin = np.rint(rng.normal(home_pts - away_pts, MARGIN_SD, SIMS))
    total = np.rint(rng.normal(home_pts + away_pts, TOTAL_SD, SIMS))
    ties = margin == 0
    margin[ties] = rng.choice([-1, 1], ties.sum())

    line, tot_line = odds["home_spread"], odds["total"]
    p_home = float(np.mean(margin > 0))
    ch = float(np.mean(margin + line > 0))
    ca = float(np.mean(margin + line < 0))
    ov = float(np.mean(total > tot_line))
    un = float(np.mean(total < tot_line))

    nv_ml = novig(odds["home_ml"], odds["away_ml"])
    nv_sp = novig(odds["home_spread_odds"], odds["away_spread_odds"])
    nv_t = novig(odds["over_odds"], odds["under_odds"])

    def lean(edge):
        if edge >= LEAN_STRONG: return "🟢 Strong lean"
        if edge >= LEAN_MIN: return "🟡 Lean"
        return "—"

    def row(market, side, line_txt, book_odds, model_p, push, book_p):
        edge = model_p - book_p
        return {"Market": market, "Side": side, "Line": line_txt,
                "Model %": f"{model_p * 100:.1f}%", "Model odds": fair_american(model_p),
                "Sportsbook odds": f"{book_odds:+d}", "Book % (no-vig)": f"{book_p * 100:.1f}%",
                "Edge": f"{edge * 100:+.1f}", "EV/$100": f"{ev100(model_p, push, book_odds):+.1f}",
                "Model leans": lean(edge), "_edge": edge}

    push_s, push_t = 1 - ch - ca, 1 - ov - un
    rows = [
        row("Moneyline", home_team, "", odds["home_ml"], p_home, 0, nv_ml[0]),
        row("Moneyline", away_team, "", odds["away_ml"], 1 - p_home, 0, nv_ml[1]),
        row("Spread", home_team, f"{line:+.1f}", odds["home_spread_odds"], ch, push_s, nv_sp[0]),
        row("Spread", away_team, f"{-line:+.1f}", odds["away_spread_odds"], ca, push_s, nv_sp[1]),
        row("Total", "Over", f"{tot_line}", odds["over_odds"], ov, push_t, nv_t[0]),
        row("Total", "Under", f"{tot_line}", odds["under_odds"], un, push_t, nv_t[1]),
    ]

    # ---- Header ----
    st.subheader(f"{away_team} @ {home_team}")
    st.caption(f"Sportsbook: {odds['provider']}  ·  {SIMS:,} simulations")
    st.markdown(f"#### Projected score: {away_team} **{away_pts:.0f}** – {home_team} **{home_pts:.0f}**")

    # ---- Bottom line ----
    best = max(rows, key=lambda r: r["_edge"])
    if best["_edge"] >= LEAN_MIN:
        label = f"{best['Side']} {best['Line']}".strip()
        st.success(f"### ✅ Best edge: {best['Market']} — {label}\n"
                   f"Model {best['Model %']} vs sportsbook {best['Book % (no-vig)']}  ·  "
                   f"edge {best['Edge']} pts  ·  EV {best['EV/$100']} per $100")
    else:
        st.info("### No edge found\nThe model agrees with the sportsbook on all three markets.")

    # ---- One card per market: Model says / Sportsbook says / Edge ----
    def card(title, model_txt, book_txt, pair):
        a, b = pair
        v = a if a["_edge"] >= b["_edge"] else b
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.markdown("**🤖 MODEL SAYS**")
            st.markdown(model_txt)
            st.markdown("**🏦 SPORTSBOOK SAYS**")
            st.markdown(book_txt)
            st.markdown("**💰 EDGE**")
            vlabel = f"{v['Side']} {v['Line']}".strip()
            if v["_edge"] >= LEAN_STRONG:
                st.success(f"🟢 Strong edge on **{vlabel}**\n\n+{v['_edge'] * 100:.1f} pts  ·  EV {v['EV/$100']} per $100")
            elif v["_edge"] >= LEAN_MIN:
                st.warning(f"🟡 Small edge on **{vlabel}**\n\n+{v['_edge'] * 100:.1f} pts  ·  EV {v['EV/$100']} per $100")
            else:
                st.markdown("⚪ No edge — model and book agree")

    def fav_name(ml_h, ml_a):
        return home_team if ml_h < ml_a else away_team

    col1, col2, col3 = st.columns(3)

    # Moneyline
    ml_pick, ml_p = (home_team, p_home) if p_home >= 0.5 else (away_team, 1 - p_home)
    bk = fav_name(odds["home_ml"], odds["away_ml"])
    bk_p = nv_ml[0] if bk == home_team else nv_ml[1]
    bk_o = odds["home_ml"] if bk == home_team else odds["away_ml"]
    with col1:
        card("Moneyline",
             f"**{ml_pick}** wins\n\n{ml_p * 100:.1f}%  (fair odds {fair_american(ml_p)})",
             f"**{bk}** favored\n\n{bk_p * 100:.1f}%  (odds {bk_o:+d})",
             (rows[0], rows[1]))

    # Spread
    if ch >= ca:
        sp_pick, sp_line, sp_p = home_team, line, ch
    else:
        sp_pick, sp_line, sp_p = away_team, -line, ca
    bk_fav = home_team if line < 0 else away_team if line > 0 else "Pick'em"
    bk_line_txt = f"{bk_fav} {-abs(line):+.1f}" if line != 0 else "Pick'em (0)"
    with col2:
        card("Spread",
             f"**{sp_pick} {sp_line:+.1f}** covers\n\n{sp_p * 100:.1f}%  ·  model line: {home_team} {-np.median(margin):+.1f}",
             f"**{bk_line_txt}**\n\n{home_team} {line:+.1f} ({odds['home_spread_odds']:+d})  /  {away_team} {-line:+.1f} ({odds['away_spread_odds']:+d})",
             (rows[2], rows[3]))

    # Total
    t_pick, t_p = ("Over", ov) if ov >= un else ("Under", un)
    with col3:
        card("Total",
             f"**{t_pick} {tot_line}**\n\n{t_p * 100:.1f}%  ·  model total: {np.median(total):.1f}",
             f"**O/U {tot_line}**\n\nOver ({odds['over_odds']:+d})  /  Under ({odds['under_odds']:+d})",
             (rows[4], rows[5]))

    with st.expander("See full numbers"):
        st.dataframe(pd.DataFrame(rows).drop(columns="_edge"), hide_index=True)
        st.caption("Book % has the vig removed. Edge = model % minus book %, in percentage points. "
                   "A big edge usually means the model is missing something (an injury, stale ratings) "
                   "rather than a real opportunity.")
