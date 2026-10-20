import re
import math
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NBA Simulator", layout="wide")
st.title("🏀 NBA Simulator")
st.caption("Live ESPN schedule, lines and injuries · opponent-adjusted team ratings · 10,000 simulations")

# =====================================================================
# SETTINGS  (change these here; they are NOT on the page)
# =====================================================================
# Rating model settings. Run the Colab notebook (NBA_Model_v2) and copy the "Chosen settings"
# it prints into PARAMS. These defaults are sensible starting points, not tuned values.
PARAMS = {"half_life": 120.0, "ridge": 30.0, "prev_w": 0.6}
BETA_M, BETA_T = 0.30, 0.25      # how far the model may move the sportsbook's margin / total (from Colab, section 7)
SD_M, SD_T = 13.0, 18.0          # typical miss on the margin / the total (from Colab)
SIMS = 10_000
LEAN_MIN, LEAN_STRONG = 0.02, 0.04
FB_HISTORY_GAMES = 10            # first-basket: team games used to estimate each player
TIP_HOME_PROB = 0.50             # chance the home team wins the opening tip
FB_TURNOVER_RATE, FB_OREB_RATE = 0.13, 0.25
BACKTEST_NOTE = ""               # optional: paste your Colab backtest result here, e.g. "Last season its spread picks won 51.2%."

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ROLE_PTS = {"Star": 6.0, "Starter": 3.0, "Rotation": 1.2, "Deep bench": 0.0}
STATUS_W = {"out": 1.0, "doubtful": 0.85, "questionable": 0.5, "day-to-day": 0.3, "probable": 0.1}
BREAKEVEN = 110 / 210


# =====================================================================
# SMALL HELPERS
# =====================================================================
def _get(url, params=None, tries=3):
    last = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
    raise last


@st.cache_data(ttl=300, show_spinner=False)
def get_cached(url, params_items=()):
    return _get(url, dict(params_items) or None)


def to_american(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100
    try:
        return int(float(s.replace("+", "")))
    except ValueError:
        return None


def spread_from_details(details, home_abbr):
    details = (details or "").strip()
    m = re.match(r"^([A-Z]{2,4})\s+([+-]?\d+(\.\d+)?)$", details)
    if details.upper() in ("EVEN", "PK"):
        return 0.0
    if m:
        val = float(m.group(2))
        return val if m.group(1) == home_abbr else -val
    return None


def dec(o): return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)
def implied(o): return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)
def ev100(pw, pp, o): return 100 * (pw * (dec(o) - 1) - (1 - pw - pp))
def short(t): return t.split()[-1]
def pct(p): return f"{p * 100:.1f}%"
def md_money(x): return f"\\${x:,.0f}"          # backslash stops Streamlit treating $ as math


def novig(a, b):
    ia, ib = implied(a), implied(b)
    return ia / (ia + ib), ib / (ia + ib)


def fair_american(p):
    p = min(max(p, 0.001), 0.999)
    return f"{-round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def live_season(d):
    return d.year + 1 if d.month >= 9 else d.year       # ESPN names a season by the year it ends


# =====================================================================
# ESPN DATA (cached so the page stays fast)
# =====================================================================
def _fetch_day(d):
    try:
        data = _get(f"{BASE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 100}, tries=2)
    except Exception:
        return []
    out = []
    for ev in data.get("events", []):
        if not ev["status"]["type"].get("completed"):
            continue
        stype = (ev.get("season") or {}).get("type")
        if stype is not None and stype != 2:
            continue
        comp = ev["competitions"][0]
        h = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        a = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        try:
            hs, as_ = float(h["score"]), float(a["score"])
        except (KeyError, ValueError, TypeError):
            continue
        out.append({"date": d, "event_id": ev["id"], "home": h["team"]["displayName"],
                    "away": a["team"]["displayName"], "hs": hs, "as": as_})
    return out


@st.cache_data(ttl=3600, show_spinner="Downloading game history (about a minute the first time)...")
def load_history(season, cap_iso):
    cap = dt.date.fromisoformat(cap_iso)
    start, end = dt.date(season - 1, 10, 1), min(dt.date(season, 4, 20), cap)
    days = [start + dt.timedelta(days=i) for i in range(max((end - start).days + 1, 0))]
    with ThreadPoolExecutor(8) as ex:
        chunks = list(ex.map(_fetch_day, days))
    return [dict(g, season=season) for c in chunks for g in c]


def parse_odds(comp, home_abbr):
    o = (comp.get("odds") or [{}])[0]
    out = {"provider": (o.get("provider") or {}).get("name", "n/a"), "total": o.get("overUnder"),
           "home_spread": spread_from_details(o.get("details"), home_abbr), "home_ml": None, "away_ml": None,
           "home_spread_odds": -110, "away_spread_odds": -110, "over_odds": -110, "under_odds": -110}
    if out["home_spread"] is None and o.get("spread") is not None:
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
def get_slate(date_iso):
    d = dt.date.fromisoformat(date_iso)
    data = _get(f"{BASE}/scoreboard", {"dates": d.strftime("%Y%m%d")})
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        games.append({"id": ev["id"], "home": home["team"]["displayName"], "away": away["team"]["displayName"],
                      "home_id": home["team"].get("id"), "away_id": away["team"].get("id"),
                      "state": ev["status"]["type"]["state"],
                      "odds": parse_odds(comp, home["team"].get("abbreviation", ""))})
    return games


@st.cache_data(ttl=600, show_spinner=False)
def get_injuries():
    data = _get(f"{BASE}/injuries")
    out = {}
    for t in data.get("injuries", []):
        out[t.get("displayName")] = [{"player": (i.get("athlete") or {}).get("displayName", "?"),
                                      "pos": ((i.get("athlete") or {}).get("position") or {}).get("abbreviation", ""),
                                      "status": i.get("status", ""), "note": (i.get("shortComment") or "")[:120]}
                                     for i in t.get("injuries", [])]
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_roster(team_id):
    # IDs of players currently on the team, so traded/departed players are never used.
    if not team_id:
        return []
    try:
        data = _get(f"{BASE}/teams/{team_id}/roster", tries=2)
    except Exception:
        return []
    ids = set()

    def walk(x):
        if isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            if x.get("id") and x.get("displayName"):
                ids.add(str(x["id"]))
            for v in x.values():
                if isinstance(v, (list, dict)):
                    walk(v)
    walk(data.get("athletes", []))
    return sorted(ids)


# =====================================================================
# THE RATING MODEL (opponent-adjusted offense/defense, recent games count more)
# =====================================================================
def build_frame(rows):
    G = pd.DataFrame(rows).sort_values(["date", "event_id"]).reset_index(drop=True)
    G["ord"] = G["date"].map(lambda d: d.toordinal())
    last, cur, hb, ab = {}, None, [], []
    for r in G.itertuples():
        if r.season != cur:
            last, cur = {}, r.season
        hb.append(1.0 if last.get(r.home) == r.ord - 1 else 0.0)
        ab.append(1.0 if last.get(r.away) == r.ord - 1 else 0.0)
        last[r.home] = r.ord
        last[r.away] = r.ord
    G["hb"], G["ab"] = hb, ab
    teams = sorted(set(G["home"]) | set(G["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    T, N = len(teams), len(G)
    X = np.zeros((2 * N, 4 + 2 * T))
    y = np.zeros(2 * N)
    r = np.arange(N)
    hi, ai = G["home"].map(idx).values, G["away"].map(idx).values
    X[:, 0] = 1
    X[2 * r, 1] = 1
    X[2 * r, 2], X[2 * r, 3] = G["hb"].values, G["ab"].values
    X[2 * r, 4 + hi] = 1
    X[2 * r, 4 + T + ai] = 1
    X[2 * r + 1, 2], X[2 * r + 1, 3] = G["ab"].values, G["hb"].values
    X[2 * r + 1, 4 + ai] = 1
    X[2 * r + 1, 4 + T + hi] = 1
    y[2 * r], y[2 * r + 1] = G["hs"].values, G["as"].values
    return {"G": G, "X": X, "y": y, "idx": idx, "T": T,
            "row_ord": np.repeat(G["ord"].values, 2), "row_season": np.repeat(G["season"].values, 2)}


def build_hist(G):
    h = {}
    for r in G.itertuples():
        h.setdefault(r.home, []).append((r.ord, r.event_id))
        h.setdefault(r.away, []).append((r.ord, r.event_id))
    return h


@st.cache_data(ttl=3600, show_spinner="Fitting team ratings...")
def fit_model(date_iso):
    d = dt.date.fromisoformat(date_iso)
    season = live_season(d)
    cap = (d - dt.timedelta(days=1)).isoformat()
    rows = load_history(season - 1, cap) + load_history(season, cap)
    if len(rows) < 400:
        return None
    F = build_frame(rows)
    d_ord = d.toordinal()
    mask = F["row_ord"] < d_ord
    age = d_ord - F["row_ord"][mask]
    w = 0.5 ** (age / PARAMS["half_life"]) * PARAMS["prev_w"] ** (season - F["row_season"][mask])
    X, y = F["X"][mask], F["y"][mask]
    pen = np.zeros(X.shape[1])
    pen[4:], pen[:4] = PARAMS["ridge"], 1e-3
    beta = np.linalg.solve(X.T @ (w[:, None] * X) + np.diag(pen), X.T @ (w * y))
    G = F["G"]
    cur = G[G["season"] == season]
    return {"beta": beta, "idx": F["idx"], "T": F["T"], "hist": build_hist(G), "season": season,
            "cur_counts": pd.concat([cur["home"], cur["away"]]).value_counts().to_dict(),
            "n_games": int(mask.sum() / 2)}


def team_score(M, own, opp, is_home, own_b2b, opp_b2b):
    b, idx, T = M["beta"], M["idx"], M["T"]
    x = b[0] + b[1] * is_home + b[2] * own_b2b + b[3] * opp_b2b
    if own in idx:
        x += b[4 + idx[own]]
    if opp in idx:
        x += b[4 + T + idx[opp]]
    return x


def injury_impact(team, injuries, roles):
    total = 0.0
    for r in injuries.get(team, []):
        role = roles.get(r["player"], "Rotation")
        pts = ROLE_PTS.get(role, 1.2) * STATUS_W.get(r["status"].lower(), 0.3)
        total += pts
    return min(total, 15.0)


def analyze_game(M, g, o, h_inj, a_inj, b2b):
    home, away = g["home"], g["away"]
    hb, ab = (1.0 if home in b2b else 0.0), (1.0 if away in b2b else 0.0)
    hp = team_score(M, home, away, 1, hb, ab)
    ap = team_score(M, away, home, 0, ab, hb)
    hp -= h_inj / 2; ap += h_inj / 2          # injuries hurt offense AND defense
    ap -= a_inj / 2; hp += a_inj / 2
    raw_m, raw_t = hp - ap, hp + ap
    has_s, has_t = o["home_spread"] is not None, o["total"] is not None
    m_mu = (-o["home_spread"]) + BETA_M * (raw_m - (-o["home_spread"])) if has_s else raw_m
    t_mu = o["total"] + BETA_T * (raw_t - o["total"]) if has_t else raw_t
    rng = np.random.default_rng()
    margin = np.rint(rng.normal(m_mu, SD_M, SIMS))
    total = np.rint(rng.normal(t_mu, SD_T, SIMS))
    ties = margin == 0
    margin[ties] = rng.choice([-1, 1], ties.sum())
    rows = []

    def add(mk, side, line, bo, mp, push, bp):
        rows.append({"market": mk, "side": side, "line": line, "book_odds": bo, "model_p": mp, "book_p": bp,
                     "edge": mp - bp, "ev": ev100(mp, push, bo)})
    ph = float(np.mean(margin > 0))
    if o["home_ml"] is not None and o["away_ml"] is not None:
        nv = novig(o["home_ml"], o["away_ml"])
        add("Moneyline", home, "", o["home_ml"], ph, 0, nv[0])
        add("Moneyline", away, "", o["away_ml"], 1 - ph, 0, nv[1])
    if has_s:
        L = o["home_spread"]
        ch, ca = float(np.mean(margin + L > 0)), float(np.mean(margin + L < 0))
        nv = novig(o["home_spread_odds"], o["away_spread_odds"])
        add("Spread", home, f"{L:+.1f}", o["home_spread_odds"], ch, 1 - ch - ca, nv[0])
        add("Spread", away, f"{-L + 0.0:+.1f}", o["away_spread_odds"], ca, 1 - ch - ca, nv[1])
    if has_t:
        T_ = o["total"]
        ov, un = float(np.mean(total > T_)), float(np.mean(total < T_))
        nv = novig(o["over_odds"], o["under_odds"])
        add("Total", "Over", f"{T_}", o["over_odds"], ov, 1 - ov - un, nv[0])
        add("Total", "Under", f"{T_}", o["under_odds"], un, 1 - ov - un, nv[1])
    return {"game": g, "odds": o, "rows": rows, "raw_m": raw_m, "raw_t": raw_t,
            "med_m": float(np.median(margin)), "med_t": float(np.median(total)),
            "hp": (t_mu + m_mu) / 2, "ap": (t_mu - m_mu) / 2}


# =====================================================================
# RENDERING THE GAME RESULT
# =====================================================================
def strength(e):
    return "🟢 Strong value" if e >= LEAN_STRONG else ("🟡 Slight value" if e >= LEAN_MIN else None)


def by_txt(home, away, m):
    return f"{short(home)} by {m:.1f}" if m >= 0 else f"{short(away)} by {-m:.1f}"


def market_card(res, market):
    rows = [r for r in res["rows"] if r["market"] == market]
    with st.container(border=True):
        st.markdown(f"### {market}")
        if not rows:
            st.caption("No sportsbook line posted yet.")
            return
        g = res["game"]
        a, b = rows
        top = a if a["model_p"] >= b["model_p"] else b
        best = a if a["edge"] >= b["edge"] else b
        if market == "Moneyline":
            fav = a if a["book_odds"] < b["book_odds"] else b
            model = f"**{short(top['side'])}** have **{pct(top['model_p'])}** odds to win.\n\nFair odds: {fair_american(top['model_p'])}"
            book = f"**{short(fav['side'])}** are favored at **{fav['book_odds']:+d}**.\n\nThat implies **{pct(fav['book_p'])}** to win."
            name = f"{short(best['side'])} moneyline"
        elif market == "Spread":
            model = (f"**{short(top['side'])} {top['line']}** have **{pct(top['model_p'])}** odds to cover.\n\n"
                     f"Model projects: {by_txt(g['home'], g['away'], res['med_m'])}")
            book = (f"{short(a['side'])} {a['line']} at {a['book_odds']:+d}\n\n{short(b['side'])} {b['line']} at {b['book_odds']:+d}")
            name = f"{short(best['side'])} {best['line']}"
        else:
            model = f"**{top['side']} {top['line']}** has **{pct(top['model_p'])}** odds to hit.\n\nModel projects: {res['med_t']:.0f} total points"
            book = f"The total is set at **{top['line']}**.\n\nOver at {a['book_odds']:+d}\n\nUnder at {b['book_odds']:+d}"
            name = f"{best['side']} {best['line']}"
        st.markdown("**🤖 MODEL SAYS**")
        st.markdown(model)
        st.markdown("**🏦 SPORTSBOOK SAYS**")
        st.markdown(book)
        st.markdown("**💰 IS THERE VALUE?**")
        lab = strength(best["edge"])
        if lab:
            msg = (f"{lab}: **{name}**\n\nThe sportsbook's odds imply this hits **{pct(best['book_p'])}** of the time. "
                   f"The model says **{pct(best['model_p'])}**.\n\nIf the model is right, a {md_money(100)} bet wins about "
                   f"**{md_money(best['ev'])}** on average over many bets.")
            (st.success if best["edge"] >= LEAN_STRONG else st.warning)(msg)
        else:
            st.markdown("⚪ **No value.** The model and sportsbook agree.")


def render_result(res):
    g = res["game"]
    st.subheader(f"{g['away']} @ {g['home']}")
    st.caption(f"Sportsbook: {res['odds']['provider']} · {SIMS:,} simulations")
    st.markdown(f"#### Projected score: {g['away']} **{res['ap']:.0f}**, {g['home']} **{res['hp']:.0f}**")
    st.caption(f"Raw rating-based projection (before blending with the sportsbook line): "
               f"{by_txt(g['home'], g['away'], res['raw_m'])}, {res['raw_t']:.0f} total points.")
    best = max(res["rows"], key=lambda r: r["edge"]) if res["rows"] else None
    if best and best["edge"] >= LEAN_MIN:
        label = best["side"] if best["market"] == "Moneyline" else f"{best['side']} {best['line']}"
        st.success(f"### ✅ Best value: {best['market']}: {label}\n"
                   f"Model gives it **{pct(best['model_p'])}**, the sportsbook's odds only imply **{pct(best['book_p'])}**.")
    elif res["rows"]:
        st.info("### No clear value\nThe model and the sportsbook roughly agree on all three markets.")
    note = (BACKTEST_NOTE + " ") if BACKTEST_NOTE else ""
    st.caption(f"Reality check: {note}Sportsbook lines are very hard to beat. Treat these as small tilts, not locks.")
    cols = st.columns(3)
    for col, market in zip(cols, ["Moneyline", "Spread", "Total"]):
        with col:
            market_card(res, market)


# =====================================================================
# FIRST BASKET SIMULATOR
# =====================================================================
def _minutes(x):
    try:
        return float(str(x).split(":")[0])
    except ValueError:
        return 0.0


def _fetch_box_raw(event_id):
    try:
        data = _get(f"{BASE}/summary", {"event": event_id}, tries=2)
    except Exception:
        return None
    players = {}
    for t in (data.get("boxscore") or {}).get("players", []):
        team = (t.get("team") or {}).get("displayName")
        for grp in t.get("statistics", []):
            names = grp.get("names") or grp.get("labels") or grp.get("keys") or []
            for a in grp.get("athletes", []):
                stt = a.get("stats") or []
                if not stt or a.get("didNotPlay"):
                    continue
                row = dict(zip(names, stt))
                ath = a.get("athlete") or {}
                m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", str(row.get("FG", "")))
                players[str(ath.get("id"))] = {
                    "name": ath.get("displayName", "?"), "team": team, "starter": bool(a.get("starter")),
                    "min": _minutes(row.get("MIN", 0)),
                    "fgm": float(m.group(1)) if m else 0.0, "fga": float(m.group(2)) if m else 0.0}
    return {"players": players}


@st.cache_data(ttl=86400, show_spinner="Loading recent box scores...")
def load_boxes(event_ids):
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(_fetch_box_raw, event_ids))
    return dict(zip(event_ids, res))


def recent_events(hist, team, before_ord, n):
    return [e for (o, e) in hist.get(team, []) if o < before_ord][-n:]


def team_profile(evs, boxes, team):
    agg = {}
    for i, e in enumerate(evs):
        box = boxes.get(e)
        if not box:
            continue
        w = 0.8 ** (len(evs) - 1 - i)           # recent games count more when deciding who starts
        for aid, p in box["players"].items():
            if p["team"] != team:
                continue
            a = agg.setdefault(aid, {"name": p["name"], "wstarts": 0.0, "min": 0.0, "fga": 0.0, "fgm": 0.0})
            a["wstarts"] += w if p["starter"] else 0.0
            a["min"] += p["min"]
            a["fga"] += p["fga"]
            a["fgm"] += p["fgm"]
    return agg


def player_params(aid, name, a):
    if not a:
        return {"id": aid, "name": name, "usage": 0.44, "fg": 0.47}
    return {"id": aid, "name": name, "usage": (a["fga"] + 0.44 * 60) / (a["min"] + 60),
            "fg": (a["fgm"] + 0.47 * 40) / (a["fga"] + 40)}


def simulate_first_basket(home5, away5, n):
    rng = np.random.default_rng()
    cum, fg = np.zeros((2, 5)), np.zeros((2, 5))
    for t, ps in enumerate((home5, away5)):
        u = np.array([p["usage"] for p in ps])
        cum[t] = np.cumsum(u / u.sum())
        fg[t] = [p["fg"] for p in ps]
    cum[:, -1] = 1.0
    off = np.where(rng.random(n) < TIP_HOME_PROB, 0, 1)
    first_off = off.copy()
    winner, scorer = np.full(n, -1), np.full(n, -1)
    active = np.ones(n, bool)
    for _ in range(100):
        if not active.any():
            break
        idx = np.where(active)[0]
        o = off[idx]
        u1, u2, u3, u4 = rng.random((4, idx.size))
        to = u1 < FB_TURNOVER_RATE                              # turnover: no shot
        sh = (u2[:, None] > cum[o]).sum(axis=1)                 # who shoots
        made = (~to) & (u3 < fg[o, sh])
        mi = idx[made]
        winner[mi], scorer[mi] = o[made], sh[made]
        active[mi] = False
        keep = (~made) & (~to) & (u4 < FB_OREB_RATE)            # offensive rebound: same team again
        sw = (~made) & (~keep)
        off[idx[sw]] = 1 - off[idx[sw]]
    done = winner >= 0
    return first_off, winner, scorer, done


def parse_odds_text(text):
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(.*?)[,\s]+([+-]?\d{2,5})\s*$", line)
        if m:
            out[m.group(1).strip().lower()] = int(m.group(2))
    return out


def first_basket_tab(M, g, injuries, now):
    home, away = g["home"], g["away"]
    st.caption("Simulates the opening tip and each possession until someone makes a field goal. "
               "Free throws do not count as the first basket (sportsbooks differ, so check your book's rules).")
    evs = {t: recent_events(M["hist"], t, now.toordinal(), FB_HISTORY_GAMES) for t in (home, away)}
    boxes = load_boxes(tuple(sorted(set(evs[home]) | set(evs[away]))))
    if not any(boxes.values()):
        st.error("Could not read box scores from ESPN. The first basket simulator needs them.")
        return
    chosen, aggs = {}, {}
    cols = st.columns(2)
    for col, team, tid in ((cols[0], home, g.get("home_id")), (cols[1], away, g.get("away_id"))):
        with col:
            st.markdown(f"**{team}**")
            agg = team_profile(evs[team], boxes, team)
            roster = set(get_roster(tid))
            if roster:
                agg = {k: a for k, a in agg.items() if k in roster}     # current roster only
            out_names = {r["player"].lower() for r in injuries.get(team, []) if r["status"].lower() in ("out", "doubtful")}
            ranked = sorted(agg.items(), key=lambda kv: (-kv[1]["wstarts"], -kv[1]["min"]))
            default = [kv[1]["name"] for kv in ranked if kv[1]["name"].lower() not in out_names][:5]
            options = [kv[1]["name"] for kv in sorted(agg.items(), key=lambda kv: -kv[1]["min"])][:15]
            for nm in default:
                if nm not in options:
                    options.append(nm)
            chosen[team] = st.multiselect(f"Starting five ({team})", options, default=default,
                                          key=f"fb_{g['id']}_{team}")
            aggs[team] = agg
            if M["cur_counts"].get(team, 0) < 5:
                st.warning("Fewer than 5 games played this season, so this lineup is a GUESS from older games. "
                           "Set the real starting five above once lineups are announced (about 30 min before tip).")
    odds_text = st.text_area("Your sportsbook's first-basket prices (optional), one per line, like: Jalen Brunson, 650",
                             key=f"fbo_{g['id']}")
    if st.button("🏀 Simulate first basket (10,000x)", type="primary", key=f"fbb_{g['id']}"):
        if len(chosen[home]) != 5 or len(chosen[away]) != 5:
            st.error("Pick exactly 5 starters for each team.")
            return
        fives = []
        for team in (home, away):
            by_name = {a["name"]: (aid, a) for aid, a in aggs[team].items()}
            five = []
            for nm in chosen[team]:
                aid, a = by_name.get(nm, (nm, None))        # unknown player: league-average shooting
                five.append(player_params(aid, nm, a))
            fives.append(five)
        first_off, winner, scorer, done = simulate_first_basket(fives[0], fives[1], SIMS)
        rows = []
        for t, (team, five) in enumerate(((home, fives[0]), (away, fives[1]))):
            for i, p in enumerate(five):
                rows.append({"Player": p["name"], "Team": short(team), "prob": float(np.mean((winner == t) & (scorer == i)))})
        T = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)
        st.session_state[f"fbres_{g['id']}"] = {
            "T": T, "home_first": float(np.mean(winner[done] == 0)), "tip_first": float(np.mean(winner[done] == first_off[done])),
            "odds": parse_odds_text(odds_text)}
    res = st.session_state.get(f"fbres_{g['id']}")
    if res:
        T, book = res["T"], res["odds"]
        top = T.iloc[0]
        st.markdown(f"### Across {SIMS:,} simulated openings, **{top['Player']}** is the most likely first scorer at "
                    f"**{pct(top['prob'])}** (fair odds {fair_american(top['prob'])}).")
        st.markdown(f"**{short(home)}** score the first basket **{pct(res['home_first'])}** of the time and "
                    f"**{short(away)}** **{pct(1 - res['home_first'])}**. The team that wins the tip scores first "
                    f"**{pct(res['tip_first'])}** of the time (this comes from the model's assumptions, not checked against real games).")
        out = []
        for r in T.itertuples():
            row = {"Player": r.Player, "Team": r.Team, "Chance to score first": pct(r.prob), "Fair odds": fair_american(r.prob)}
            o = book.get(r.Player.lower())
            if o is not None:
                ev = ev100(r.prob, 0, o)
                row["Your odds"] = f"{o:+d}"
                row["Book implies"] = pct(implied(o))
                row["Profit per $100 if model is right"] = f"${ev:,.0f}" if ev >= 0 else f"-${-ev:,.0f}"
                row["Worth it?"] = "🟢 Yes" if ev > 0 else "No"
            out.append(row)
        st.dataframe(pd.DataFrame(out), hide_index=True)
        st.caption("First-basket prices carry a large built-in sportsbook fee, so a price has to beat the fair odds by a wide margin "
                   "to be worth taking. 'Worth it?' assumes this simulator is accurate, and its accuracy on real games has only been "
                   "checked in the Colab notebook.")


# =====================================================================
# RESULTS LOG (saves every pick, grades it from ESPN's final scores)
# =====================================================================
LOG_PATH = "picks_log.csv"
LOG_COLS = ["key", "date", "event_id", "game", "market", "side", "side_role", "line", "home_spread", "total_line",
            "odds", "model_p", "book_p", "edge", "ev", "flagged", "strength", "saved_at", "result", "profit"]


def empty_log():
    return pd.DataFrame(columns=LOG_COLS)


def load_log():
    try:
        df = pd.read_csv(LOG_PATH, dtype={"event_id": str, "key": str})
    except Exception:
        df = st.session_state.get("log_df", empty_log())
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[LOG_COLS].copy()
    df["flagged"] = df["flagged"].map(lambda v: str(v).lower() in ("true", "1", "1.0"))
    return df


def save_log(df):
    st.session_state["log_df"] = df
    try:
        df.to_csv(LOG_PATH, index=False)
        return True
    except Exception:
        return False


def rows_to_log(res, date_iso):
    g, o = res["game"], res["odds"]
    out = []
    for r in res["rows"]:
        role = ("home" if r["side"] == g["home"] else "away") if r["market"] != "Total" else r["side"].lower()
        lab = strength(r["edge"])
        out.append({"key": f"{date_iso}|{g['id']}|{r['market']}|{role}", "date": date_iso, "event_id": str(g["id"]),
                    "game": f"{g['away']} @ {g['home']}", "market": r["market"], "side": r["side"], "side_role": role,
                    "line": r["line"], "home_spread": o["home_spread"] if r["market"] == "Spread" else np.nan,
                    "total_line": o["total"] if r["market"] == "Total" else np.nan,
                    "odds": r["book_odds"], "model_p": r["model_p"], "book_p": r["book_p"], "edge": r["edge"], "ev": r["ev"],
                    "flagged": r["edge"] >= LEAN_MIN, "strength": ("Strong" if "Strong" in lab else "Slight") if lab else "",
                    "saved_at": dt.datetime.now().isoformat(timespec="seconds"), "result": np.nan, "profit": np.nan})
    return out


def upsert_log(log, new_rows):
    # Re-saving the same slate updates picks that are not graded yet; graded picks are never changed.
    new = pd.DataFrame(new_rows, columns=LOG_COLS)
    graded_keys = set(log.loc[log["result"].isin(["W", "L", "P"]), "key"])
    new = new[~new["key"].isin(graded_keys)]
    keep = log[~log["key"].isin(set(new["key"]))]
    return pd.concat([keep, new], ignore_index=True)[LOG_COLS]


@st.cache_data(ttl=600, show_spinner=False)
def get_finals(date_iso):
    return {g["event_id"]: g for g in _fetch_day(dt.date.fromisoformat(date_iso))}


def result_for(row, f):
    margin, total = f["hs"] - f["as"], f["hs"] + f["as"]
    role, mk = row["side_role"], row["market"]
    if mk == "Moneyline":
        won, push = (margin > 0 if role == "home" else margin < 0), False
    elif mk == "Spread":
        cover = margin + float(row["home_spread"])
        push, won = cover == 0, (cover > 0 if role == "home" else cover < 0)
    else:
        diff = total - float(row["total_line"])
        push, won = diff == 0, (diff > 0 if role == "over" else diff < 0)
    if push:
        return "P", 0.0
    return ("W", 100 * (dec(float(row["odds"])) - 1)) if won else ("L", -100.0)


def grade_log(log):
    log = log.copy()
    n_new = 0
    pending = log[~log["result"].isin(["W", "L", "P"])]
    for d in sorted(pending["date"].dropna().unique()):
        finals = get_finals(str(d))
        for i in pending.index[pending["date"] == d]:
            f = finals.get(str(log.at[i, "event_id"]))
            if f:
                log.at[i, "result"], log.at[i, "profit"] = result_for(log.loc[i], f)
                n_new += 1
    return log, n_new


def record_line(df):
    w, l, p = (df["result"] == "W").sum(), (df["result"] == "L").sum(), (df["result"] == "P").sum()
    return f"{w}-{l}" + (f"-{p}" if p else "")


def summarize_picks(df):
    d = df[df["result"].isin(["W", "L"])]
    n = len(d)
    wins = int((d["result"] == "W").sum())
    return {"Picks": len(df), "Record (W-L-P)": record_line(df),
            "Win %": f"{wins / n * 100:.1f}%" if n else "-",
            "Break-even %": f"{d['odds'].astype(float).map(implied).mean() * 100:.1f}%" if n else "-",
            "Profit ($100 per pick)": f"${df['profit'].astype(float).sum():,.0f}",
            "ROI": f"{df['profit'].astype(float).sum() / (100 * n) * 100:+.1f}%" if n else "-"}


def results_tab():
    st.caption("Every pick you save is graded automatically from ESPN's final scores. This is the real test of the model, "
               "because it uses the actual lines and injuries from before each game.")
    log = load_log()
    if len(log):
        with st.spinner("Checking final scores..."):
            log, n_new = grade_log(log)
        if n_new:
            save_log(log)
            st.success(f"Graded {n_new} new picks.")
    graded = log[log["result"].isin(["W", "L", "P"])].copy()
    for c in ("odds", "model_p", "book_p", "edge", "profit"):
        graded[c] = pd.to_numeric(graded[c])

    if log.empty:
        st.info("Nothing saved yet. Open the **Full slate** tab, run it, and press **Save picks to my results log** before games start.")
    else:
        flagged = graded[graded["flagged"]]
        st.markdown("### 💰 The value picks (the ones the app flagged)")
        if flagged.empty:
            st.caption("No graded value picks yet. Check back after the saved games finish.")
        else:
            sm = summarize_picks(flagged)
            cols = st.columns(len(sm))
            for col, (k, v) in zip(cols, sm.items()):
                col.metric(k, v)
            n_dec = int(flagged["result"].isin(["W", "L"]).sum())
            if n_dec < 100:
                st.info(f"Only {n_dec} graded value picks so far, which is far too few to judge. Results swing wildly in small samples. "
                        "Wait for several hundred before drawing conclusions.")
            else:
                roi = flagged["profit"].sum() / (100 * n_dec)
                st.info(("Value picks are profitable so far, but check that it holds up over more picks and isn't a few lucky bets. "
                         if roi > 0 else "Value picks are not beating the sportsbook so far. If this holds with a few hundred more picks, the model has no real edge. ")
                        + "Note that this assumes you could get the logged odds.")
            for label, col in (("By market", "market"), ("By strength", "strength")):
                rows = {str(k): summarize_picks(flagged[flagged[col] == k]) for k in flagged[col].dropna().unique() if str(k)}
                if rows:
                    st.markdown(f"**{label}**")
                    st.dataframe(pd.DataFrame(rows).T)

        nonpush = graded[graded["result"].isin(["W", "L"])].copy()
        st.markdown("### 🎯 Is the model more accurate than the sportsbook?")
        if len(nonpush) < 40:
            st.caption("Needs at least 40 graded sides. This one uses every side of every saved market, not just flagged picks.")
        else:
            y = (nonpush["result"] == "W").astype(float)
            b_model, b_book = ((nonpush["model_p"] - y) ** 2).mean(), ((nonpush["book_p"] - y) ** 2).mean()
            c1, c2 = st.columns(2)
            c1.metric("Model error (Brier, lower is better)", f"{b_model:.4f}")
            c2.metric("Sportsbook error (Brier, lower is better)", f"{b_book:.4f}")
            st.caption("If the model's error is not clearly lower than the sportsbook's, the model adds nothing beyond the line. "
                       "The difference needs a few hundred games to mean anything.")
            nonpush["bucket"] = pd.cut(nonpush["model_p"], [0, .3, .4, .45, .5, .55, .6, .7, 1.0])
            cal = nonpush.assign(won=y).groupby("bucket", observed=True).agg(
                picks=("won", "size"), model_said=("model_p", "mean"), actually_won=("won", "mean")).round(3)
            st.markdown("**Calibration: when the model says X%, how often does it happen?**")
            st.dataframe(cal)

    with st.expander("My saved picks, backup and restore"):
        st.caption("On Streamlit Cloud the saved file can be erased when the app restarts or is redeployed. "
                   "Download a backup now and then and upload it here to restore. Anyone who opens this site can see and change the log.")
        if len(log):
            st.dataframe(log.drop(columns=["key"]).sort_values("date", ascending=False), hide_index=True)
        st.download_button("⬇️ Download my log (CSV)", log.to_csv(index=False), file_name="picks_log.csv", mime="text/csv")
        up = st.file_uploader("Restore from a saved log (CSV)", type="csv")
        if up is not None and st.button("Merge uploaded log"):
            try:
                other = pd.read_csv(up, dtype={"event_id": str, "key": str})
                for c in LOG_COLS:
                    if c not in other.columns:
                        other[c] = np.nan
                merged = pd.concat([log, other[LOG_COLS]], ignore_index=True)
                merged["_g"] = merged["result"].isin(["W", "L", "P"])
                merged = merged.sort_values("_g", ascending=False).drop_duplicates("key").drop(columns="_g")
                save_log(merged[LOG_COLS])
                st.success(f"Log restored: {len(merged)} picks.")
            except Exception as e:
                st.error(f"Could not read that file: {e}")
        confirm = st.checkbox("I'm sure I want to erase my whole log")
        if st.button("🗑️ Erase log") and confirm:
            save_log(empty_log())
            st.success("Log erased.")


# =====================================================================
# FULL SLATE
# =====================================================================
def pick_name(r):
    if r["market"] == "Moneyline":
        return f"{short(r['side'])} moneyline"
    if r["market"] == "Total":
        return f"{r['side']} {r['line']}"
    return f"{short(r['side'])} {r['line']}"


def slate_tab(M, date, games, injuries, b2b):
    st.caption("Runs every game on the date and ranks the best spots. Injured players are all treated as Rotation-level here, "
               "so for a game where a star is out, open the **Game odds** tab and set his role.")
    if not games:
        st.info("No games on this date.")
        return
    key = f"slate_{date.isoformat()}"
    if st.button("🔥 Run full slate (10,000x per game)", type="primary", key="run_slate"):
        with st.spinner(f"Simulating {len(games)} games..."):
            st.session_state[key] = [analyze_game(M, g, g["odds"], injury_impact(g["home"], injuries, {}),
                                                  injury_impact(g["away"], injuries, {}), b2b) for g in games]
    results = st.session_state.get(key)
    if not results:
        return
    flagged = sorted([(res, r) for res in results for r in res["rows"] if r["edge"] >= LEAN_MIN], key=lambda x: -x[1]["edge"])
    st.markdown("### 🎯 Best spots today")
    if not flagged:
        st.info("No value found: the model and the sportsbook agree across the slate (or no lines are posted yet).")
    else:
        st.dataframe(pd.DataFrame([{
            "Game": f"{short(res['game']['away'])} @ {short(res['game']['home'])}", "Market": r["market"], "Pick": pick_name(r),
            "Odds": f"{r['book_odds']:+d}", "Model chance": pct(r["model_p"]), "Sportsbook implies": pct(r["book_p"]),
            "Edge (pts)": f"{r['edge'] * 100:+.1f}", "Profit per $100 if model is right": f"${r['ev']:,.0f}",
            "Value": strength(r["edge"])} for res, r in flagged]), hide_index=True)
        st.caption("Sorted by edge. Big edges usually mean the model is missing something (often an injury), not a real opportunity. "
                   "Sportsbook lines are very hard to beat.")

    pre = [res for res in results if res["game"]["state"] == "pre" and res["rows"]]
    st.markdown("### 💾 Track these picks")
    st.caption(f"Saves every market side for the {len(pre)} games that haven't started (flagged picks and the rest, so the model's accuracy "
               "can be checked too). Do this BEFORE games start. The Results tab grades them afterward.")
    if not pre:
        st.info("Every game has already started or has no lines posted, so there is nothing new to save.")
    elif st.button("💾 Save picks to my results log", key="save_slate"):
        log = load_log()
        new_rows = [row for res in pre for row in rows_to_log(res, date.isoformat())]
        log = upsert_log(log, new_rows)
        ok = save_log(log)
        st.success(f"Saved {len(new_rows)} market sides from {len(pre)} games." + ("" if ok else " (Could not write the file, so they're kept for this session only. Download a backup in the Results tab.)"))

    st.markdown("### Every game")
    for res in results:
        g = res["game"]
        best = max(res["rows"], key=lambda r: r["edge"]) if res["rows"] else None
        tag = f"  ·  {strength(best['edge'])}: {pick_name(best)}" if best and best["edge"] >= LEAN_MIN else ""
        with st.expander(f"{g['away']} @ {g['home']}{tag}"):
            render_result(res)


# =====================================================================
# THE PAGE
# =====================================================================
date = st.date_input("Game date", dt.date.today())
M = fit_model(date.isoformat())
if M is None:
    st.error("Could not download enough game history from ESPN to build team ratings. Try again in a minute.")
    st.stop()
st.caption(f"Ratings built from {M['n_games']:,} games (this season and last), most recent weighted heaviest.")

try:
    games = get_slate(date.isoformat())
    injuries = get_injuries()
    live_ok = True
except Exception as e:
    games, injuries, live_ok = [], {}, False
    st.error(f"Could not reach ESPN ({e}). Use the manual matchup.")

try:
    prev_games = get_slate((date - dt.timedelta(days=1)).isoformat())
    b2b_teams = {x for pg in prev_games for x in (pg["home"], pg["away"])}
except Exception:
    b2b_teams = set()

labels = [f"{g['away']} @ {g['home']}" for g in games] + ["✏️ Manual matchup"]
choice = st.selectbox("Select game", labels)
if not games and live_ok:
    st.info("No games found for this date (the season may not have started). Pick another date or use the manual matchup.")

blank = {"provider": "manual", "total": None, "home_spread": None, "home_ml": None, "away_ml": None,
         "home_spread_odds": -110, "away_spread_odds": -110, "over_odds": -110, "under_odds": -110}
if choice == "✏️ Manual matchup":
    teams = sorted(M["idx"].keys())
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", teams, index=0)
    away_team = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    g = {"id": "manual", "home": home_team, "away": away_team, "home_id": None, "away_id": None, "state": "pre"}
    odds = dict(blank)
else:
    g = games[labels.index(choice)]
    odds = dict(g["odds"])
    if g["state"] != "pre":
        st.warning("This game has started or finished, so ESPN may no longer show pregame lines.")
gid = g["id"]

st.caption("The game selector above applies to the Game odds and First basket tabs. Full slate and Results cover the whole date.")
tab_game, tab_slate, tab_fb, tab_results = st.tabs(["📊 Game odds", "📋 Full slate", "🏀 First basket", "📈 Results"])

with tab_game:
    if None in (odds["home_ml"], odds["away_ml"], odds["home_spread"], odds["total"]):
        with st.expander("Some sportsbook lines are missing. Enter them manually", expanded=True):
            a, b, c = st.columns(3)
            odds["home_ml"] = a.number_input(f"{g['home']} moneyline", value=odds["home_ml"] or -110, step=5, key=f"hml{gid}")
            odds["away_ml"] = a.number_input(f"{g['away']} moneyline", value=odds["away_ml"] or -110, step=5, key=f"aml{gid}")
            odds["home_spread"] = b.number_input(f"{g['home']} spread", value=float(odds["home_spread"] or 0.0), step=0.5, key=f"sp{gid}")
            odds["total"] = c.number_input("Total", value=float(odds["total"] or 224.5), step=0.5, key=f"tot{gid}")

    st.subheader("🩹 Injuries")
    st.caption("Live from ESPN. Set the Role for key players (default is Rotation) so the model knows who actually matters.")
    roles_by_team, inj_pts = {}, {}
    cols = st.columns(2)
    for col, team in ((cols[0], g["home"]), (cols[1], g["away"])):
        with col:
            st.markdown(f"**{team}**" + ("  ·  back-to-back" if team in b2b_teams else ""))
            rows = injuries.get(team, [])
            if not rows:
                st.caption("No injuries reported." if live_ok else "Injury data unavailable.")
                inj_pts[team] = 0.0
                continue
            df = pd.DataFrame(rows).rename(columns={"player": "Player", "pos": "Pos", "status": "Status", "note": "Note"})
            df["Role"] = "Rotation"
            edited = st.data_editor(df, key=f"inj_{gid}_{team}", hide_index=True, disabled=["Player", "Pos", "Status", "Note"],
                                    column_config={"Role": st.column_config.SelectboxColumn(options=list(ROLE_PTS), required=True)})
            roles = dict(zip(edited["Player"], edited["Role"]))
            inj_pts[team] = injury_impact(team, injuries, roles)
            st.caption(f"Estimated impact: **-{inj_pts[team]:.1f} pts**")

    st.markdown("---")
    if st.button("🔥 Simulate game (10,000x)", type="primary", key=f"sim_{gid}"):
        if g["home"] == g["away"]:
            st.error("A team can't play itself.")
        else:
            st.session_state[f"res_{gid}"] = analyze_game(M, g, odds, inj_pts[g["home"]], inj_pts[g["away"]], b2b_teams)
    if f"res_{gid}" in st.session_state:
        render_result(st.session_state[f"res_{gid}"])

with tab_slate:
    slate_tab(M, date, games, injuries, b2b_teams)

with tab_fb:
    first_basket_tab(M, g, injuries, date)

with tab_results:
    results_tab()

with st.expander("How this works and how much to trust it"):
    st.markdown(
        "- **Ratings:** every team gets an offense and a defense rating fitted across all games (so beating good teams counts for more), "
        "with recent games weighted more and last season fading out.\n"
        "- **Sportsbook blend:** the model is only allowed to move the sportsbook's line part of the way, because the line is very hard to beat.\n"
        "- **Injuries:** ESPN doesn't rate players, so the Role you pick decides how much each absence matters.\n"
        "- **Accuracy:** the settings at the top of `app.py` should be updated from the Colab notebook's backtest. "
        "Past testing found that betting this kind of model's picks did not beat the sportsbook's closing lines.")
