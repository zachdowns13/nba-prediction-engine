# NBA_Model_v2
# Paste each section below into its own Colab cell, in order (or paste it all into one cell).

# ======================================================================
# CELL 1
# ======================================================================
#@title 1. Settings
SEASON = 2026          #@param {type:"integer"}
# Season to TEST on (ESPN names a season by the year it ends: 2026 = 2025-26).
# The season before it is used to tune the model, and the one before that as its starting history.

GAME_DATE = ""         #@param {type:"string"}
# For the live predictions at the end. Blank = today. Format 2026-10-21. (Colab uses UTC.)
TEAM_FILTER = ""       #@param {type:"string"}
SIMS = 10000           #@param {type:"integer"}
LEAN_MIN = 0.02        #@param {type:"number"}
LEAN_STRONG = 0.04     #@param {type:"number"}
FETCH_ODDS = True      #@param {type:"boolean"}
RUN_TUNING = True      #@param {type:"boolean"}
MAX_WORKERS = 8        #@param {type:"integer"}

# Who your stars are (for injuries in the live section). Roles: Star, Starter, Rotation, Deep bench.
# Example: ROLE_OVERRIDES = {"Jalen Brunson": "Star"}
ROLE_OVERRIDES = {}

# ---------------- First basket simulator ----------------
FB_SIMS = 10000            #@param {type:"integer"}
FB_HISTORY_GAMES = 10      #@param {type:"integer"}
# Each player's shooting is estimated from his team's last N games.
TIP_HOME_PROB = 0.50       #@param {type:"number"}
# Chance the HOME team wins the opening tip (nobody has a reliable edge here, so 50/50).
FB_TURNOVER_RATE = 0.13    #@param {type:"number"}
FB_OREB_RATE = 0.25        #@param {type:"number"}
RUN_FB_BACKTEST = True     #@param {type:"boolean"}
FB_BACKTEST_DAYS = 30      #@param {type:"integer"}

# Starting lineups. Leave empty to use projected starters (who started most of the last N games).
# Once lineups are announced (about 30 min before tip), enter them exactly, 5 names per team:
# STARTER_OVERRIDES = {"New York Knicks": ["Jalen Brunson", "Mikal Bridges", "OG Anunoby", "Josh Hart", "Karl-Anthony Towns"]}
STARTER_OVERRIDES = {}

# Your sportsbook's first-basket prices, to see if any are worth taking. Example:
# FIRST_BASKET_ODDS = {"Jalen Brunson": 650, "Karl-Anthony Towns": 800}
FIRST_BASKET_ODDS = {}

# ======================================================================
# CELL 2
# ======================================================================
#@title 2. Data helpers
import re, math, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests, numpy as np, pandas as pd
from IPython.display import display, Markdown

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NOW = dt.date.fromisoformat(GAME_DATE.strip()) if GAME_DATE.strip() else dt.date.today()
LIVE_SEASON = NOW.year + 1 if NOW.month >= 9 else NOW.year

def get_json(url, params=None, tries=3):
    last = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
    raise last

def to_american(x):
    if x is None: return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK"): return 100
    try: return int(float(s.replace("+", "")))
    except ValueError: return None

def spread_from_details(details, home_abbr):
    details = (details or "").strip()
    m = re.match(r"^([A-Z]{2,4})\s+([+-]?\d+(\.\d+)?)$", details)
    if details.upper() in ("EVEN", "PK"): return 0.0
    if m:
        val = float(m.group(2))
        return val if m.group(1) == home_abbr else -val
    return None

def fetch_day(d):
    # Completed regular-season games on one day.
    try:
        data = get_json(f"{BASE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 100})
    except Exception:
        return []
    out = []
    for ev in data.get("events", []):
        if not ev["status"]["type"].get("completed"): continue
        stype = (ev.get("season") or {}).get("type")
        if stype is not None and stype != 2: continue
        comp = ev["competitions"][0]
        h = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        a = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        try: hs, as_ = float(h["score"]), float(a["score"])
        except (KeyError, ValueError, TypeError): continue
        out.append({"date": d, "event_id": ev["id"], "home": h["team"]["displayName"], "away": a["team"]["displayName"],
                    "home_abbr": h["team"].get("abbreviation", ""), "away_abbr": a["team"].get("abbreviation", ""),
                    "hs": hs, "as": as_})
    return out

def fetch_standings(season):
    try: data = get_json(STANDINGS_URL, {"season": season})
    except Exception: return {}
    out = {}
    def walk(node):
        for e in (node.get("standings") or {}).get("entries", []):
            s = {x.get("name"): x.get("value") for x in e.get("stats", [])}
            gp = s.get("gamesPlayed") or ((s.get("wins") or 0) + (s.get("losses") or 0))
            pf = s.get("avgPointsFor") or ((s.get("pointsFor") or 0) / gp if gp else 0)
            pa = s.get("avgPointsAgainst") or ((s.get("pointsAgainst") or 0) / gp if gp else 0)
            if gp and pf and pa: out[e["team"]["displayName"]] = {"gp": gp, "pf": pf, "pa": pa}
        for c in node.get("children", []): walk(c)
    walk(data)
    return out

def fetch_odds_for_event(ev_id, home_abbr):
    url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{ev_id}/competitions/{ev_id}/odds"
    try: data = get_json(url, {"limit": 10}, tries=2)
    except Exception: return None
    for it in data.get("items", []):
        total, sp = it.get("overUnder"), spread_from_details(it.get("details"), home_abbr)
        if sp is not None and total: return {"home_spread": sp, "total": float(total)}
    return None

# ---- live-game helpers ----
def parse_odds(comp, home_abbr):
    o = (comp.get("odds") or [{}])[0]
    out = {"provider": (o.get("provider") or {}).get("name", "n/a"), "total": o.get("overUnder"),
           "home_spread": spread_from_details(o.get("details"), home_abbr), "home_ml": None, "away_ml": None,
           "home_spread_odds": -110, "away_spread_odds": -110, "over_odds": -110, "under_odds": -110}
    if out["home_spread"] is None and o.get("spread") is not None: out["home_spread"] = float(o["spread"])
    h, a = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    out["home_ml"], out["away_ml"] = to_american(h.get("moneyLine")), to_american(a.get("moneyLine"))
    ml = o.get("moneyline") or {}
    if out["home_ml"] is None: out["home_ml"] = to_american(((ml.get("home") or {}).get("close") or {}).get("odds"))
    if out["away_ml"] is None: out["away_ml"] = to_american(((ml.get("away") or {}).get("close") or {}).get("odds"))
    out["home_spread_odds"] = to_american(h.get("spreadOdds")) or -110
    out["away_spread_odds"] = to_american(a.get("spreadOdds")) or -110
    out["over_odds"] = to_american(o.get("overOdds")) or -110
    out["under_odds"] = to_american(o.get("underOdds")) or -110
    return out

def load_games(day):
    data = get_json(f"{BASE}/scoreboard", {"dates": day.strftime("%Y%m%d")})
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        games.append({"id": ev["id"], "home": home["team"]["displayName"], "away": away["team"]["displayName"],
                      "state": ev["status"]["type"]["state"], "odds": parse_odds(comp, home["team"].get("abbreviation", ""))})
    return games

def load_injuries():
    data = get_json(f"{BASE}/injuries")
    out = {}
    for t in data.get("injuries", []):
        out[t.get("displayName")] = [{"player": (i.get("athlete") or {}).get("displayName", "?"),
                                      "status": i.get("status", ""), "note": (i.get("shortComment") or "")[:120]}
                                     for i in t.get("injuries", [])]
    return out

# ---- math ----
def phi(x): return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))
def dec(o): return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)
def implied(o): return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)
def novig(a, b):
    ia, ib = implied(a), implied(b)
    return ia / (ia + ib), ib / (ia + ib)
def ev100(pw, pp, o): return 100 * (pw * (dec(o) - 1) - (1 - pw - pp))
def fair_american(p):
    p = min(max(p, 0.001), 0.999)
    return f"{-round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"
def short(t): return t.split()[-1]
BREAKEVEN = 110 / 210
print("Helpers loaded.")

# ======================================================================
# CELL 3
# ======================================================================
#@title 3. Download game history (cached after the first time)
SEASON_CACHE = {}

def get_season(s):
    if s in SEASON_CACHE: return SEASON_CACHE[s]
    start, end = dt.date(s - 1, 10, 1), min(dt.date(s, 4, 20), NOW - dt.timedelta(days=1))
    days = [start + dt.timedelta(days=i) for i in range(max((end - start).days + 1, 0))]
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        chunks = list(ex.map(fetch_day, days))
    games = [dict(g, season=s) for c in chunks for g in c]
    SEASON_CACHE[s] = games
    print(f"Season {s-1}-{str(s)[2:]}: {len(games)} games")
    return games

TUNE_SEASON, HIST_SEASON = SEASON - 1, SEASON - 2
for s in (HIST_SEASON, TUNE_SEASON, SEASON):
    get_season(s)
if min(len(SEASON_CACHE[s]) for s in (HIST_SEASON, TUNE_SEASON, SEASON)) < 800:
    print("⚠️ At least one season has far fewer games than the ~1,230 expected. Results will be unreliable.")

# ======================================================================
# CELL 4
# ======================================================================
#@title 4. The model: opponent-adjusted offense/defense ratings (weighted ridge regression)
def build_frame(seasons):
    rows = [g for s in seasons for g in get_season(s)]
    G = pd.DataFrame(rows).sort_values(["date", "event_id"]).reset_index(drop=True)
    G["ord"] = G["date"].map(lambda d: d.toordinal())
    last, cur, hb, ab = {}, None, [], []
    for r in G.itertuples():
        if r.season != cur: last, cur = {}, r.season          # rest days reset each season
        hb.append(1.0 if last.get(r.home) == r.ord - 1 else 0.0)
        ab.append(1.0 if last.get(r.away) == r.ord - 1 else 0.0)
        last[r.home] = r.ord; last[r.away] = r.ord
    G["hb"], G["ab"] = hb, ab
    teams = sorted(set(G["home"]) | set(G["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    T, N = len(teams), len(G)
    X = np.zeros((2 * N, 4 + 2 * T)); y = np.zeros(2 * N)
    r = np.arange(N); hi = G["home"].map(idx).values; ai = G["away"].map(idx).values
    X[:, 0] = 1
    X[2 * r, 1] = 1;  X[2 * r, 2] = G["hb"].values; X[2 * r, 3] = G["ab"].values
    X[2 * r, 4 + hi] = 1; X[2 * r, 4 + T + ai] = 1             # home team's offense vs away team's defense
    X[2 * r + 1, 2] = G["ab"].values; X[2 * r + 1, 3] = G["hb"].values
    X[2 * r + 1, 4 + ai] = 1; X[2 * r + 1, 4 + T + hi] = 1     # away team's offense vs home team's defense
    y[2 * r] = G["hs"].values; y[2 * r + 1] = G["as"].values
    return {"G": G, "X": X, "y": y, "teams": teams, "idx": idx, "T": T,
            "row_ord": np.repeat(G["ord"].values, 2), "row_season": np.repeat(G["season"].values, 2)}

def fit_ratings(F, mask, weights, ridge):
    X, y = F["X"][mask], F["y"][mask]
    pen = np.zeros(X.shape[1]); pen[4:] = ridge; pen[:4] = 1e-3
    A = X.T @ (weights[:, None] * X) + np.diag(pen)
    return np.linalg.solve(A, X.T @ (weights * y))

def weights_for(F, mask, day_ord, season, half_life, prev_w):
    age = day_ord - F["row_ord"][mask]
    return 0.5 ** (age / half_life) * prev_w ** (season - F["row_season"][mask])

def walk_forward(F, season, half_life, ridge, prev_w):
    # Predict every game of `season` using only games played before that day.
    G, X = F["G"], F["X"]
    ords, seasons = G["ord"].values, G["season"].values
    out_i, hp, ap = [], [], []
    for d in sorted(set(ords[seasons == season])):
        mask = F["row_ord"] < d
        if mask.sum() < 400: continue
        beta = fit_ratings(F, mask, weights_for(F, mask, d, season, half_life, prev_w), ridge)
        gi = np.where((ords == d) & (seasons == season))[0]
        out_i += list(gi); hp += list(X[2 * gi] @ beta); ap += list(X[2 * gi + 1] @ beta)
    P = G.loc[out_i].copy()
    P["home_pts"], P["away_pts"] = hp, ap
    P["margin_pred"] = P["home_pts"] - P["away_pts"]; P["total_pred"] = P["home_pts"] + P["away_pts"]
    P["margin"] = P["hs"] - P["as"]; P["total"] = P["hs"] + P["as"]; P["home_won"] = (P["margin"] > 0).astype(float)
    return P.reset_index(drop=True)

def run_v1(season, prior):
    # The previous simple model (season averages), kept for comparison.
    LGP = float(np.mean([v["pf"] for v in prior.values()])) if len(prior) >= 10 else 114.5
    state, lg_sum, lg_n, last, preds = {}, 0.0, 0, {}, []
    by_day = {}
    for g in get_season(season): by_day.setdefault(g["date"], []).append(g)
    def rating(t, lg):
        p = prior.get(t)
        po, pa = (lg + (p["pf"] - LGP) * 0.7, lg + (p["pa"] - LGP) * 0.7) if p else (lg, lg)
        gp, pf, pa_ = state.get(t, [0, 0.0, 0.0])
        if gp == 0: return po, pa
        w = gp / (gp + 15.0)
        return w * pf / gp + (1 - w) * po, w * pa_ / gp + (1 - w) * pa
    for d in sorted(by_day):
        wl = lg_n / (lg_n + 300.0); lg = wl * (lg_sum / lg_n if lg_n else LGP) + (1 - wl) * LGP
        for g in by_day[d]:
            ho, ha = rating(g["home"], lg); ao, aa = rating(g["away"], lg)
            hp, ap = ho + aa - lg + 1.0, ao + ha - lg - 1.0
            if last.get(g["home"]) == d - dt.timedelta(days=1): hp -= 0.75; ap += 0.75
            if last.get(g["away"]) == d - dt.timedelta(days=1): ap -= 0.75; hp += 0.75
            preds.append({"event_id": g["event_id"], "margin_pred": hp - ap, "total_pred": hp + ap})
        for g in by_day[d]:
            for t, f, a in ((g["home"], g["hs"], g["as"]), (g["away"], g["as"], g["hs"])):
                s = state.setdefault(t, [0, 0.0, 0.0]); s[0] += 1; s[1] += f; s[2] += a; last[t] = d
            lg_sum += g["hs"] + g["as"]; lg_n += 2
    return pd.DataFrame(preds)

def metrics(df, sd=13.0):
    p = phi(df["margin_pred"] / sd); hw = df["home_won"]; b = hw.mean()
    return {"Winner accuracy": (((p > 0.5) == (hw == 1)).mean(), max(b, 1 - b)),
            "Win-prob error (Brier, lower=better)": (((p - hw) ** 2).mean(), ((b - hw) ** 2).mean()),
            "Margin error (avg pts off)": ((df["margin"] - df["margin_pred"]).abs().mean(), (df["margin"] - df["margin"].mean()).abs().mean()),
            "Total error (avg pts off)": ((df["total"] - df["total_pred"]).abs().mean(), (df["total"] - df["total"].mean()).abs().mean())}
print("Model code loaded.")

# ======================================================================
# CELL 5
# ======================================================================
#@title 5. Tune on last season (grid search)
F_BT = build_frame([HIST_SEASON, TUNE_SEASON, SEASON])
GRID = [(hl, rg, pw) for hl in (45, 90, 180) for rg in (10, 30, 90) for pw in (0.4, 0.8)]
BEST = {"half_life": 90, "ridge": 30, "prev_w": 0.6}     # sensible defaults if tuning is switched off

if RUN_TUNING:
    print(f"Testing {len(GRID)} settings on the {TUNE_SEASON-1}-{str(TUNE_SEASON)[2:]} season (this is the only place tuning happens)...")
    res = []
    for hl, rg, pw in GRID:
        P = walk_forward(F_BT, TUNE_SEASON, hl, rg, pw)
        res.append({"half_life (days)": hl, "ridge": rg, "last-season weight": pw,
                    "margin error": (P["margin"] - P["margin_pred"]).abs().mean(),
                    "total error": (P["total"] - P["total_pred"]).abs().mean()})
    R = pd.DataFrame(res)
    R["score"] = R["margin error"] + R["total error"]
    R = R.sort_values("score").reset_index(drop=True)
    display(R.head(6).round(3).drop(columns="score"))
    b = R.iloc[0]
    BEST = {"half_life": float(b["half_life (days)"]), "ridge": float(b["ridge"]), "prev_w": float(b["last-season weight"])}
print("Chosen settings:", BEST)

# ======================================================================
# CELL 6
# ======================================================================
#@title 6. Grade it on a season the tuning never saw
def run_v2(season):
    return walk_forward(F_BT, season, BEST["half_life"], BEST["ridge"], BEST["prev_w"])

T2 = run_v2(SEASON)
prior = fetch_standings(SEASON - 1)
T1 = run_v1(SEASON, prior) if prior else None
if T1 is None:
    print("(Could not load last season's standings, so the old v1 model is skipped in the comparison.)")

def compare(sub2, sub1, label):
    m2 = metrics(sub2)
    cols = {"v2 (new)": {k: v[0] for k, v in m2.items()}, "dumb baseline": {k: v[1] for k, v in m2.items()}}
    if sub1 is not None:
        m1 = metrics(sub1)
        cols = {"v2 (new)": cols["v2 (new)"], "v1 (old)": {k: v[0] for k, v in m1.items()}, "dumb baseline": cols["dumb baseline"]}
    display(Markdown(f"### {label}  ({len(sub2)} games)"))
    display(pd.DataFrame(cols).round(3))

if T1 is not None:
    T1 = T1.merge(T2[["event_id", "margin", "total", "home_won"]], on="event_id")
    common = T2[T2["event_id"].isin(T1["event_id"])].reset_index(drop=True)
else:
    common = T2
compare(common, T1, "All games")
late = common["event_id"].isin(common.groupby("event_id").head(1)["event_id"]) & (common["date"] >= common["date"].min() + dt.timedelta(days=45))
compare(common[late].reset_index(drop=True), None if T1 is None else T1[T1["event_id"].isin(common[late]["event_id"])], "After the first 45 days of the season")

sd_m = float((T2["margin"] - T2["margin_pred"]).std()); sd_t = float((T2["total"] - T2["total_pred"]).std())
display(Markdown(f"**Spread of v2's misses:** margin SD = **{sd_m:.1f}**, total SD = **{sd_t:.1f}**."))

cal = T2.assign(p=phi(T2["margin_pred"] / sd_m)).assign(bucket=lambda d: pd.cut(d["p"], [0, .2, .3, .4, .5, .6, .7, .8, 1.0]))
cal = cal.groupby("bucket", observed=True).agg(games=("home_won", "size"), model_said=("p", "mean"), home_actually_won=("home_won", "mean")).round(3)
display(Markdown("### Calibration: when v2 says X%, does it happen X% of the time?"))
display(cal)

# ======================================================================
# CELL 7
# ======================================================================
#@title 7. Versus the sportsbook line: how much should the model count?
ODDS_CACHE = {}
def attach_lines(P):
    need = [(e, a) for e, a in zip(P["event_id"], P["home_abbr"]) if e not in ODDS_CACHE]
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for (e, a), f in zip(need, ex.map(lambda t: fetch_odds_for_event(*t), need)): ODDS_CACHE[e] = f
    Q = P.copy()
    Q["home_spread"] = [(ODDS_CACHE.get(e) or {}).get("home_spread", np.nan) for e in Q["event_id"]]
    Q["mkt_total"] = [(ODDS_CACHE.get(e) or {}).get("total", np.nan) for e in Q["event_id"]]
    Q = Q.dropna(subset=["home_spread", "mkt_total"]).copy()
    Q["mkt_margin"] = -Q["home_spread"]
    return Q

def fit_beta(L, actual, mkt, mod):
    x, yv = L[mod] - L[mkt], L[actual] - L[mkt]
    return float(np.clip(np.sum(x * yv) / np.sum(x * x), 0, 1))

def summarize(x):
    n = len(x); w = x["win"].sum(); p = x["push"].sum(); dcd = n - p
    wr = w / dcd if dcd else np.nan
    ci = 1.96 * math.sqrt(wr * (1 - wr) / dcd) if dcd else np.nan
    return pd.Series({"bets": n, "win %": round(wr * 100, 1), "± (95%)": round(ci * 100, 1),
                      "profit per $100 bet": round((w * (100 / 1.1) - (dcd - w) * 100) / n, 2),
                      "clearly beats breakeven?": "YES" if dcd and wr - ci > BREAKEVEN else "no"})

def quartile_table(diff, win, push, title):
    d = pd.DataFrame({"gap": diff.abs().values, "win": win.values, "push": push.values})
    d["grp"] = pd.qcut(d["gap"].rank(method="first"), 4, labels=["smallest disagreement", "2nd", "3rd", "biggest disagreement"])
    out = pd.DataFrame({str(k): summarize(g) for k, g in d.groupby("grp", observed=True)}).T
    out.loc["ALL"] = summarize(d)
    display(Markdown(f"### {title}")); display(out)
    return out

BETA_M, BETA_T = 0.3, 0.3          # fallback trust levels if lines aren't available
SD_M, SD_T = sd_m, sd_t
BT_SUMMARY = {}
if not FETCH_ODDS:
    print("FETCH_ODDS is off.")
else:
    print("Fetching sportsbook lines (can take a few minutes)...")
    LT = attach_lines(run_v2(TUNE_SEASON)); LB = attach_lines(T2)
    print(f"Lines found: {len(LT)} tuning-season games, {len(LB)} test-season games.")
    if len(LT) < 200 or len(LB) < 200:
        print("⚠️ ESPN didn't keep enough old lines for this test. Using default trust of 30% in the model.")
    else:
        BETA_M = fit_beta(LT, "margin", "mkt_margin", "margin_pred")
        BETA_T = fit_beta(LT, "total", "mkt_total", "total_pred")
        display(Markdown(f"**Learned from the tuning season:** the best forecast of the margin is the sportsbook line plus "
                         f"**{BETA_M:.0%}** of the way toward v2's number. For totals, **{BETA_T:.0%}**. "
                         f"(0% would mean the model adds nothing; 100% would mean ignore the line.)"))
        LB["margin_blend"] = LB["mkt_margin"] + BETA_M * (LB["margin_pred"] - LB["mkt_margin"])
        LB["total_blend"] = LB["mkt_total"] + BETA_T * (LB["total_pred"] - LB["mkt_total"])
        for lab, act, mk, md_, bl in (("Margin", "margin", "mkt_margin", "margin_pred", "margin_blend"),
                                      ("Total", "total", "mkt_total", "total_pred", "total_blend")):
            t = pd.Series({"Sportsbook line": (LB[act] - LB[mk]).abs().mean(), "v2 model alone": (LB[act] - LB[md_]).abs().mean(),
                           "Line + learned model tilt": (LB[act] - LB[bl]).abs().mean()}).round(2)
            display(Markdown(f"### {lab}: average points off on the TEST season (lower is better)")); display(t.to_frame("avg miss"))
        SD_M = float((LB["margin"] - LB["margin_blend"]).std()); SD_T = float((LB["total"] - LB["total_blend"]).std())

        cover = LB["margin"] + LB["home_spread"]
        s_tbl = quartile_table(LB["margin_pred"] - LB["mkt_margin"], pd.Series(np.where(LB["margin_pred"] - LB["mkt_margin"] > 0, cover > 0, cover < 0)),
                               pd.Series(cover == 0), "Betting v2's side of every spread (test season, grouped by how much v2 disagreed)")
        dt_ = LB["total"] - LB["mkt_total"]
        t_tbl = quartile_table(LB["total_pred"] - LB["mkt_total"], pd.Series(np.where(LB["total_pred"] - LB["mkt_total"] > 0, dt_ > 0, dt_ < 0)),
                               pd.Series(dt_ == 0), "Betting v2's side of every total (test season)")
        BT_SUMMARY = {"spread": s_tbl.loc["ALL", "win %"], "total": t_tbl.loc["ALL", "win %"]}
        display(Markdown(f"*Breakeven at -110 is {BREAKEVEN*100:.1f}%. Sportsbook lines from ESPN's old-odds feed may not be true closing lines, "
                         f"and this test has no injury data, so it is tougher on the model than real use.*"))

# ======================================================================
# CELL 8
# ======================================================================
#@title 8. Today's slate with v2
ROLE_PTS = {"star": 6.0, "starter": 3.0, "rotation": 1.2, "deep bench": 0.0}
STATUS_W = {"out": 1.0, "doubtful": 0.85, "questionable": 0.5, "day-to-day": 0.3, "probable": 0.1}
ROLES = {k.lower(): v for k, v in ROLE_OVERRIDES.items()}
rng = np.random.default_rng()

# Fit the final ratings on everything played before the game date
F_LIVE = build_frame([LIVE_SEASON - 2, LIVE_SEASON - 1, LIVE_SEASON])
d_now = NOW.toordinal()
mask = F_LIVE["row_ord"] < d_now
W = weights_for(F_LIVE, mask, d_now, LIVE_SEASON, BEST["half_life"], BEST["prev_w"])
BETA = fit_ratings(F_LIVE, mask, W, BEST["ridge"])
T_L, IDX_L = F_LIVE["T"], F_LIVE["idx"]
print(f"Ratings fitted on {int(mask.sum() / 2)} games (settings: {BEST}).")

def team_score(own, opp, is_home, own_b2b, opp_b2b):
    x = BETA[0] + BETA[1] * is_home + BETA[2] * own_b2b + BETA[3] * opp_b2b
    if own in IDX_L: x += BETA[4 + IDX_L[own]]
    if opp in IDX_L: x += BETA[4 + T_L + IDX_L[opp]]
    return x

def injury_impact(team, injuries):
    rows, total = [], 0.0
    for r in injuries.get(team, []):
        role = ROLES.get(r["player"].lower(), "rotation")
        pts = ROLE_PTS.get(role, 1.2) * STATUS_W.get(r["status"].lower(), 0.3); total += pts
        rows.append({"Player": r["player"], "Status": r["status"], "Role": role.title(), "Impact (pts)": round(pts, 1), "Note": r["note"]})
    return min(total, 15.0), rows

def analyze_game(g, injuries, b2b):
    home, away, o = g["home"], g["away"], g["odds"]
    hb, ab = 1.0 if home in b2b else 0.0, 1.0 if away in b2b else 0.0
    h_inj, h_rows = injury_impact(home, injuries); a_inj, a_rows = injury_impact(away, injuries)
    hp = team_score(home, away, 1, hb, ab); ap = team_score(away, home, 0, ab, hb)
    hp -= h_inj / 2; ap += h_inj / 2; ap -= a_inj / 2; hp += a_inj / 2       # injuries: hurt offense AND defense
    raw_m, raw_t = hp - ap, hp + ap
    hs_, ht_ = o["home_spread"] is not None, o["total"] is not None
    m_mu = (-o["home_spread"]) + BETA_M * (raw_m - (-o["home_spread"])) if hs_ else raw_m
    t_mu = o["total"] + BETA_T * (raw_t - o["total"]) if ht_ else raw_t
    margin = np.rint(rng.normal(m_mu, SD_M, SIMS)); total = np.rint(rng.normal(t_mu, SD_T, SIMS))
    ties = margin == 0; margin[ties] = rng.choice([-1, 1], ties.sum())
    rows = []
    def add(mk, side, line, bo, mp, push, bp):
        rows.append({"market": mk, "side": side, "line": line, "book_odds": bo, "model_p": mp, "book_p": bp, "edge": mp - bp, "ev": ev100(mp, push, bo)})
    ph = float(np.mean(margin > 0))
    if o["home_ml"] is not None and o["away_ml"] is not None:
        nv = novig(o["home_ml"], o["away_ml"]); add("Moneyline", home, "", o["home_ml"], ph, 0, nv[0]); add("Moneyline", away, "", o["away_ml"], 1 - ph, 0, nv[1])
    if hs_:
        L = o["home_spread"]; ch, ca = float(np.mean(margin + L > 0)), float(np.mean(margin + L < 0)); nv = novig(o["home_spread_odds"], o["away_spread_odds"])
        add("Spread", home, f"{L:+.1f}", o["home_spread_odds"], ch, 1 - ch - ca, nv[0]); add("Spread", away, f"{-L:+.1f}", o["away_spread_odds"], ca, 1 - ch - ca, nv[1])
    if ht_:
        T_ = o["total"]; ov, un = float(np.mean(total > T_)), float(np.mean(total < T_)); nv = novig(o["over_odds"], o["under_odds"])
        add("Total", "Over", f"{T_}", o["over_odds"], ov, 1 - ov - un, nv[0]); add("Total", "Under", f"{T_}", o["under_odds"], un, 1 - ov - un, nv[1])
    return {"game": g, "rows": rows, "raw_m": raw_m, "raw_t": raw_t, "med_m": float(np.median(margin)), "med_t": float(np.median(total)),
            "hp": (t_mu + m_mu) / 2, "ap": (t_mu - m_mu) / 2, "inj": {home: h_rows, away: a_rows}}

def pct(p): return f"{p * 100:.1f}%"
def by_txt(home, away, m): return f"{short(home)} by {m:.1f}" if m >= 0 else f"{short(away)} by {-m:.1f}"
def money(x): return f"\\${x:,.0f}"
def strength(e): return "🟢 Strong value" if e >= LEAN_STRONG else ("🟡 Slight value" if e >= LEAN_MIN else None)

def market_block(res, market):
    rows = [r for r in res["rows"] if r["market"] == market]
    if not rows: return f"**{market}**: no sportsbook line posted yet.\n"
    g = res["game"]; a, b = rows
    top = a if a["model_p"] >= b["model_p"] else b; best = a if a["edge"] >= b["edge"] else b
    if market == "Moneyline":
        fav = a if a["book_odds"] < b["book_odds"] else b
        model = f"**{short(top['side'])}** have **{pct(top['model_p'])}** odds to win (fair odds {fair_american(top['model_p'])})."
        book = f"**{short(fav['side'])}** are favored at **{fav['book_odds']:+d}**, which implies **{pct(fav['book_p'])}** to win."
    elif market == "Spread":
        m = res["med_m"]; who = f"{short(g['home'])} by {m:.1f}" if m > 0 else f"{short(g['away'])} by {abs(m):.1f}"
        model = f"**{short(top['side'])} {top['line']}** have **{pct(top['model_p'])}** odds to cover. Model projects {who}."
        book = f"{short(a['side'])} {a['line']} at {a['book_odds']:+d}, {short(b['side'])} {b['line']} at {b['book_odds']:+d}."
    else:
        model = f"**{top['side']} {top['line']}** has **{pct(top['model_p'])}** odds to hit. Model projects {res['med_t']:.0f} total points."
        book = f"The total is set at **{top['line']}**. Over at {a['book_odds']:+d}, Under at {b['book_odds']:+d}."
    name = f"{short(best['side'])} moneyline" if market == "Moneyline" else (f"{best['side']} {best['line']}" if market == "Total" else f"{short(best['side'])} {best['line']}")
    lab = strength(best["edge"])
    value = (f"{lab}: **{name}**. The sportsbook's odds imply {pct(best['book_p'])}; the model says {pct(best['model_p'])}. "
             f"If the model is right, a {money(100)} bet wins about **{money(best['ev'])}** on average over many bets.") if lab else "⚪ No value. The model and the sportsbook agree."
    return f"**{market}**\n\n🤖 **Model says:** {model}\n\n🏦 **Sportsbook says:** {book}\n\n💰 **Is there value?** {value}\n"

def show(res):
    g = res["game"]
    display(Markdown(f"## {g['away']} @ {g['home']}"))
    if g["state"] != "pre": display(Markdown("⚠️ This game has started or finished, so pregame lines may be gone."))
    display(Markdown(f"**Projected score:** {g['away']} {res['ap']:.0f}, {g['home']} {res['hp']:.0f}  \n"
                     f"Raw v2 rating-based projection (before blending with the line): {by_txt(g['home'], g['away'], res['raw_m'])}, {res['raw_t']:.0f} total.  \n"
                     f"Sportsbook: {g['odds']['provider']}"))
    for m in ("Moneyline", "Spread", "Total"): display(Markdown(market_block(res, m)))
    for team, rows in res["inj"].items():
        if rows:
            display(Markdown(f"**Injuries: {team}**" + ("" if ROLES else "  \n*Everyone is set to Rotation. Add your stars to ROLE_OVERRIDES in Settings.*")))
            display(pd.DataFrame(rows))

try:
    games = load_games(NOW); injuries = load_injuries()
except Exception as e:
    games, injuries = [], {}; print(f"Could not load today's data: {e}")
if TEAM_FILTER.strip():
    q = TEAM_FILTER.strip().lower(); games = [g for g in games if q in g["home"].lower() or q in g["away"].lower()]
try: b2b_teams = {x for g in load_games(NOW - dt.timedelta(days=1)) for x in (g["home"], g["away"])}
except Exception: b2b_teams = set()

if not games:
    print(f"No games found for {NOW}. The season may not have started; try another date.")
else:
    results = [analyze_game(g, injuries, b2b_teams) for g in games]
    note = ("Backtest of this same method on last season: betting its side won "
            f"{BT_SUMMARY['spread']}% of spreads and {BT_SUMMARY['total']}% of totals (breakeven is 52.4%). " if BT_SUMMARY else "")
    display(Markdown(f"# Slate for {NOW:%A, %B %d, %Y}"))
    display(Markdown(f"> **Reality check:** {note}Sportsbook lines are very hard to beat. Treat these as small tilts, not locks."))
    summ = []
    for r in results:
        gm = f"{short(r['game']['away'])} @ {short(r['game']['home'])}"
        if r["rows"]:
            b = max(r["rows"], key=lambda x: x["edge"]); nm = b["side"] if b["market"] == "Moneyline" else f"{b['side']} {b['line']}"
            summ.append({"Game": gm, "Best spot": f"{b['market']}: {nm}", "Model": pct(b["model_p"]), "Sportsbook": pct(b["book_p"]), "Value": strength(b["edge"]) or "⚪ No value"})
        else: summ.append({"Game": gm, "Best spot": "No lines posted", "Model": "", "Sportsbook": "", "Value": ""})
    display(pd.DataFrame(summ))
    for r in results: show(r)

# ======================================================================
# CELL 9
# ======================================================================
#@title 9. First basket simulator
# What it simulates: the opening tip, then possession after possession (turnovers, shots, rebounds)
# until someone makes a field goal. Each possession's shooter is picked by his shot rate, and he
# makes it at his own field-goal %. The first made field goal ends that simulated game.
# Free throws are NOT counted as the "first basket", and sportsbooks differ on this, so check your book's rules.
BOX_CACHE = {}
_warned = [False]

def _minutes(x):
    try: return float(str(x).split(":")[0])
    except ValueError: return 0.0

def fetch_box(event_id):
    # Box score (who played, who started, shots) plus the first made field goal of one game.
    if event_id in BOX_CACHE: return BOX_CACHE[event_id]
    try: data = get_json(f"{BASE}/summary", {"event": event_id}, tries=2)
    except Exception:
        BOX_CACHE[event_id] = None; return None
    players = {}
    for t in (data.get("boxscore") or {}).get("players", []):
        team = (t.get("team") or {}).get("displayName")
        for grp in t.get("statistics", []):
            names = grp.get("names") or grp.get("labels") or grp.get("keys") or []
            for a in grp.get("athletes", []):
                st = a.get("stats") or []
                if not st or a.get("didNotPlay"): continue
                row = dict(zip(names, st)); ath = a.get("athlete") or {}
                m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", str(row.get("FG", "")))
                players[str(ath.get("id"))] = {"name": ath.get("displayName", "?"), "team": team, "starter": bool(a.get("starter")),
                                               "min": _minutes(row.get("MIN", 0)),
                                               "fgm": float(m.group(1)) if m else 0.0, "fga": float(m.group(2)) if m else 0.0}
    first = None
    for p in data.get("plays", []):
        if p.get("scoringPlay") and (p.get("scoreValue") or 0) >= 2:
            parts = p.get("participants") or []
            aid = str(((parts[0].get("athlete") or {}).get("id"))) if parts else None
            first = {"athlete_id": aid, "text": p.get("text", "")}
            break
    if not players and not _warned[0]:
        _warned[0] = True
        print("⚠️ Could not read a box score from ESPN. If this keeps happening, ESPN changed its format. "
              f"Top-level keys received: {list(data.keys())[:12]}")
    BOX_CACHE[event_id] = {"players": players, "first": first}
    return BOX_CACHE[event_id]

def build_hist(G):
    h = {}
    for r in G.itertuples():
        h.setdefault(r.home, []).append((r.ord, r.event_id)); h.setdefault(r.away, []).append((r.ord, r.event_id))
    return h

def team_profile(hist, team, before_ord, n):
    evs = [e for (o, e) in hist.get(team, []) if o < before_ord][-n:]
    agg = {}
    for e in evs:
        box = fetch_box(e)
        if not box: continue
        for aid, p in box["players"].items():
            if p["team"] != team: continue
            a = agg.setdefault(aid, {"name": p["name"], "starts": 0, "min": 0.0, "fga": 0.0, "fgm": 0.0})
            a["starts"] += 1 if p["starter"] else 0; a["min"] += p["min"]; a["fga"] += p["fga"]; a["fgm"] += p["fgm"]
    return agg, len(evs)

def player_params(aid, name, a):
    # Shot rate per minute and FG%, nudged toward league average so small samples don't go wild.
    if not a: return {"id": aid, "name": name, "usage": 0.44, "fg": 0.47}
    return {"id": aid, "name": name, "usage": (a["fga"] + 0.44 * 60) / (a["min"] + 60), "fg": (a["fgm"] + 0.47 * 40) / (a["fga"] + 40)}

def pick_five(agg, exclude=(), override=None):
    if override:
        if len(override) != 5: raise ValueError("STARTER_OVERRIDES needs exactly 5 names per team.")
        by_name = {a["name"].lower(): (aid, a) for aid, a in agg.items()}
        out = []
        for nm in override:
            aid, a = by_name.get(nm.lower(), (nm, None))     # unknown player: league-average shooting
            out.append(player_params(aid, nm, a))
        return out
    ex = {x.lower() for x in exclude}
    ranked = sorted((kv for kv in agg.items() if kv[1]["name"].lower() not in ex), key=lambda kv: (-kv[1]["starts"], -kv[1]["min"]))[:5]
    return [player_params(aid, a["name"], a) for aid, a in ranked] if len(ranked) == 5 else None

def simulate_first_basket(home5, away5, n, tip_home, to_rate, oreb_rate, rng):
    cum, fg = np.zeros((2, 5)), np.zeros((2, 5))
    for t, ps in enumerate((home5, away5)):
        u = np.array([p["usage"] for p in ps]); cum[t] = np.cumsum(u / u.sum()); fg[t] = [p["fg"] for p in ps]
    cum[:, -1] = 1.0
    off = np.where(rng.random(n) < tip_home, 0, 1); first_off = off.copy()
    winner, scorer = np.full(n, -1), np.full(n, -1)
    active = np.ones(n, bool)
    for _ in range(100):
        if not active.any(): break
        idx = np.where(active)[0]; o = off[idx]
        u1, u2, u3, u4 = rng.random((4, idx.size))
        to = u1 < to_rate                                       # turnover: no shot
        sh = (u2[:, None] > cum[o]).sum(axis=1)                 # who shoots
        made = (~to) & (u3 < fg[o, sh])
        mi = idx[made]; winner[mi] = o[made]; scorer[mi] = sh[made]; active[mi] = False
        miss = (~made) & (~to)
        keep = miss & (u4 < oreb_rate)                          # offensive rebound: same team again
        sw = (~made) & (~keep)
        off[idx[sw]] = 1 - off[idx[sw]]
    return first_off, winner, scorer

def first_basket_table(home, away, home5, away5, n=FB_SIMS):
    fo, win, sc = simulate_first_basket(home5, away5, n, TIP_HOME_PROB, FB_TURNOVER_RATE, FB_OREB_RATE, rng)
    done = win >= 0
    rows = []
    for t, (team, five) in enumerate(((home, home5), (away, away5))):
        for i, p in enumerate(five):
            rows.append({"Player": p["name"], "Team": short(team), "pid": p["id"], "prob": float(np.mean((win == t) & (sc == i)))})
    T = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)
    return T, float(np.mean(win[done] == 0)), float(np.mean(win[done] == fo[done])), float(done.mean())
print("First basket simulator loaded.")

# ======================================================================
# CELL 10
# ======================================================================
#@title 10. First basket odds for today's games
FB_HIST = build_hist(F_LIVE["G"])
_cg = F_LIVE["G"][F_LIVE["G"]["season"] == LIVE_SEASON]
CUR_COUNTS = pd.concat([_cg["home"], _cg["away"]]).value_counts().to_dict()
OUT_STATUS = ("out", "doubtful")

def run_first_basket(g):
    home, away = g["home"], g["away"]
    fives, notes = [], []
    for team in (home, away):
        agg, n_games = team_profile(FB_HIST, team, NOW.toordinal(), FB_HISTORY_GAMES)
        excl = [r["player"] for r in injuries.get(team, []) if r["status"].lower() in OUT_STATUS]
        try: five = pick_five(agg, excl, STARTER_OVERRIDES.get(team))
        except ValueError as e:
            print(e); return
        if not five:
            print(f"Not enough box-score history to project {team}'s starters. Set STARTER_OVERRIDES for them."); return
        fives.append(five)
        if not STARTER_OVERRIDES.get(team):
            this_season = CUR_COUNTS.get(team, 0)
            notes.append(f"{short(team)}: projected from the last {n_games} games" + (f" (only {this_season} this season, so last season's roster is mixed in)" if this_season < FB_HISTORY_GAMES else "")
                         + (f"; out/doubtful players removed: {', '.join(excl)}" if excl else ""))
    T, p_home_first, p_tip_first, done = first_basket_table(home, away, fives[0], fives[1])
    display(Markdown(f"## 🏀 First basket: {g['away']} @ {g['home']}"))
    top = T.iloc[0]
    display(Markdown(f"Across {FB_SIMS:,} simulated openings, **{top['Player']}** is the most likely first scorer at **{pct(top['prob'])}** "
                     f"(fair odds {fair_american(top['prob'])}).  \n"
                     f"**{short(home)}** score the first basket **{pct(p_home_first)}** of the time, **{short(away)}** **{pct(1 - p_home_first)}**. "
                     f"The team that gets the opening tip scores first **{pct(p_tip_first)}** of the time."))
    out = []
    for r in T.itertuples():
        row = {"Player": r.Player, "Team": r.Team, "Chance to score first": pct(r.prob), "Fair odds": fair_american(r.prob)}
        if FIRST_BASKET_ODDS:
            o = next((v for k, v in FIRST_BASKET_ODDS.items() if k.lower() == r.Player.lower()), None)
            if o is not None:
                ev = ev100(r.prob, 0, o)
                row["Your odds"] = f"{o:+d}"; row["Book implies"] = pct(implied(o))
                row["Profit per $100 if model is right"] = f"${ev:,.0f}" if ev >= 0 else f"-${-ev:,.0f}"
                row["Worth it?"] = "🟢 Yes" if ev > 0 else "No"
        out.append(row)
    display(pd.DataFrame(out))
    display(Markdown("*" + " | ".join(notes) + "*" if notes else ""))
    if FIRST_BASKET_ODDS:
        display(Markdown("*First-basket prices carry a large built-in sportsbook fee, so a price must beat the fair odds by a wide margin "
                         "to be worth taking. 'Worth it?' assumes this simulator is accurate; see the backtest below for how accurate it has been.*"))

if not games:
    print("No games loaded, so nothing to simulate. Set GAME_DATE to a day with games.")
else:
    display(Markdown("> **Early in the season**, projected starters come partly from last season's games and rosters change. "
                     "Once lineups are announced, put them in STARTER_OVERRIDES for accurate results."))
    for g in games:
        run_first_basket(g)

# ======================================================================
# CELL 11
# ======================================================================
#@title 11. Backtest the first-basket simulator on real past games
if not RUN_FB_BACKTEST:
    print("RUN_FB_BACKTEST is off.")
else:
    G = F_BT["G"]; HB = build_hist(G)
    last = G.loc[G["season"] == SEASON, "ord"].max()
    window = G[(G["season"] == SEASON) & (G["ord"] > last - FB_BACKTEST_DAYS)].reset_index(drop=True)
    need = set(window["event_id"])
    for r in window.itertuples():
        for t in (r.home, r.away):
            need |= set([e for (o, e) in HB[t] if o < r.ord][-FB_HISTORY_GAMES:])
    todo = [e for e in need if e not in BOX_CACHE]
    print(f"Downloading {len(todo)} box scores ({len(window)} games being tested)... a minute or two.")
    with ThreadPoolExecutor(MAX_WORKERS) as ex: list(ex.map(fetch_box, todo))

    rows = []
    for r in window.itertuples():
        box = BOX_CACHE.get(r.event_id)
        if not box or not box["first"] or not box["first"]["athlete_id"]: continue
        fives = []
        for team in (r.home, r.away):
            agg, _ = team_profile(HB, team, r.ord, FB_HISTORY_GAMES)
            five = pick_five(agg)
            if not five: break
            fives.append(five)
        if len(fives) < 2: continue
        T, p_home_first, _, _ = first_basket_table(r.home, r.away, fives[0], fives[1], n=3000)
        aid = box["first"]["athlete_id"]
        actual_team = (box["players"].get(aid) or {}).get("team")
        hit = T.index[T["pid"] == aid]
        rows.append({"in_top10": len(hit) > 0, "p_actual": float(T.loc[hit[0], "prob"]) if len(hit) else 0.0,
                     "rank": int(hit[0]) + 1 if len(hit) else 11,
                     "p_home_first": p_home_first, "home_first": 1.0 if actual_team == r.home else (0.0 if actual_team == r.away else np.nan)})
    B = pd.DataFrame(rows)
    if len(B) < 30:
        print(f"Only {len(B)} usable games, which is too few to judge. ESPN's box score/play data may not match what the code expects.")
    else:
        inn = B[B["in_top10"]]
        tm = B.dropna(subset=["home_first"])
        out = pd.DataFrame({
            "First scorer was one of the 10 projected starters": [f"{B['in_top10'].mean():.0%}"],
            "Avg chance the model gave the actual scorer": [f"{inn['p_actual'].mean():.1%}"],
            "(random guess among 10 starters)": ["10.0%"],
            "Actual scorer was the model's #1 pick": [f"{(B['rank'] == 1).mean():.1%}"],
            "...was in the model's top 3": [f"{(B['rank'] <= 3).mean():.1%}"],
            "...was in the model's top 5": [f"{(B['rank'] <= 5).mean():.1%}"]}).T
        out.columns = [f"{len(B)} games"]
        display(Markdown(f"### How the first-basket simulator did on {len(B)} real games")); display(out)
        display(Markdown(f"**Team level:** the model said the home team scores first {tm['p_home_first'].mean():.1%} of the time; "
                         f"in reality it happened {tm['home_first'].mean():.1%} of the time ({len(tm)} games)."))
        display(Markdown("*If the model's average chance for the actual scorer is clearly above 10% and its #1 / top-3 / top-5 hit rates beat "
                         "10% / 30% / 50%, the simulator is finding real signal. A first basket is close to a coin flip among ten players, "
                         "so with only a few hundred games these numbers are noisy. Positive-looking results here still do not mean a price "
                         "is a good bet, because books charge a large fee on this market.*"))
