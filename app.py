import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="NBA Advanced Analytics Engine", layout="wide")

st.title("🏀 NBA Advanced Matchup Prediction Engine")
st.subheader("10,000x Monte Carlo Simulation & Sportsbook Analytics Dashboard")

# =====================================================================
# SYSTEM RESILIENCY LAYER: INTERNALLY FAULT-TOLERANT DATA STORAGE
# =====================================================================
# If your CSV files fail to load or are completely missing, this block
# silently catches the error and keeps your app online without crashing the site.
try:
    # Attempting standard data cohort extraction
    player_db = pd.read_csv('nba_players.csv')
    nba_teams = sorted(list(player_db['TEAM_NAME'].unique()))
    st.sidebar.success("📊 Roster database connected.")
except Exception:
    # Safe internal fallback parameters if the file tracking system is missing from GitHub
    st.sidebar.warning("⚠️ Roster database offline. Utilizing analytical strength profiles.")
    
# Clean analytical baseline rating templates (Offensive/Defensive power relative to league average)
team_analytics_profiles = {
    "Atlanta Hawks":          {"off_rtg": 1.5,  "def_rtg": 1.5},
    "Boston Celtics":         {"off_rtg": 4.5,  "def_rtg": -3.5},
    "Brooklyn Nets":          {"off_rtg": -4.0, "def_rtg": 2.5},
    "Charlotte Hornets":      {"off_rtg": -1.0, "def_rtg": 0.5},
    "Chicago Bulls":          {"off_rtg": -1.5, "def_rtg": 2.0},
    "Cleveland Cavaliers":    {"off_rtg": 2.5,  "def_rtg": -1.5},
    "Dallas Mavericks":       {"off_rtg": 1.0,  "def_rtg": 1.0},
    "Denver Nuggets":         {"off_rtg": 4.0,  "def_rtg": 0.5},
    "Detroit Pistons":        {"off_rtg": 3.5,  "def_rtg": -2.0},
    "Golden State Warriors":  {"off_rtg": 0.5,  "def_rtg": 0.5},
    "Houston Rockets":        {"off_rtg": 1.0,  "def_rtg": -1.5},
    "Indiana Pacers":         {"off_rtg": 1.5,  "def_rtg": 3.5},
    "Los Angeles Clippers":   {"off_rtg": -0.5, "def_rtg": -0.5},
    "Los Angeles Lakers":     {"off_rtg": 2.0,  "def_rtg": 1.0},
    "Memphis Grizzlies":      {"off_rtg": -1.5, "def_rtg": 1.5},
    "Miami Heat":             {"off_rtg": 0.5,  "def_rtg": -1.0},
    "Milwaukee Bucks":        {"off_rtg": 0.5,  "def_rtg": 2.5},
    "Minnesota Timberwolves": {"off_rtg": 2.5,  "def_rtg": -0.5},
    "New Orleans Pelicans":   {"off_rtg": -0.5, "def_rtg": 2.0},
    "New York Knicks":        {"off_rtg": 3.0,  "def_rtg": -1.5},
    "Oklahoma City Thunder":  {"off_rtg": 5.0,  "def_rtg": -4.5},
    "Orlando Magic":          {"off_rtg": 0.0,  "def_rtg": -2.0},
    "Philadelphia 76ers":     {"off_rtg": 1.5,  "def_rtg": -0.5},
    "Phoenix Suns":           {"off_rtg": 1.0,  "def_rtg": -0.5},
    "Portland Trail Blazers": {"off_rtg": -1.0, "def_rtg": 0.5},
    "Sacramento Kings":       {"off_rtg": 0.5,  "def_rtg": 2.5},
    "San Antonio Spurs":      {"off_rtg": 3.0,  "def_rtg": -2.5},
    "Toronto Raptors":        {"off_rtg": 0.0,  "def_rtg": 1.0},
    "Utah Jazz":              {"off_rtg": -1.0, "def_rtg": 3.5},
    "Washington Wizards":     {"off_rtg": -3.5, "def_rtg": 4.5}
}
nba_teams = sorted(list(team_analytics_profiles.keys()))

# =====================================================================
# INTERACTIVE SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("🎯 Game Control Settings")
game_pace = st.sidebar.slider("Projected Possessions (Pace):", min_value=90.0, max_value=105.0, value=98.5, step=0.5)
pace_scalar = game_pace / 98.5

st.sidebar.subheader("🎰 Sportsbook Vegas Lines")
vegas_spread = st.sidebar.number_input("Market Point Spread:", value=0.0, step=0.5)
vegas_total = st.sidebar.number_input("Market Over/Under Line:", value=224.5, step=0.5)

# =====================================================================
# MATCHUP SELECTOR INTERFACE
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Home Team")
    home_team = st.selectbox("Select Home Team:", nba_teams, index=22) # Defaults to 76ers
    home_b2b = st.checkbox("Home Team is on a Back-to-Back", key="h_b2b")
    home_injury = st.checkbox("Home Team has Key Injuries", key="h_inj")

with col2:
    st.header("✈️ Away Team")
    away_team = st.selectbox("Select Away Team:", nba_teams, index=1) # Defaults to Celtics
    away_b2b = st.checkbox("Away Team is on a Back-to-Back", key="a_b2b")
    away_injury = st.checkbox("Away Team has Key Injuries", key="a_inj")

st.markdown("---")

# =====================================================================
# ADVANCED 10,000x NUMPY SIMULATION ENGINE
# =====================================================================
if st.button("🔥 Run 10,000x Advanced Analytics Simulation", use_container_width=True):
    if home_team == away_team:
        st.error("⚠️ System Check Error: A franchise cannot run simulations against itself.")
    else:
        with st.spinner(f"Compiling 10,000 distinct data models..."):
            simulations = 10000
            league_baseline = 114.0
            
            home_data = team_analytics_profiles[home_team]
            away_data = team_analytics_profiles[away_team]
            
            # Core scoring probability formula math
            home_mean = (league_baseline + home_data["off_rtg"] + away_data["def_rtg"] + 3.0) * pace_scalar
            away_mean = (league_baseline + away_data["off_rtg"] + home_data["def_rtg"]) * pace_scalar
            
            # Apply situational handicap modifiers
            if home_b2b:    home_mean -= 1.5
            if home_injury: home_mean -= 3.0
            if away_b2b:    away_mean -= 1.5
            if away_injury: away_mean -= 3.0
            
            # Run massive vector distributions instantaneously using NumPy
            std_dev = 11.2
            home_sim_scores = np.random.normal(home_mean, std_dev, simulations)
            away_sim_scores = np.random.normal(away_mean, std_dev, simulations)
            
            # Aggregate simulation loop results
            home_wins = np.sum(home_sim_scores > away_sim_scores)
            home_win_pct = (home_wins / simulations) * 100
            away_win_pct = 100.0 - home_win_pct
            
            avg_home_projected = int(round(np.mean(home_sim_scores)))
            avg_away_projected = int(round(np.mean(away_sim_scores)))
            predicted_total = avg_home_projected + avg_away_projected
            
            # Render dashboard metrics columns safely
            st.success("📊 10,000 Generative Scenarios Compiled Successfully!")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                winner = home_team if home_wins > (simulations/2) else away_team
                st.metric(label="🏆 Simulated Winner", value=winner)
            with res_col2:
                prob = max(home_win_pct, away_win_pct)
                st.metric(label="🎯 Algorithmic Confidence", value=f"{prob:.1f}%")
            with res_col3:
                st.metric(label="🏀 Predicted Total Score", value=f"{predicted_total} PTS")
                
            st.markdown("### 📋 Expected Box Score Breakdown")
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric(label=f"🏠 {home_team} Average Score", value=f"{avg_home_projected} PTS", delta=f"{home_win_pct:.1f}% Win Chance")
            with score_col2:
                st.metric(label=f"✈️ {away_team} Average Score", value=f"{avg_away_projected} PTS", delta=f"{away_win_pct:.1f}% Win Chance")
                
            # Sportsbook Vegas Valuation Analysis Engine
            st.markdown("### 🎰 Sportsbook Betting Line EV Outcomes")
            bet_col1, bet_col2 = st.columns(2)
            
            with bet_col1:
                home_cover_count = np.sum((home_sim_scores - away_sim_scores) + vegas_spread > 0)
                home_cover_pct = (home_cover_count / simulations) * 100
                st.markdown(f"**Point Spread Market Analysis (Line: {vegas_spread})**")
                st.progress(int(home_cover_pct))
                st.write(f"🏠 {home_team} Covers Spread Chance: **{home_cover_pct:.1f}%**")
                st.write(f"✈️ {away_team} Covers Spread Chance: **{(100.0 - home_cover_pct):.1f}%**")
                
            with bet_col2:
                simulated_totals = home_sim_scores + away_sim_scores
                over_count = np.sum(simulated_totals > vegas_total)
                over_pct = (over_count / simulations) * 100
                st.markdown(f"**Total Points Market Analysis (Line: {vegas_total})**")
                st.progress(int(over_pct))
                st.write(f"🔥 **OVER** {vegas_total} Probability: **{over_pct:.1f}%**")
                st.write(f"🧊 **UNDER** {vegas_total} Probability: **{(100.0 - over_pct):.1f}%**")
