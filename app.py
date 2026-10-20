import streamlit as st
import numpy as np

st.set_page_config(page_title="NBA Prediction Engine", layout="wide")

st.title("🏀 NBA Matchup Prediction Engine")
st.subheader("10,000x Monte Carlo Simulation Dashboard")

# Complete, clean list of all 30 NBA teams for your dropdown menus
nba_teams = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", 
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", 
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", 
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", 
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", 
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", 
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", 
    "Utah Jazz", "Washington Wizards"
]

# Layout setup using a clean 2-column comparison structure
col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Home Team")
    home_team = st.selectbox("Select Home Team:", nba_teams, index=22) # Defaults to 76ers

with col2:
    st.header("✈️ Away Team")
    away_team = st.selectbox("Select Away Team:", nba_teams, index=1) # Defaults to Celtics

st.markdown("---")

# The simulation execution trigger button
if st.button("🔥 Run 10,000x Matchup Simulation", use_container_width=True):
    if home_team == away_team:
        st.error("⚠️ A team cannot play against itself! Please select two different franchises.")
    else:
        with st.spinner(f"Running 10,000 algorithmic simulations for {away_team} @ {home_team}..."):
            
            # Number of iterations
            simulations = 10000
            
            # Baseline parameters (average league scoring with variation standard deviation)
            base_home_mean = 114.0 + 3.0  # Assumes standard 3-point home court advantage
            base_away_mean = 114.0
            std_deviation = 11.5          # Standard NBA score variation buffer
            
            # Generate 10,000 randomized distributions instantly using NumPy
            home_simulated_scores = np.random.normal(base_home_mean, std_deviation, simulations)
            away_simulated_scores = np.random.normal(base_away_mean, std_deviation, simulations)
            
            # Track head-to-head wins within the 10k data points
            home_wins = np.sum(home_simulated_scores > away_simulated_scores)
            away_wins = simulations - home_wins
            
            # Calculate win probabilities
            home_win_pct = (home_wins / simulations) * 100
            away_win_pct = (away_wins / simulations) * 100
            
            # Calculate average scores
            avg_home_score = round(np.mean(home_simulated_scores))
            avg_away_score = round(np.mean(away_simulated_scores))
            
            # Determine overall project path values
            if home_wins > away_wins:
                winner_text = f"{home_team}"
                probability_text = f"{home_win_pct:.1f}% Probability"
            else:
                winner_text = f"{away_team}"
                probability_text = f"{away_win_pct:.1f}% Probability"
                
            st.success("📊 10,000 Simulations Compiled Successfully!")
            
            # Display primary summary metrics row
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="🏆 Projected Winner", value=winner_text)
            with res_col2:
                st.metric(label="🎯 Win Likelihood", value=probability_text)
            with res_col3:
                st.metric(label="🏀 Predicted Total", value=f"{avg_home_score + avg_away_score} Total Points")
                
            st.markdown("### 📋 Average Score Projections")
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric(label=f"🏠 {home_team} Score", value=f"{avg_home_score} pts", delta=f"{home_win_pct:.1f}% Win Chance")
            with score_col2:
                st.metric(label=f"✈️ {away_team} Score", value=f"{avg_away_score} pts", delta=f"{away_win_pct:.1f}% Win Chance")
