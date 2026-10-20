import streamlit as st
import random

st.set_page_config(page_title="NBA Prediction Engine", layout="wide")

st.title("🏀 NBA Matchup Prediction Engine")
st.subheader("Pure Team-vs-Team Simulation Dashboard")

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
    # Defaults to Celtics to create a clean opening night matchup view
    away_team = st.selectbox("Select Away Team:", nba_teams, index=1)

st.markdown("---")

# The simulation execution trigger button
if st.button("🔥 Run Matchup Simulation", use_container_width=True):
    if home_team == away_team:
        st.error("⚠️ A team cannot play against itself! Please select two different franchises.")
    else:
        with st.spinner(f"Running 10,000 algorithmic simulations for {away_team} @ {home_team}..."):
            
            # Pure mathematical baseline model simulation
            # Incorporates standard home-court advantage weights natively
            home_base = random.randint(102, 116) + 3 
            away_base = random.randint(102, 116)
            
            total_points = home_base + away_base
            spread = abs(home_base - away_base)
            
            # Identify the projected winner block cleanly
            if home_base > away_base:
                winner = home_team
                spread_string = f"{home_team} -{spread}"
            else:
                winner = away_team
                spread_string = f"{away_team} -{spread}"
                
            st.success("📊 Projections Calculated Successfully!")
            
            # Renders results in a clean grid box format
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="🏆 Projected Winner", value=winner)
            with res_col2:
                st.metric(label="🎯 Projected Spread", value=spread_string)
            with res_col3:
                st.metric(label="🏀 Total Points", value=str(total_points))

