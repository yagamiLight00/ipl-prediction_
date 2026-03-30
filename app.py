import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load files 
# -------------------------------
model  = pickle.load(open('model.pkl', 'rb'))
df     = pickle.load(open('df.pkl',    'rb'))
col    = pickle.load(open('columns.pkl', 'rb'))

# -------------------------------
# Constants
# -------------------------------
TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

CITIES = [
    'Chennai', 'New Delhi', 'Ahmedabad', 'Kolkata', 'Lucknow',
    'Mumbai', 'Mohali', 'Jaipur', 'Bengaluru', 'Hyderabad'
]

TEAM_CITY = {
    'Chennai Super Kings':        'Chennai',
    'Delhi Capitals':             'New Delhi',
    'Gujarat Titans':             'Ahmedabad',
    'Kolkata Knight Riders':      'Kolkata',
    'Lucknow Super Giants':       'Lucknow',
    'Mumbai Indians':             'Mumbai',
    'Punjab Kings':               'Mohali',
    'Rajasthan Royals':           'Jaipur',
    'Royal Challengers Bengaluru':'Bengaluru',
    'Sunrisers Hyderabad':        'Hyderabad'
}

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="IPL Predictor", page_icon="🏏")
st.title("🏏 IPL Match Predictor")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", TEAMS)
with col2:
    team2 = st.selectbox("Team 2", [t for t in TEAMS if t != team1])

city = st.selectbox("Match City", CITIES)

st.subheader("Toss")
col3, col4 = st.columns(2)
with col3:
    toss_winner  = st.selectbox("Toss Winner",   [team1, team2])
with col4:
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Winner"):

    prev = df  # use full historical data

    # --- Win rate ---
    rows1 = prev[(prev['team1'] == team1) | (prev['team2'] == team1)]
    rows2 = prev[(prev['team1'] == team2) | (prev['team2'] == team2)]

    t1_wr = (rows1['winner'] == team1).sum() / len(rows1) if len(rows1) > 0 else 0.5
    t2_wr = (rows2['winner'] == team2).sum() / len(rows2) if len(rows2) > 0 else 0.5

    # --- Form (last 5) ---
    t1_last5 = rows1.tail(5)
    t2_last5 = rows2.tail(5)

    t1_form = (t1_last5['winner'] == team1).sum() / 5 if len(t1_last5) == 5 else 0.5
    t2_form = (t2_last5['winner'] == team2).sum() / 5 if len(t2_last5) == 5 else 0.5

    # --- Momentum ---
    momentum = t1_form - t2_form

    # --- Head to head ---
    h2h_rows = prev[
        ((prev['team1'] == team1) & (prev['team2'] == team2)) |
        ((prev['team1'] == team2) & (prev['team2'] == team1))
    ]
    h2h_val = (h2h_rows['winner'] == team1).sum() / len(h2h_rows) if len(h2h_rows) > 0 else 0.5

    # --- Toss effect ---
    toss_effect_val = 1 if toss_winner == team1 else 0

    # --- Venue win rate ---
    venue_rows = prev[
        ((prev['team1'] == team1) | (prev['team2'] == team1)) &
        (prev['city'] == city)
    ]
    venue_wr = (venue_rows['winner'] == team1).sum() / len(venue_rows) if len(venue_rows) > 0 else 0.5

    # --- Build feature row matching EXACTLY what the model was trained on ---
    input_data = pd.DataFrame([[0] * len(col)], columns=col)

    input_data['id']              = df['id'].max() + 1
    input_data['toss_decision']   = 1 if toss_decision == 'bat' else 0
    input_data['team1_win_rate']  = t1_wr
    input_data['team2_win_rate']  = t2_wr
    input_data['team1_form']      = t1_form
    input_data['team2_form']      = t2_form
    input_data['momentum']        = momentum
    input_data['head_to_head']    = h2h_val
    input_data['toss_effect']     = toss_effect_val
    input_data['venue_win_rate']  = venue_wr

    # Reindex to match training columns exactly
    input_data = input_data.reindex(columns=col, fill_value=0)

    # --- Predict  
    probs = model.predict_proba(input_data)[0]

    # Clip to avoid 100% / 0% outputs
    probs = np.clip(probs, 0.05, 0.95)
    probs = probs / probs.sum()

    # 
    t1_prob = round(float(probs[1]) * 100, 1)
    t2_prob = round(float(probs[0]) * 100, 1)

    winner = team1 if t1_prob > t2_prob else team2
    winner_prob = t1_prob if winner == team1 else t2_prob

    # --- Display ---
    st.success(f"🏆 Predicted Winner: **{winner}** ({winner_prob}%)")
    st.write(f"**{team1}:** {t1_prob}%")
    #  
    st.progress(float(t1_prob / 100))
    st.write(f"**{team2}:** {t2_prob}%")
    st.progress(float(t2_prob / 100))

    with st.expander("📊 Features used for this prediction"):
        st.dataframe(input_data)