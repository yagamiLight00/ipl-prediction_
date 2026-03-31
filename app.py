import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
df    = pickle.load(open('df.pkl',    'rb'))
col   = pickle.load(open('columns.pkl', 'rb'))

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

st.set_page_config(page_title="IPL Predictor", page_icon="🏏")
st.title("🏏 IPL Match Predictor")

c1, c2 = st.columns(2)
with c1:
    team1 = st.selectbox("Team 1", TEAMS)
with c2:
    team2 = st.selectbox("Team 2", [t for t in TEAMS if t != team1])

city = st.selectbox("Match City", CITIES)

c3, c4 = st.columns(2)
with c3:
    toss_winner   = st.selectbox("Toss Winner",   [team1, team2])
with c4:
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

if st.button("Predict Winner"):
    prev = df

    def win_rate(team):
        m = (prev['team1']==team)|(prev['team2']==team)
        total = m.sum()
        wins  = (prev.loc[m,'winner']==team).sum()
        return wins/total if total>0 else 0.5

    def form(team):
        m = prev[(prev['team1']==team)|(prev['team2']==team)].tail(8)
        return (m['winner']==team).mean() if len(m)>0 else 0.5

    t1_wr = win_rate(team1);  t2_wr = win_rate(team2)
    t1_f  = form(team1);      t2_f  = form(team2)

    h_rows = prev[
        ((prev['team1']==team1)&(prev['team2']==team2))|
        ((prev['team1']==team2)&(prev['team2']==team1))
    ]
    h2h_val = (h_rows['winner']==team1).mean() if len(h_rows)>0 else 0.5

    v_rows = prev[
        ((prev['team1']==team1)|(prev['team2']==team1)) & (prev['city']==city)
    ]
    venue_val = (v_rows['winner']==team1).mean() if len(v_rows)>0 else 0.5

    toss_eff_val = 1 if toss_winner==team1 else 0
    td_val       = 1 if toss_decision=='bat' else 0

    # ✅ EXACT same feature names and logic as train.py
    input_data = pd.DataFrame([{
        'toss_decision':  td_val,
        'wr_diff':        t1_wr - t2_wr,
        'form_diff':      t1_f  - t2_f,
        'momentum':       t1_f  - t2_f,
        'h2h':            h2h_val,
        'toss_eff':       toss_eff_val,
        'venue_diff':     venue_val - 0.5,
        'toss_advantage': toss_eff_val * td_val,
    }])

    input_data = input_data.reindex(columns=col, fill_value=0)

    probs = model.predict_proba(input_data)[0]
    probs = np.clip(probs, 0.05, 0.95)
    probs = probs / probs.sum()

    t1_prob = round(float(probs[1]) * 100, 1)
    t2_prob = round(float(probs[0]) * 100, 1)

    winner      = team1 if t1_prob > t2_prob else team2
    winner_prob = t1_prob if winner==team1 else t2_prob

    st.success(f"🏆 Predicted Winner: **{winner}** ({winner_prob}%)")
    st.write(f"**{team1}:** {t1_prob}%")
    st.progress(float(t1_prob / 100))
    st.write(f"**{team2}:** {t2_prob}%")
    st.progress(float(t2_prob / 100))

    with st.expander("📊 Features used for this prediction"):
        st.dataframe(input_data)