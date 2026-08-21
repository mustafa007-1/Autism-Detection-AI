import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Autism Detection Spectrum", page_icon="🧠")

st.title("🧠 Autism Detection Spectrum")
st.write("Answer the questions below to assess autism traits.")

# --- LOAD MODEL ---
if not os.path.exists('asd_model.pkl'):
    st.error("Model files not found! Please run 'train_model.py' first.")
    st.stop()

model = joblib.load('asd_model.pkl')
ohe = joblib.load('onehot_encoder.pkl')

# --- USER INPUTS ---
with st.form("my_form"):
    st.subheader("Questionnaire (AQ-10)")
    
    # The 10 Questions
    q_text = [
        "1. I often notice small sounds when others do not",
        "2. I usually concentrate more on the whole picture, not the details",
        "3. I find it easy to do more than one thing at once",
        "4. I find it easy to read between the lines when someone is talking",
        "5. I know how to tell if someone listening to me is getting bored",
        "6. I find it easy to work out what someone is thinking or feeling just by looking",
        "7. I like to collect detailed information about things",
        "8. I find it difficult to work out people's intentions",
        "9. New situations make me anxious",
        "10. I prefer to do things the same way over and over again"
    ]
    
    # Create Radio Buttons for Questions
    answers = []
    for q in q_text:
        ans = st.radio(q, ["Disagree", "Agree"], horizontal=True)
        answers.append(1 if ans == "Agree" else 0)

    st.subheader("Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 12, 100, 25)
        gender = st.selectbox("Gender", ["Female", "Male"])
        ethnicity = st.selectbox("Ethnicity", [
            'White-European', 'Asian', 'Middle Eastern ', 'Black', 'Latino',
            'Hispanic', 'South Asian', 'Others', 'Pasifika', 'Turkish'
        ])
    with col2:
        jaundice = st.selectbox("Born with jaundice?", ["No", "Yes"])
        family = st.selectbox("Family member with ASD?", ["No", "Yes"])
        app_before = st.selectbox("Used app before?", ["No", "Yes"])

    submitted = st.form_submit_button("Get Result")

# --- PREDICTION ---
if submitted:
    # 1. Gather Data
    data = {
        'a1_score': answers[0], 'a2_score': answers[1], 'a3_score': answers[2],
        'a4_score': answers[3], 'a5_score': answers[4], 'a6_score': answers[5],
        'a7_score': answers[6], 'a8_score': answers[7], 'a9_score': answers[8],
        'a10_score': answers[9],
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'jaundice': 1 if jaundice == "Yes" else 0,
        'autism': 1 if family == "Yes" else 0,
        'used_app_before': 1 if app_before == "Yes" else 0,
        'ethnicity': ethnicity
    }
    
    input_df = pd.DataFrame([data])

    # 2. Process Ethnicity (Same as training)
    eth_encoded = ohe.transform(input_df[['ethnicity']])
    eth_df = pd.DataFrame(eth_encoded, columns=ohe.get_feature_names_out())
    
    # Combine and Clean
    input_df = pd.concat([input_df.drop('ethnicity', axis=1), eth_df], axis=1)
    
    # Align columns with model
    # (This fixes the 'feature names' error)
    expected_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    # 3. Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # 4. Show Result
    st.divider()
    if prediction == 1:
        st.error(f"**Result:** High likelihood of ASD traits ({prob:.0%} confidence)")
    else:
        st.success(f"**Result:** Low likelihood of ASD traits ({prob:.0%} confidence)")