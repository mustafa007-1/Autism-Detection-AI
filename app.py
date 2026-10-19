"""Streamlit screening app.  Run:  streamlit run app.py"""
import joblib
import sklearn
import streamlit as st

from asd import aq10
from asd.config import MODEL_PATH
from asd.data import build_features

st.set_page_config(page_title="Autism Trait Screening", page_icon="🧠")


@st.cache_resource
def load_bundle(path: str) -> dict:
    return joblib.load(path)


st.title("🧠 Autism Trait Screening (AQ-10)")
st.caption("A screening aid only — it cannot diagnose autism. If you have concerns, "
           "talk to a doctor or a qualified specialist.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
model = bundle["model"]
metrics = bundle["metrics"]
if metrics["sklearn_version"] != sklearn.__version__:
    st.warning(f"The model was trained with scikit-learn {metrics['sklearn_version']} but "
               f"{sklearn.__version__} is installed. Re-run `python train_model.py`.")

age = st.number_input("Age of the person being screened", min_value=aq10.ADOLESCENT_MIN_AGE,
                      max_value=100, value=25, step=1)
group = aq10.age_group(age)
if group == aq10.ADOLESCENT:
    st.info("Ages 12–16 use the **AQ-10 Adolescent** questionnaire, completed by a parent or carer "
            "about the young person.")
else:
    st.info("Ages 17+ use the **AQ-10 Adult** questionnaire. Answer about the person being screened.")

with st.form("screening"):
    st.subheader("Questionnaire")
    responses = [
        # The key includes the group so answers reset when the questionnaire changes.
        st.radio(f"{i}. {item.text}", aq10.RESPONSES, index=None, horizontal=True, key=f"{group}_q{i}")
        for i, item in enumerate(aq10.QUESTIONNAIRES[group], start=1)
    ]

    st.subheader("Background")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=None, key="gender")
        ethnicity = st.selectbox("Ethnicity", model.known_ethnicities(), index=None, key="ethnicity")
    with col2:
        jaundice = st.radio("Born with jaundice?", ["No", "Yes"], horizontal=True, key="jaundice")
        family = st.radio("Immediate family member with autism?", ["No", "Yes"], horizontal=True,
                          key="family")
        used_before = st.radio("Used a screening app before?", ["No", "Yes"], horizontal=True,
                               key="used_before")

    submitted = st.form_submit_button("Get result", type="primary")

missing = [f"question {i}" for i, r in enumerate(responses, start=1) if r is None]
missing += [name for name, value in (("gender", gender), ("ethnicity", ethnicity)) if value is None]

if submitted and missing:
    st.error(f"Please answer everything before continuing. Missing: {', '.join(missing)}.")
elif submitted:
    scores = aq10.score_answers(group, responses)
    total = sum(scores)
    X = build_features(scores, age, male=gender == "Male", jaundice=jaundice == "Yes",
                       family_asd=family == "Yes", used_app_before=used_before == "Yes",
                       ethnicity=ethnicity)
    prob = float(model.predict_proba(X)[0, 1])
    model_positive = prob >= 0.5
    rule_positive = total > aq10.REFERRAL_CUTOFF

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("AQ-10 score", f"{total} / 10")
    c2.metric("Model probability of ASD traits", f"{prob:.0%}")

    if model_positive:
        st.error("**Higher likelihood of autism traits.** Consider asking a doctor about a specialist "
                 "autism assessment.")
    else:
        st.success("**Lower likelihood of autism traits.** If you still have concerns, a doctor can advise.")
    if model_positive != rule_positive:
        st.info(f"The model and the AQ-10 cutoff (score above {aq10.REFERRAL_CUTOFF}) disagree here, "
                "so treat this result as uncertain.")

with st.expander("About this model"):
    test = metrics["test"]
    st.markdown(
        f"- One ensemble (logistic regression + random forest + gradient boosting) per age group; "
        f"the app uses the one matching the age entered.\n"
        f"- Held-out test set ({metrics['n_test']} people): accuracy {test['accuracy']:.1%}, "
        f"ROC AUC {test['roc_auc']:.3f}.\n"
        f"- The training labels come from the AQ-10 cutoff itself, not from clinical diagnoses, so the "
        f"model reproduces the questionnaire rather than adding diagnostic evidence beyond it."
    )
