import streamlit as st
from src.doctor_patient.crew import (
    run_symptom_flow,
    run_drug_flow,
    run_summary_flow,
)

st.set_page_config(page_title="Doctor–Patient Assistant", layout="centered")

st.title("🧪 Doctor–Patient Dialogue Assistant")
st.caption("Research-only demo. Not medical advice.")


# ------------------------------
# STATE
# ------------------------------

if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.chief = ""
    st.session_state.symptom_options = []
    st.session_state.selected_symptoms = []
    st.session_state.drug_options = []
    st.session_state.selected_drugs = []
    st.session_state.summary = ""


def goto(step: int):
    st.session_state.step = step
    st.rerun()


# ------------------------------
# STEP 1 – Chief complaint
# ------------------------------

if st.session_state.step == 1:
    st.header("Step 1 — Describe your symptoms")

    chief = st.text_area("Your complaint:", value=st.session_state.chief, height=150)

    if st.button("Find possible symptoms"):
        if chief.strip():
            st.session_state.chief = chief.strip()
            with st.spinner("Analyzing..."):
                st.session_state.symptom_options = run_symptom_flow(chief)
            goto(2)
        else:
            st.warning("Please enter your complaint.")


# ------------------------------
# STEP 2 – Confirm symptoms
# ------------------------------

elif st.session_state.step == 2:
    st.header("Step 2 — Select symptoms")

    opts = st.session_state.symptom_options
    if not opts:
        st.error("No symptom options found. Try again.")
        if st.button("Back"):
            goto(1)
    else:
        selected = st.multiselect("Which symptoms apply?", options=opts)
        none = st.checkbox("None of these")

        if st.button("Continue"):
            if none:
                st.session_state.selected_symptoms = []
            else:
                st.session_state.selected_symptoms = selected

            with st.spinner("Finding drug history patterns..."):
                st.session_state.drug_options = run_drug_flow(
                    st.session_state.chief,
                    st.session_state.selected_symptoms,
                )
            goto(3)

        if st.button("Back"):
            goto(1)


# ------------------------------
# STEP 3 – Confirm medications
# ------------------------------

elif st.session_state.step == 3:
    st.header("Step 3 — Select medications")

    opts = st.session_state.drug_options
    selected = st.multiselect("Which medications apply?", options=opts)
    none = st.checkbox("None of these medications")

    if st.button("Generate summary"):
        if none:
            st.session_state.selected_drugs = []
        else:
            st.session_state.selected_drugs = selected

        with st.spinner("Generating summary..."):
            st.session_state.summary = run_summary_flow(
                st.session_state.chief,
                st.session_state.selected_symptoms,
                st.session_state.selected_drugs,
            )
        goto(4)

    if st.button("Back"):
        goto(2)


# ------------------------------
# STEP 4 – Summary
# ------------------------------

elif st.session_state.step == 4:
    st.header("Step 4 — Summary")
    st.markdown(st.session_state.summary)

    st.info("This is a research demo. Not medical advice.")

    if st.button("Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
