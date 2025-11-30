import sys
from datetime import datetime

import streamlit as st

# Make sure Python can find the src/ package
# (when running from project root, this usually isn't needed,
#  but it's harmless and makes things robust)
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from doctor_patient.crew import DoctorPatient  # type: ignore


st.set_page_config(
    page_title="Doctor–Patient Dialogue Crew",
    layout="wide",
)

st.title("🩺 Doctor–Patient Research Assistant")

st.markdown(
    """
This app uses your **CrewAI DoctorPatient crew**:

- `researcher` → gathers information about a topic  
- `reporting_analyst` → turns it into a detailed markdown report  

Fill in the topic and run the crew.
"""
)

# Inputs
default_year = datetime.now().year
topic = st.text_input(
    "Topic",
    placeholder="e.g. Type 2 diabetes lifestyle changes",
)
current_year = st.number_input(
    "Current year",
    value=default_year,
    step=1,
)

run_button = st.button("🚀 Run crew")

if run_button:
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        st.info("Running crew… this may take a bit.")
        with st.spinner("Agents are working..."):
            crew = DoctorPatient().crew()
            result = crew.kickoff(
                inputs={
                    "topic": topic.strip(),
                    "current_year": int(current_year),
                }
            )

        st.success("Done!")

        st.subheader("📄 Generated Report")
        # reporting_task is already configured to output markdown
        st.markdown(result)
