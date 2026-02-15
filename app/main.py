"""
This Streamlit application provides a user interface for interacting with the
Loan Approval Prediction API. It allows users to input applicant details,
submit them to a backend FastAPI service, and visualize:

- Predicted loan approval probability
- Feature-level contribution breakdown (SHAP values)
- Optional recommended interest rate to improve approval odds
- Updated probability after applying the recommendation

Notes
-----
    1. This application is intended for demonstration purposes.
    2. All heavy computation (model inference, SHAP, LIME) occurs in the backend.
    3. The frontend is stateless and does not store user data.

"""


# ---------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------

import os
# Operating system utilities (file paths, env)


# ---------------------------------------------------------------------
# Third-Party Libraries
# ---------------------------------------------------------------------

import streamlit as st      # Frontend UI framework
import requests             # HTTP client for calling backend API
import pandas as pd         # Data manipulation (SHAP results formatting)
import altair as alt        # Declarative visualization library for charts


# =====================================================================
# Configuration
# =====================================================================

# Read backend API URL from environment variable.
# Defaults to local development server if not provided.
API_URL = os.getenv("API_URL", "http://localhost:8000/")


# =====================================================================
# Page Header
# =====================================================================

st.title("🔍 Loan Approval Predictor")

st.markdown(
    "Enter a few simple details about the applicant and click **Analyze application**. "
    "It will show the estimated approval chance, the top factors that drove the decision, "
    "and a suggested interest rate (if any) to improve the chances."
)

linkedin = "https://www.linkedin.com/in/jaymin-mistry-data-science"
st.markdown(f'You can connect with me on LinkedIn [here]({linkedin}).')


# =====================================================================
# Display Name Mapping
# =====================================================================

# Maps backend feature keys to user-friendly labels for chart display.
DISPLAY_NAME_MAP = {
    "person_age": "Age",
    "person_education": "Education",
    "person_income": "Annual income",
    "person_home_ownership": "Home ownership",
    "previous_loan_defaults": "Previous loan defaults",
    "loan_amount": "Loan amount",
    "loan_intent": "Purpose",
    "loan_interest_rate": "Interest rate",
    "credit_score": "Credit score",
}


# =====================================================================
# User Input Form
# =====================================================================

# Uses Streamlit form to batch user inputs and submit once.
with st.form("loan_form"):

    st.markdown("### Applicant details")

    # Split layout into two columns for better UI organization
    left, right = st.columns(2)

    # Left Column Inputs
    with left:
        person_age = st.number_input(
            "Age (years)", min_value=18, max_value=80, value=35, step=1
        )

        person_education = st.selectbox(
            "Education",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
            index=2,
        )

        person_income = st.number_input(
            "Annual income (USD)",
            min_value=8000.0,
            max_value=175000.0,
            value=50000.0,
            step=10000.0,
            format="%.2f",
            help="Enter the gross annual income (before tax)."
        )

        person_home_ownership = st.selectbox(
            "Home ownership",
            ["Rent", "Mortgage", "Own", "Other"],
            index=1,
        )

        previous_loan_defaults = st.selectbox(
            "Previous loan defaults",
            ["No", "Yes"],
            index=0,
            help="Has the applicant defaulted on a loan before?"
        )

    # Right Column Inputs
    with right:
        loan_amount = st.number_input(
            "Loan amount (USD)",
            min_value=500.0,
            max_value=25000.0,
            value=10000.0,
            step=1000.0,
            format="%.2f"
        )

        loan_intent = st.selectbox(
            "Purpose of loan",
            ["Venture", "Medical", "Personal", "Education", "Home Improvement", "Debt Consolidation"],
            index=2,
        )

        loan_interest_rate = st.number_input(
            "Interest rate (percent)",
            min_value=5.0,
            max_value=20.0,
            value=12.0,
            step=0.01,
            format="%.2f",
            help="The interest rate for this loan offer."
        )

        credit_score = st.number_input(
            "Credit score",
            min_value=300,
            max_value=900,
            value=650,
            step=1
        )

    # Submit button triggers API call
    submitted = st.form_submit_button("Analyze application", type="primary")


# =====================================================================
# API Call & Result Display
# =====================================================================

if submitted:

    # Construct API payload
    payload = {
        "person_age": int(person_age),
        "person_education": person_education,
        "person_income": float(person_income),
        "person_home_ownership": person_home_ownership,
        "loan_amount": float(loan_amount),
        "loan_intent": loan_intent,
        "loan_interest_rate": float(loan_interest_rate),
        "credit_score": int(credit_score),
        "previous_loan_defaults": previous_loan_defaults,
    }

    # Call backend API. Displays spinner while waiting
    with st.spinner("Analyzing application — this usually takes a few seconds..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=30)
            resp.raise_for_status()  # Raise error for HTTP 4xx/5xx
            data = resp.json()
        except requests.exceptions.RequestException:
            st.error("Couldn't reach the model API. Please check the server and try again.")
            st.stop()

    # Extract scalar results
    current_pred = data.get("current_pred")
    new_pred = data.get("new_pred")
    rec_rate = data.get("rec_rate")

    # Convert probabilities to percent strings for display
    current_pred_pct = f"{current_pred * 100:.2f}%" if current_pred is not None else "n/a"
    new_pred_pct = f"{new_pred * 100:.2f}%" if new_pred is not None else "n/a"

    # Extract feature contributions (SHAP values)
    contributions = {
        k: v for k, v in data.items()
        if k not in ("current_pred", "new_pred", "rec_rate")
    }

    df = (
        pd.DataFrame.from_dict(contributions, orient="index", columns=["contribution"])
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    df["abs_contribution"] = df["contribution"].abs()

    # Map backend feature keys to UI-friendly labels
    df["feature_display"] = df["feature"].map(DISPLAY_NAME_MAP).fillna(df["feature"])

    # Sort by strongest impact
    df = df.sort_values("abs_contribution", ascending=False)

    # Display Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current approval chance", current_pred_pct)
    col2.metric(
        "Recommended interest rate",
        f"{rec_rate:.2f}%" if rec_rate is not None else "n/a"
    )
    col3.metric("Approval chance with the new rate", new_pred_pct)

    # Display SHAP Contribution Chart
    st.subheader("Factor contributions")
    st.caption(
        "Blue bars helped the applicant; red bars made approval less likely."
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Effect on approval (%)"),
            y=alt.Y(
                "feature_display:N",
                sort=alt.SortField("abs_contribution", order="descending"),
                title="Factor"
            ),
            color=alt.condition(
                alt.datum.contribution > 0,
                alt.value("#1f77b4"),
                alt.value("#d62728")
            ),
            tooltip=[
                alt.Tooltip("feature_display:N", title="Factor"),
                alt.Tooltip("contribution:Q", title="Effect", format=".2f")
            ]
        )
        .properties(height=400)
    )

    st.altair_chart(chart, width="stretch")

    # Add some empty space after the chart
    spacer = st.empty()
    spacer.markdown("<br>", unsafe_allow_html=True)

    # Disclaimer Section
    st.markdown(
        """
        <div>

            ⚠️ Note

            This app is for demonstration only and not an official loan decision tool.

            The approval chance shown here is an estimate and may not reflect real-world 
            decisions. It might not make sense for all kinds of inputs. More training 
            data and validation could lead to better predictions and clearer explanations.
        
        </div>
        """,
        unsafe_allow_html=True,
    )
