import streamlit as st
import requests
import pandas as pd
import altair as alt
import os

# --- CONFIG ---
API_URL = os.getenv("API_URL", "http://localhost:8000/")

st.title("🔍 Loan Approval Predictor")
st.markdown(
    "Enter a few simple details about the applicant and click **Analyze application**. "
    "It will show the estimated approval chance, the top factors that drove the decision, "
    "and a suggested interest rate (if any) to improve the chances."
)

linkedin = "https://www.linkedin.com/in/jaymin-mistry-data-science"
st.markdown(f'You can connect with me on LinkedIn [here]({linkedin}).')

# Map backend feature keys -> friendly labels (matches the form)
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


# --- INPUTS ---
with st.form("loan_form"):
    st.markdown("### Applicant details")
    left, right = st.columns(2)

    with left:
        person_age = st.number_input("Age (years)", min_value=18, max_value=80, value=35, step=1)
        person_education = st.selectbox(
            "Education",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
            index=2,
        )
        person_income = st.number_input(
            "Annual income (USD)", min_value=8000.0, max_value=175000.0, value=50000.0, step=10000.0, format="%.2f",
            help="Enter the gross annual income (before tax)."
        )
        person_home_ownership = st.selectbox(
            "Home ownership",
            ["Rent", "Mortgage", "Own", "Other"],
            index=1,
        )
        previous_loan_defaults = st.selectbox(
            "Previous loan defaults", ["No", "Yes"], index=0,
            help="Has the applicant defaulted on a loan before?"
        )

    with right:
        loan_amount = st.number_input(
            "Loan amount (USD)", min_value=500.0, max_value=25000.0, value=10000.0, step=1000.0, format="%.2f"
        )
        loan_intent = st.selectbox(
            "Purpose of loan",
            ["Venture", "Medical", "Personal", "Education", "Home Improvement", "Debt Consolidation"],
            index=2,
        )
        loan_interest_rate = st.number_input(
            "Interest rate (percent)", min_value=5.0, max_value=20.0, value=12.0, step=0.01, format="%.2f",
            help="The interest rate for this loan offer."
        )
        credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650, step=1)

    submitted = st.form_submit_button("Analyze application", type="primary")

# --- CALL API & SHOW RESULTS ---
if submitted:
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

    with st.spinner("Analyzing application — this usually takes a few seconds..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error("Couldn't reach the model API. Please check the server and try again.")
            st.stop()

    # Extract scalars
    current_pred = data.get("current_pred")
    new_pred = data.get("new_pred")
    rec_rate = data.get("rec_rate")
    
    # Convert to percent where appropriate (guard against None)
    current_pred_pct = f"{current_pred * 100:.2f}%" if current_pred is not None else "n/a"
    new_pred_pct = f"{new_pred * 100:.2f}%" if (new_pred is not None) else "n/a"

    # Extract feature contributions (everything except the scalar keys)
    contributions = {k: v for k, v in data.items() if k not in ("current_pred", "new_pred", "rec_rate")}
    df = pd.DataFrame.from_dict(contributions, orient="index", columns=["contribution"]).reset_index()
    df = df.rename(columns={"index": "feature"})
    df["abs_contribution"] = df["contribution"].abs()
    
    # Map backend keys to friendly display names
    df["feature_display"] = df["feature"].map(DISPLAY_NAME_MAP)
    df = df.sort_values("abs_contribution", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current approval chance", current_pred_pct)
    col2.metric("Recommended interest rate", f"{rec_rate:.2f}%" if rec_rate is not None else "n/a")
    col3.metric("Approval chance with the new rate", new_pred_pct)

    st.subheader("Factor contributions")
    st.caption("The chart below shows which applicant details moved the decision the most. "
               "Blue bars helped the applicant; red bars made approval less likely.")

    # Altair bar chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Effect on approval (%)"),
            y=alt.Y("feature_display:N", sort=alt.SortField("abs_contribution", order="descending"), title="Factor"),
            color=alt.condition(alt.datum.contribution > 0, alt.value("#1f77b4"), alt.value("#d62728")),
            tooltip=[
                alt.Tooltip("feature_display:N", title="Factor"),
                alt.Tooltip("contribution:Q", title="Effect", format=".2f")
            ]
        )
        .properties(height=400)
    )
    
    st.altair_chart(chart, width="stretch")
    
    spacer = st.empty()
    spacer.markdown("<br>", unsafe_allow_html=True)
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
