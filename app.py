from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="C",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}


FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


@st.cache_resource
def load_models():
    loaded = {}
    missing = []

    for name, path in MODEL_FILES.items():
        if path.exists():
            loaded[name] = joblib.load(path)
        else:
            missing.append(str(path))

    return loaded, missing


models, missing_models = load_models()


st.title("Credit Risk Analyzer")
st.caption(
    "Estimate the probability of serious delinquency within the next two years."
)

st.warning(
    "Educational project only. This model is not a validated lending system "
    "and must not be used on its own to approve, reject, or price credit."
)

if missing_models:
    st.error(
        "Some model files were not found. Check that these files exist:\n\n"
        + "\n".join(f"- `{path}`" for path in missing_models)
    )

if not models:
    st.stop()


with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox(
        "Choose a trained model",
        list(models.keys()),
    )
    st.divider()
    st.caption(
        "The selected model is loaded from your local models directory."
    )


st.subheader("Customer financial details")
st.write("Enter the values for one customer, then select **Predict risk**.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        utilization = st.number_input(
            "Revolving utilization of unsecured lines",
            min_value=0.0,
            value=0.30,
            step=0.05,
            help="Use a value in the same units as the training dataset.",
        )
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=120,
            value=40,
            step=1,
        )
        past_due_30_59 = st.number_input(
            "Times 30–59 days past due",
            min_value=0,
            value=0,
            step=1,
        )
        debt_ratio = st.number_input(
            "Debt ratio",
            min_value=0.0,
            value=0.30,
            step=0.05,
        )
        monthly_income = st.number_input(
            "Monthly income",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            help="Use the same currency and units as the training data.",
        )

    with col2:
        open_credit_lines = st.number_input(
            "Open credit lines and loans",
            min_value=0,
            value=8,
            step=1,
        )
        past_due_90 = st.number_input(
            "Times 90+ days past due",
            min_value=0,
            value=0,
            step=1,
        )
        real_estate_loans = st.number_input(
            "Real estate loans or lines",
            min_value=0,
            value=1,
            step=1,
        )
        past_due_60_89 = st.number_input(
            "Times 60–89 days past due",
            min_value=0,
            value=0,
            step=1,
        )
        dependents = st.number_input(
            "Number of dependents",
            min_value=0,
            value=0,
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict risk",
        type="primary",
        use_container_width=True,
    )


if submitted:
    customer = pd.DataFrame(
        [[
            utilization,
            age,
            past_due_30_59,
            debt_ratio,
            monthly_income,
            open_credit_lines,
            past_due_90,
            real_estate_loans,
            past_due_60_89,
            dependents,
        ]],
        columns=FEATURES,
    )

    try:
        model = models[selected_model]
        probability = float(model.predict_proba(customer)[0, 1])

        st.divider()
        st.subheader("Prediction")

        metric_col, detail_col = st.columns([1, 2])

        with metric_col:
            st.metric(
                "Estimated delinquency probability",
                f"{probability:.1%}",
            )
            st.progress(max(0.0, min(1.0, probability)))

        with detail_col:
            st.write(f"**Model:** {selected_model}")
            st.write(
                "This is the model's estimated probability of the target "
                "class `SeriousDlqin2yrs = 1`."
            )
            st.write(
                "It is not necessarily a calibrated real-world probability. "
                "No risk-category cutoff has been validated for this project."
            )

        with st.expander("Review submitted input"):
            st.dataframe(customer, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the saved model was trained with these exact feature "
            "names and that its dependencies are installed."
        )


with st.expander("About this project"):
    st.markdown(
        """
        - **Target:** `SeriousDlqin2yrs`
        - **Models:** Logistic Regression, Decision Tree, Random Forest
        - **Dataset:** Give Me Some Credit
        - **Output:** Estimated probability for the positive class
        """
    )
