from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# 2. Project Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "logistic_regression_model.pkl"
)

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)


# --------------------------------------------------
# 3. Load Model and Dataset
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_data()

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}"
    )
    st.stop()

except Exception as error:
    st.error(f"Error loading model or data: {error}")
    st.stop()


# --------------------------------------------------
# 4. Prepare Feature Information
# --------------------------------------------------

if "Churn" not in df.columns:
    st.error("The dataset must contain a 'Churn' column.")
    st.stop()

X = df.drop(columns=["Churn"])

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# --------------------------------------------------
# 5. Application Header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
    """
    ### Predict Customer Churn Using Machine Learning

    This application uses a trained Logistic Regression
    model to predict whether a customer is likely to leave
    the company.

    Enter customer details below to generate a prediction.
    """
)

st.divider()


# --------------------------------------------------
# 6. Dashboard Metrics
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    f"{len(df):,}"
)

col2.metric(
    "Model",
    "Logistic Regression"
)

col3.metric(
    "ROC-AUC Score",
    "0.836"
)

st.divider()


# --------------------------------------------------
# 7. Customer Input Form
# --------------------------------------------------

st.subheader("👤 Enter Customer Details")

st.write(
    "Provide the customer's information below."
)

customer_data = {}

with st.form("customer_prediction_form"):

    st.markdown("#### Customer Information")

    col1, col2 = st.columns(2)

    for index, column in enumerate(X.columns):

        if column in numeric_columns:

            minimum = float(X[column].min())
            maximum = float(X[column].max())
            default = float(X[column].median())

            if pd.api.types.is_integer_dtype(X[column]):

                with col1 if index % 2 == 0 else col2:

                    customer_data[column] = st.number_input(
                        column.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1
                    )

            else:

                with col1 if index % 2 == 0 else col2:

                    customer_data[column] = st.number_input(
                        column.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.1
                    )

        elif column in categorical_columns:

            options = (
                X[column]
                .dropna()
                .unique()
                .tolist()
            )

            if not options:
                continue

            with col1 if index % 2 == 0 else col2:

                customer_data[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options=options
                )

        else:

            with col1 if index % 2 == 0 else col2:

                customer_data[column] = st.text_input(
                    column.replace("_", " ").title()
                )

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# --------------------------------------------------
# 8. Make Prediction
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame(
        [customer_data],
        columns=X.columns
    )

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        positive_index = next(
            (
                i
                for i, label in enumerate(classes)
                if str(label).strip().lower() == "yes"
            ),
            1
        )

        churn_probability = probabilities[positive_index]

        st.divider()

        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if str(prediction).strip().lower() == "yes":

                st.error(
                    "⚠️ This customer is likely to churn."
                )

            else:

                st.success(
                    "✅ This customer is likely to stay."
                )

        with col2:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        st.progress(
            float(churn_probability)
        )

        st.markdown(
            "### 💡 Customer Retention Recommendation"
        )

        if churn_probability >= 0.70:

            st.warning(
                """
                **High Churn Risk**

                - Contact the customer proactively.
                - Offer personalized retention benefits.
                - Review pricing and service concerns.
                - Provide dedicated customer support.
                """
            )

        elif churn_probability >= 0.40:

            st.info(
                """
                **Medium Churn Risk**

                - Monitor customer engagement.
                - Offer relevant service upgrades.
                - Collect customer feedback.
                - Improve customer experience.
                """
            )

        else:

            st.success(
                """
                **Low Churn Risk**

                - Maintain service quality.
                - Reward customer loyalty.
                - Continue engagement activities.
                """
            )

        st.caption(
            "Prediction is based on the trained machine "
            "learning model and provided customer details."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Machine Learning Project | "
    "Built using Python, Scikit-learn and Streamlit"
)