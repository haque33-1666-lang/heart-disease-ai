
import os
import pickle
import pandas as pd
import streamlit as st


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease AI Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .probability {
        font-size: 42px;
        font-weight: 700;
    }

    .risk-text {
        font-size: 25px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        BASE_DIR,
        "random_forest_model.pkl"
    )

    preprocessor_path = os.path.join(
        BASE_DIR,
        "preprocessor.pkl"
    )

    model_info_path = os.path.join(
        BASE_DIR,
        "model_info.pkl"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    with open(model_info_path, "rb") as f:
        model_info = pickle.load(f)

    return model, preprocessor, model_info


model, preprocessor, model_info = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease AI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning Based Heart Disease Risk Prediction System
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Model",
        model_info["model_name"]
    )


with metric2:

    st.metric(
        "Accuracy",
        f"{model_info['accuracy'] * 100:.2f}%"
    )


with metric3:

    st.metric(
        "Precision",
        f"{model_info['precision'] * 100:.2f}%"
    )


with metric4:

    st.metric(
        "F1 Score",
        f"{model_info['f1_score'] * 100:.2f}%"
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("❤️ Heart Disease AI")

st.sidebar.markdown(
    """
    ### About

    This application uses a trained
    **Random Forest machine learning model**
    to estimate heart disease risk based
    on patient information.
    """
)

st.sidebar.divider()

st.sidebar.subheader("Model Details")

st.sidebar.write(
    f"**Model:** {model_info['model_name']}"
)

st.sidebar.write(
    f"**Accuracy:** "
    f"{model_info['accuracy'] * 100:.2f}%"
)

st.sidebar.write(
    f"**Precision:** "
    f"{model_info['precision'] * 100:.2f}%"
)

st.sidebar.write(
    f"**Recall:** "
    f"{model_info['recall'] * 100:.2f}%"
)

st.sidebar.write(
    f"**F1 Score:** "
    f"{model_info['f1_score'] * 100:.2f}%"
)

st.sidebar.write(
    f"**ROC-AUC:** "
    f"{model_info['roc_auc']:.4f}"
)

st.sidebar.divider()

st.sidebar.info(
    "For research and educational purposes only."
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📋 Patient Information</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the patient's information below to estimate "
    "the heart disease risk."
)


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    general_health = st.selectbox(
        "General Health",
        [
            "Excellent",
            "Very Good",
            "Good",
            "Fair",
            "Poor"
        ]
    )


with col2:

    exercise = st.selectbox(
        "Exercise",
        [
            "No",
            "Yes"
        ]
    )


with col3:

    depression = st.selectbox(
        "Depression",
        [
            "No",
            "Yes"
        ]
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col4, col5, col6 = st.columns(3)


with col4:

    diabetes = st.selectbox(
        "Diabetes",
        [
            "No",
            "Yes"
        ]
    )


with col5:

    sex = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )


with col6:

    age_category = st.selectbox(
        "Age Category",
        [
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+"
        ]
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col7, col8, col9 = st.columns(3)


with col7:

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=70.0,
        step=0.1
    )


with col8:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )


with col9:

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "No",
            "Yes"
        ]
    )


# ------------------------------------------------------------
# ROW 4
# ------------------------------------------------------------

alcohol_consumption = st.selectbox(
    "Alcohol Consumption",
    [
        "No",
        "Yes"
    ]
)


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({

        "General_Health": [
            general_health
        ],

        "Exercise": [
            exercise
        ],

        "Depression": [
            depression
        ],

        "Diabetes": [
            diabetes
        ],

        "Sex": [
            sex
        ],

        "Age_Category": [
            age_category
        ],

        "Weight_(kg)": [
            weight
        ],

        "BMI": [
            bmi
        ],

        "Smoking_History": [
            smoking_history
        ],

        "Alcohol_Consumption": [
            alcohol_consumption
        ]
    })


    try:

        # ----------------------------------------------------
        # PREPROCESS INPUT
        # ----------------------------------------------------

        processed_input = (
            preprocessor.transform(
                input_data
            )
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probability = (
            model.predict_proba(
                processed_input
            )[0][1]
        )


        threshold = (
            model_info["prediction_threshold"]
        )


        prediction = int(
            probability >= threshold
        )


        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )


        result1, result2 = st.columns(2)


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        with result1:

            st.metric(
                "Heart Disease Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


        # ----------------------------------------------------
        # RISK RESULT
        # ----------------------------------------------------

        with result2:

            if prediction == 1:

                st.error(
                    "⚠️ HIGHER RISK"
                )

                st.markdown(
                    """
                    The model estimates a higher
                    probability of heart disease
                    for the entered information.
                    """
                )

            else:

                st.success(
                    "✅ LOWER RISK"
                )

                st.markdown(
                    """
                    The model estimates a lower
                    probability of heart disease
                    for the entered information.
                    """
                )


        # ----------------------------------------------------
        # PREDICTION SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📌 Prediction Summary"
        )


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                "**Patient Information**"
            )

            st.write(
                f"General Health: {general_health}"
            )

            st.write(
                f"Age Category: {age_category}"
            )

            st.write(
                f"Sex: {sex}"
            )

            st.write(
                f"BMI: {bmi}"
            )

            st.write(
                f"Exercise: {exercise}"
            )


        with summary_col2:

            st.write(
                "**Risk Information**"
            )

            st.write(
                f"Probability: "
                f"{probability * 100:.2f}%"
            )

            st.write(
                f"Threshold: "
                f"{threshold * 100:.2f}%"
            )

            st.write(
                f"Prediction: "
                f"{'Higher Risk' if prediction == 1 else 'Lower Risk'}"
            )

            st.write(
                f"Model: "
                f"{model_info['model_name']}"
            )


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.warning(
            """
            ⚠️ Medical Disclaimer:

            This application is intended for research
            and educational purposes only. The prediction
            generated by this machine learning model is
            not a medical diagnosis and should not replace
            professional medical advice.
            """
        )


    except Exception as e:

        st.error(
            "❌ Prediction Error"
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
    ❤️ Heart Disease AI Prediction System<br>
    Random Forest Machine Learning Model<br>
    Research Project
    </div>
    """,
    unsafe_allow_html=True
)
