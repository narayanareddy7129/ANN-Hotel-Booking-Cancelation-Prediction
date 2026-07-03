import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# COMPAT PATCH: newer scikit-learn versions (1.3+) added an internal
# `_RemainderColsList` class inside sklearn.compose._column_transformer to
# track remainder='passthrough' columns when pickling a ColumnTransformer.
# If preprocessor.pkl was created with a newer sklearn than what's installed
# here, unpickling fails with:
#   "Can't get attribute '_RemainderColsList' on
#    <module 'sklearn.compose._column_transformer' ...>"
# This stub recreates that class (it's just a list subclass) so joblib.load
# can resolve the reference, regardless of which sklearn version is active.
# ---------------------------------------------------------------------------
import sklearn.compose._column_transformer as _ct_module

if not hasattr(_ct_module, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Compat stub for older/newer scikit-learn versions."""
        pass

    _ct_module._RemainderColsList = _RemainderColsList

# ---------------------------------------------------------------------------
# IMPORTANT: FrequencyEncoder must be defined here, with this exact same
# class body, BEFORE joblib.load() is called. preprocessor.pkl was pickled
# while this class lived in the notebook's __main__ namespace. When you run
# `streamlit run app.py`, this file also becomes the __main__ module, so
# redefining the class here (same module name + same class name) lets
# pickle successfully reconstruct the fitted FrequencyEncoder instead of
# raising "Can't get attribute 'FrequencyEncoder' on <module 'main'>".
# ---------------------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.freq_maps_ = {}
        for col in X.columns:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.freq_maps_[col]).fillna(0)
        return X_transformed.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return input_features


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "model.keras"
PREPROCESSOR_PATH = "preprocessor.pkl"

ADR_UPPER_CAP = 226.875  # IQR upper bound computed in the training notebook

# Rare one-hot columns that appeared inconsistently across train/test splits
# during training; the final ANN was trained without them, so we drop them
# if the preprocessor's output still contains them.
DROP_IF_PRESENT = ["ohe__reserved_room_type_P", "ohe__assigned_room_type_P"]

RAW_FEATURE_COLUMNS = [
    "hotel", "arrival_date_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "assigned_room_type", "deposit_type", "days_in_waiting_list",
    "customer_type", "total_of_special_requests",
    "lead_time_log", "adr_capped", "waiting_log",
    "previous_cancellations_log", "previous_cancellations_bool", "total_night_stay",
]

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", page_icon="🏨", layout="wide")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def engineer_features(raw: dict) -> pd.DataFrame:
    lead_time_log = np.log1p(raw["lead_time"])
    adr_capped = min(raw["adr"], ADR_UPPER_CAP)
    waiting_log = np.log1p(raw["days_in_waiting_list"])
    previous_cancellations_log = np.log1p(raw["previous_cancellations"])
    previous_cancellations_bool = int(raw["previous_cancellations"] != 0)
    total_night_stay = raw["stays_in_weekend_nights"] + raw["stays_in_week_nights"]

    row = {
        "hotel": raw["hotel"],
        "arrival_date_month": raw["arrival_date_month"],
        "stays_in_weekend_nights": raw["stays_in_weekend_nights"],
        "stays_in_week_nights": raw["stays_in_week_nights"],
        "adults": raw["adults"],
        "children": raw["children"],
        "country": raw["country"],
        "market_segment": raw["market_segment"],
        "distribution_channel": raw["distribution_channel"],
        "is_repeated_guest": raw["is_repeated_guest"],
        "previous_cancellations": raw["previous_cancellations"],
        "previous_bookings_not_canceled": raw["previous_bookings_not_canceled"],
        "reserved_room_type": raw["reserved_room_type"],
        "assigned_room_type": raw["assigned_room_type"],
        "deposit_type": raw["deposit_type"],
        "days_in_waiting_list": raw["days_in_waiting_list"],
        "customer_type": raw["customer_type"],
        "total_of_special_requests": raw["total_of_special_requests"],
        "lead_time_log": lead_time_log,
        "adr_capped": adr_capped,
        "waiting_log": waiting_log,
        "previous_cancellations_log": previous_cancellations_log,
        "previous_cancellations_bool": previous_cancellations_bool,
        "total_night_stay": total_night_stay,
    }
    return pd.DataFrame([row], columns=RAW_FEATURE_COLUMNS)


def align_to_model_input(transformed_df: pd.DataFrame, expected_width: int) -> pd.DataFrame:
    df = transformed_df.copy()

    cols_to_drop = [c for c in DROP_IF_PRESENT if c in df.columns]
    if cols_to_drop and (df.shape[1] - len(cols_to_drop)) == expected_width:
        return df.drop(columns=cols_to_drop)

    if df.shape[1] > expected_width:
        df = df.iloc[:, :expected_width]
    elif df.shape[1] < expected_width:
        for i in range(expected_width - df.shape[1]):
            df[f"_pad_{i}"] = 0.0

    return df


def predict(model, preprocessor, input_df: pd.DataFrame) -> float:
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_df = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
    transformed_df = align_to_model_input(transformed_df, model.input_shape[-1])

    proba = float(model.predict(transformed_df.values, verbose=0)[0][0])
    return proba


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
try:
    model, preprocessor = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(
        "Could not load `model.keras` and/or `preprocessor.pkl`. "
        "Make sure both files are in the same folder as this app.\n\n"
        f"Details: {e}"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Hotel Booking Cancellation Predictor")
st.markdown("Enter raw booking details in the sidebar to predict whether the booking is likely to be **cancelled**.")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Booking Details")

hotel = st.sidebar.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
lead_time = st.sidebar.slider("Lead Time (days)", 0, 737, 90)
arrival_month = st.sidebar.selectbox("Arrival Month", [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
])
stays_weekend = st.sidebar.slider("Weekend Nights", 0, 20, 1)
stays_week = st.sidebar.slider("Weekday Nights", 0, 50, 2)
adults = st.sidebar.slider("Adults", 1, 10, 2)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 10])
country = st.sidebar.text_input("Country Code (e.g. PRT, GBR, USA)", "PRT")
market_segment = st.sidebar.selectbox("Market Segment", [
    "Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation", "Undefined",
])
distribution_channel = st.sidebar.selectbox("Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
reserved_room_type = st.sidebar.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L"])
assigned_room_type = st.sidebar.selectbox("Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"])
deposit_type = st.sidebar.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])
customer_type = st.sidebar.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
adr = st.sidebar.number_input("Average Daily Rate (ADR)", 0.0, 600.0, 100.0)
prev_cancellations = st.sidebar.slider("Previous Cancellations", 0, 26, 0)
prev_not_cancelled = st.sidebar.slider("Previous Bookings Not Cancelled", 0, 72, 0)
days_waiting = st.sidebar.slider("Days in Waiting List", 0, 391, 0)
total_special_requests = st.sidebar.slider("Total Special Requests", 0, 5, 0)
is_repeated_guest = st.sidebar.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x else "No")

raw_input = {
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_month": arrival_month,
    "stays_in_weekend_nights": stays_weekend,
    "stays_in_week_nights": stays_week,
    "adults": adults,
    "children": children,
    "country": country,
    "market_segment": market_segment,
    "distribution_channel": distribution_channel,
    "reserved_room_type": reserved_room_type,
    "assigned_room_type": assigned_room_type,
    "deposit_type": deposit_type,
    "customer_type": customer_type,
    "adr": adr,
    "previous_cancellations": prev_cancellations,
    "previous_bookings_not_canceled": prev_not_cancelled,
    "days_in_waiting_list": days_waiting,
    "total_of_special_requests": total_special_requests,
    "is_repeated_guest": is_repeated_guest,
}

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.sidebar.button("Predict", type="primary"):
    if not artifacts_loaded:
        st.warning("Cannot predict — model/preprocessor failed to load. See error above.")
    else:
        try:
            input_df = engineer_features(raw_input)
            proba = predict(model, preprocessor, input_df)
            pred = int(proba >= 0.5)

            st.subheader("Prediction Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                if pred == 1:
                    st.error("Likely to CANCEL")
                else:
                    st.success("Likely NOT to Cancel")
            with col2:
                st.metric("Cancellation Probability", f"{proba * 100:.1f}%")
            with col3:
                st.metric("Confidence", "High" if abs(proba - 0.5) > 0.3 else "Medium")

            st.divider()
            st.subheader("Booking Summary")
            st.dataframe(pd.DataFrame([raw_input]).T.rename(columns={0: "Value"}), use_container_width=True)

            with st.expander("See engineered feature values sent to the preprocessor"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Fill in the booking details on the left and click Predict.")

st.markdown("---")
st.caption(
    "Model: Artificial Neural Network (Keras Sequential, tuned with Optuna) · "
    "Preprocessing: FrequencyEncoder (country) + OneHotEncoder (categoricals) + StandardScaler (numeric)"
)
