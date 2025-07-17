import streamlit as st
import pandas as pd
import joblib
import base64

# Load trained model
model = joblib.load("best_random_forest_model.pkl")

# --- Add background image and full styling ---
def add_bg_image(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #FFFFFF;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.75);
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        section[data-testid="stSidebar"] .st-radio label span {{
            font-weight: 600;
            font-size: 16px;
        }}

        .block-container {{
            background-color: rgba(0,0,0,0.6);
            padding: 3rem 4rem;
            border-radius: 16px;
        }}

        .stTextInput>div>div>input,
        .stSelectbox>div>div>div>input,
        .stNumberInput>div>div>input {{
            background-color: rgba(255,255,255,0.1) !important;
            color: white !important;
            border: 1px solid #FFD700;
            border-radius: 10px;
        }}

        .stSelectbox label, .stSlider label, .stNumberInput label {{
            color: #FFD700 !important;
            font-weight: 600;
        }}

        .stSlider > div {{
            background-color: rgba(255,255,255,0.1);
            border: 1px solid #FFD700;
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 18px;
        }}

        .stButton>button {{
            background-color: #FFD700;
            color: #000;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
        }}

        .stDownloadButton>button {{
            background-color: #FF4C61;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
        }}

        .stDataFrame thead tr th {{
            background-color: #FFD700;
            color: #000;
        }}

        .stDataFrame tbody tr td {{
            background-color: rgba(255,255,255,0.05);
            color: white;
        }}

        .custom-title {{
            text-align: center;
            background: linear-gradient(to right, #FFD700, #FF8C00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 42px;
            font-weight: 900;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin-top: 20px;
        }}

        .custom-subtitle {{
            text-align: center;
            font-size: 20px;
            font-style: italic;
            color: #FFFFFF;
            text-shadow: 1px 1px 3px black;
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
add_bg_image("background.jpg")

# --- Styled Title ---
st.markdown("<div class='custom-title'>📦 Smart Delivery ETA Checker</div>", unsafe_allow_html=True)
st.markdown("<div class='custom-subtitle'>Predict if your ecommerce product will be delivered <i>on time</i>.</div>", unsafe_allow_html=True)
st.markdown("---")

# --- Prediction Logic ---
def predict_delivery(df):
    predictions = model.predict(df)
    return ["✅ On Time" if p == 1 else "❌ Not On Time" for p in predictions]

# --- Sidebar Input Selection ---
input_mode = st.sidebar.radio("Choose Input Method", ["Manual Entry", "CSV Upload"])

# --- Manual Entry ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Order Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        customer_rating = st.slider("Customer Rating", 1, 5, 3)
        prior_purchases = st.selectbox("Prior Purchases", list(range(0, 21)))
        customer_calls = st.slider("Customer Care Calls", 0, 10, 3)

    with col2:
        cost = st.slider("Cost of the Product", 0, 500, 150)
        weight = st.slider("Weight in grams", 100, 10000, 3000, step=100)

    with col3:
        discount = st.slider("Discount Offered (%)", 0, 100, 10)
        mode_of_shipment = st.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
        warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])
        product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])

    input_df = pd.DataFrame([{
        "Customer_rating": customer_rating,
        "Cost_of_the_Product": cost,
        "Discount_offered": discount,
        "Prior_purchases": prior_purchases,
        "Weight_in_gms": weight,
        "Customer_care_calls": customer_calls,
        "Mode_of_Shipment": mode_of_shipment,
        "Warehouse_block": warehouse_block,
        "Product_importance": product_importance
    }])

    if st.button("🔮 Predict"):
        result = predict_delivery(input_df)
        st.success("✅ Prediction complete!")

        # Styled Result Box
        color = "#4CAF50" if "On Time" in result[0] else "#FF5252"
        st.markdown(
            f"""
            <div style='
                background-color: rgba(255, 255, 255, 0.08);
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                color: {color};
                box-shadow: 0 0 12px {color};
            '>
                📈 Prediction Result: {result[0]}
            </div>
            """,
            unsafe_allow_html=True
        )

# --- CSV Upload ---
elif input_mode == "CSV Upload":
    st.subheader("📤 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("📊 Preview of Uploaded Data:")
            st.dataframe(data.head())

            if st.button("🔮 Predict"):
                result = predict_delivery(data)
                data["Prediction"] = result
                st.success("✅ Prediction added to uploaded data!")
                st.dataframe(data)

                csv = data.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions as CSV", csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error reading the file: {e}")
