
"""
app.py
Energy Cluster Predictor using K-Means
Run: streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------- Page Configuration ----------

st.set_page_config(
    page_title="Energy Cluster Predictor",
    page_icon="⚡",
    layout="wide"
)


# ---------- Step 1: Load Model and Dataset ----------

MODEL_PATH = "model/kmeans_model.joblib"
DATA_PATH = "data/energy_clustered.csv"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Make sure the trained K-Means model is saved in the model folder.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Make sure energy_clustered.csv is inside the data folder.")
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    scaler = saved["scaler"]
    FEATURES = saved["features"]

    data = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()


# ---------- Step 2: Validate Dataset ----------

required_columns = FEATURES + ["cluster"]

if not all(col in data.columns for col in required_columns):
    st.error(
        f"Dataset must contain these columns: {required_columns}"
    )
    st.stop()

if data.empty:
    st.error("The dataset is empty.")
    st.stop()


# ---------- Step 3: Calculate Cluster Centres ----------

centres = pd.DataFrame(
    scaler.inverse_transform(model.cluster_centers_),
    columns=FEATURES
)

avg_energy = data["energy_consumption"].mean()
avg_temp = data["temperature"].mean()

COLORS = [
    "#00B894",
    "#0984E3",
    "#FDCB6E",
    "#E17055",
    "#6C5CE7",
    "#FD79A8"
]

names = {}
advice = {}

for i, row in centres.iterrows():

    energy_level = (
        "High"
        if row["energy_consumption"] > avg_energy
        else "Low"
    )

    weather = (
        "Hot"
        if row["temperature"] > avg_temp
        else "Cool"
    )

    names[i] = f"{energy_level} usage · {weather} weather"

    if energy_level == "High":
        advice[i] = (
            "Heavy load: shift usage away from peak hours "
            "and use energy-efficient equipment."
        )
    else:
        advice[i] = (
            "Light load: this cluster represents relatively "
            "lower energy consumption."
        )

data["segment"] = data["cluster"].map(names)


# ---------- Step 4: Page Styling ----------

st.markdown("""
<style>

.hero {
    padding: 26px 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
    background: linear-gradient(
        120deg,
        #00B894,
        #0984E3 60%,
        #6C5CE7
    );
}

.hero h1 {
    margin: 0;
}

.hero p {
    margin: 6px 0 0;
    opacity: 0.95;
}

.result {
    padding: 22px;
    border-radius: 16px;
    background: #1A1F2B;
    border-left: 7px solid var(--c);
}

.result h2 {
    margin: 0;
    color: var(--c);
}

.result p {
    margin: 8px 0 0;
    color: #D0D6E0;
}

</style>

<div class="hero">
    <h1>⚡ Energy Cluster Predictor</h1>
    <p>
        Enter your energy consumption and temperature.
        The K-Means model predicts the corresponding energy group.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Step 5: User Input ----------

st.sidebar.header("🔧 Enter Values")

energy = st.sidebar.slider(
    "Energy Consumption (kWh)",
    min_value=50.0,
    max_value=500.0,
    value=275.0,
    step=1.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=15.0,
    max_value=35.0,
    value=25.0,
    step=0.1
)

st.sidebar.caption("Ranges should match the training data.")


# ---------- Step 6: Prediction ----------

new_data = pd.DataFrame(
    [[energy, temperature]],
    columns=FEATURES
)

new_scaled = scaler.transform(new_data)

cluster = int(model.predict(new_scaled)[0])

color = COLORS[cluster % len(COLORS)]


# ---------- Step 7: Display Prediction ----------

left, right = st.columns([1, 1.4])

with left:

    st.subheader("🔍 Prediction")

    st.markdown(
        f"""
        <div class="result" style="--c:{color}">
            <h2>Cluster {cluster}</h2>
            <b>{names[cluster]}</b>
            <p>{advice[cluster]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2 = st.columns(2)

    c1.metric(
        "Your Energy",
        f"{energy:.0f} kWh",
        f"{energy - avg_energy:+.0f} vs average"
    )

    c2.metric(
        "Temperature",
        f"{temperature:.1f} °C",
        f"{temperature - avg_temp:+.1f} vs average"
    )

    # Calculate distance from each cluster centre

    distances = np.linalg.norm(
        new_scaled - model.cluster_centers_,
        axis=1
    )

    st.caption(
        "Distance to each cluster centre (smaller = closer)"
    )

    distance_df = pd.DataFrame({
        "Cluster": [
            f"Cluster {i}"
            for i in range(len(distances))
        ],
        "Distance": distances
    })

    st.bar_chart(
        distance_df.set_index("Cluster")
    )


# ---------- Step 8: Visualize Clusters ----------

with right:

    st.subheader("📊 Where Your Reading Falls")

    sample_data = data.sample(
        n=min(2500, len(data)),
        random_state=1
    )

    fig = px.scatter(
        sample_data,
        x="temperature",
        y="energy_consumption",
        color="segment",
        opacity=0.4,
        template="plotly_dark",
        labels={
            "temperature": "Temperature (°C)",
            "energy_consumption": "Energy (kWh)"
        }
    )

    # Highlight user input

    fig.add_trace(
        go.Scatter(
            x=[temperature],
            y=[energy],
            mode="markers",
            name="Your Input",
            marker=dict(
                size=20,
                color="white",
                symbol="star",
                line=dict(width=2, color="black")
            )
        )
    )

    # Display cluster centres

    fig.add_trace(
        go.Scatter(
            x=centres["temperature"],
            y=centres["energy_consumption"],
            mode="markers",
            name="Cluster Centres",
            marker=dict(
                size=14,
                color="red",
                symbol="x"
            )
        )
    )

    fig.update_layout(
        height=470,
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------- Step 9: CSV Prediction ----------

st.divider()

st.subheader("📁 Predict from a CSV File")

st.caption(
    "Upload a CSV containing: energy_consumption, temperature"
)

file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if file is not None:

    try:

        df_in = pd.read_csv(file)

        if set(FEATURES).issubset(df_in.columns):

            df_in["cluster"] = model.predict(
                scaler.transform(df_in[FEATURES])
            )

            df_in["segment"] = df_in["cluster"].map(names)

            st.success("Predictions generated successfully!")

            st.dataframe(
                df_in,
                use_container_width=True
            )

            st.download_button(
                label="⬇️ Download Predictions",
                data=df_in.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

        else:

            st.error(
                f"Your CSV must contain these columns: {FEATURES}"
            )

    except Exception as e:

        st.error(f"Error processing CSV: {e}")


# ---------- Step 10: Cluster Summary ----------

with st.expander("📋 Cluster Summary"):

    summary = data.groupby("segment").agg(
        readings=("cluster", "size"),
        avg_energy=("energy_consumption", "mean"),
        avg_temp=("temperature", "mean")
    ).round(2)

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.caption(
        "Summary of energy consumption patterns "
        "identified by the K-Means algorithm."
    )
