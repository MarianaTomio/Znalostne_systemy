import pandas as pd
import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Naive Bayes", layout="centered")
st.title("Naive Bayesova inferenčná sieť")

try:
    df = pd.read_csv("weather_forecast.csv")
except FileNotFoundError:
    st.error("Súbor 'weather_forecast.csv' sa nenašiel v tomto priečinku.")
    st.stop()

st.subheader("Náhľad dát")
st.dataframe(df.head())

target_col = st.selectbox("Cieľ (target):", df.columns, index=len(df.columns) - 1)
feature_cols = [c for c in df.columns if c != target_col]
st.caption(f"Features: {', '.join(feature_cols)}")

if "model" not in st.session_state:
    st.session_state["model"] = None
    st.session_state["target_col"] = None
    st.session_state["feature_cols"] = None

if st.button("Natrénovať model"):
    # Naive Bayes štruktúra: všetky features -> target
    edges = [(f, target_col) for f in feature_cols]
    model = DiscreteBayesianNetwork(edges)

    model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5.0,
    )
    model.check_model()

    st.session_state["model"] = model
    st.session_state["target_col"] = target_col
    st.session_state["feature_cols"] = feature_cols

    st.success("Model natrénovaný.")
    st.write("Hrany:", edges)

    #A priori rozdelenie cieľovej premennej
    inference = VariableElimination(model)
    prior_distribution = inference.query(variables=[target_col])

    st.subheader("A priori rozdelenie cieľovej premennej")
    st.write(prior_distribution)

    with st.expander("Zobraziť inferenčnú sieť"):
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())

        fig, ax = plt.subplots(figsize=(5, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            node_size=2200,
            node_color="#89c2ff",
            ax=ax,
            arrows=False,
        )
        ax.set_axis_off()
        ax.set_title("Štruktúra Bayesovej siete")
        st.pyplot(fig)

#Inferencia s evidenciou (posterior)
if st.session_state.get("model") is not None:
    model = st.session_state["model"]
    target_col = st.session_state["target_col"]
    feature_cols = st.session_state["feature_cols"]

    st.subheader("Inferencia s evidenciou")

    #premenné, ktoré idu ako evidencia
    selected_features = st.multiselect(
        "Vyber premenné ako evidenciu:", feature_cols
    )

    evidence = {}
    for feat in selected_features:
        #možné hodnoty pre danú premennú (Sunny, Overcast, Rain, ...)
        possible_vals = sorted(df[feat].unique())
        val = st.selectbox(
            f"Hodnota pre {feat}:",
            possible_vals,
            key=f"evid_{feat}",
        )
        evidence[feat] = val

    if st.button("Vypočítať posterior"):
        inference = VariableElimination(model)

        #ak evidencia nie je zadaná, dostaneme opäť prior(pravdepodobnosť predtým)
        if evidence:
            posterior = inference.query(variables=[target_col], evidence=evidence)
        else:
            posterior = inference.query(variables=[target_col])

        st.write("Zadaná evidencia:", evidence)
        st.subheader("Posteriorné rozdelenie cieľovej premennej") #výsledná pravdepodobnosť
        st.write(posterior)

        try:
            state_names = posterior.state_names[target_col]
            probs = posterior.values
            result_df = pd.DataFrame(
                {"Hodnota": state_names, "Pravdepodobnosť": probs}
            )
            st.table(result_df)
        except Exception:
            pass


