"""
app.py : dashboard for PS-003 (Federated Traffic-Flow Prediction).

Put this file next to main.py (the backup program), data_pipeline.py and PEMS08.npz.
Install:  pip install streamlit pandas numpy torch matplotlib
Run:      streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import streamlit as st

import main as core
from data_pipeline import (build_clients, pick_sensors, naive_baseline_scores,
                           load_flow, comm_bytes_per_round)

st.set_page_config(page_title="Federated Traffic Forecasting", layout="wide")
LABELS = {"centralised": "Centralised", "fed_iid": "Federated (IID)",
          "fed_noniid": "Federated (non-IID)"}


@st.cache_data
def read_file(path):
    return load_flow(path)


@st.cache_data
def read_upload(name, raw):
    if name.lower().endswith(".npz"):
        z = np.load(io.BytesIO(raw))
        a = z["data" if "data" in z.files else z.files[0]]
        return (a[..., 0] if a.ndim == 3 else a).astype(np.float32)
    return pd.read_csv(io.BytesIO(raw)).select_dtypes("number").to_numpy(np.float32)


# ------------------------------ sidebar ------------------------------------
sb = st.sidebar
sb.header("1. Dataset")
source = sb.radio("Source", ["PEMS08 file", "Upload .npz / .csv", "Synthetic demo"])
flow, err = None, None
try:
    if source == "PEMS08 file":
        path = sb.text_input("Path", "PEMS08.npz")
        flow = read_file(path)
    elif source == "Upload .npz / .csv":
        up = sb.file_uploader("PEMS-style .npz, or CSV (rows = time, columns = sensors)",
                              type=["npz", "csv"])
        flow = read_upload(up.name, up.getvalue()) if up else None
    else:
        flow = core.make_synthetic_flow()
except Exception as e:  # show a readable message instead of a traceback
    err = f"Could not load data: {e}"

sb.header("2. Experiment settings")
n_clients = sb.slider("Clients (simulated intersections)", 3, 5, 4)
rounds = sb.slider("Federated rounds", 5, 40, 15)
local_epochs = sb.slider("Local epochs per round", 1, 3, 1)
input_len = sb.select_slider("Past steps used as input (5 min each)", [6, 12, 24], 12)
lr = sb.select_slider("Learning rate", [3e-4, 1e-3, 3e-3], 1e-3)
seeds = sb.multiselect("Random seeds", [0, 1, 2, 3, 4], [0, 1, 2])
run = sb.button("Run experiment", type="primary", disabled=flow is None)

st.title("Federated Traffic-Flow Prediction")
st.caption("PS-003 | Centralised vs federated (FedAvg), IID vs non-IID, with communication cost")
if err:
    st.error(err)
if flow is None:
    st.info("Choose or upload a dataset in the sidebar to begin.")
    st.stop()
if flow.ndim != 2 or flow.shape[1] < n_clients or flow.shape[0] < 500:
    st.error("Data must be 2-D (time x sensors) with at least 500 time steps and "
             f"{n_clients} sensors. Got shape {flow.shape}.")
    st.stop()

ids = pick_sensors(flow, n_clients)

# ------------------------------ run ----------------------------------------
if run and seeds:
    rows, naive, jobs, done = [], None, len(seeds) * 3, 0
    bar = st.progress(0.0, "Starting")
    for seed in seeds:
        for part in ("iid", "noniid"):
            clients, scaler = build_clients(flow, ids, input_len=input_len,
                                            partition=part, seed=seed)
            naive = naive or naive_baseline_scores(clients, scaler)
            bar.progress(done / jobs, f"Seed {seed}: federated {part}")
            for e in core.run_federated(clients, scaler, seed, rounds, local_epochs,
                                        lr, 128, input_len):
                rows.append(dict(method=f"fed_{part}", seed=seed, **e))
            done += 1
            if part == "noniid":
                bar.progress(done / jobs, f"Seed {seed}: centralised")
                for e in core.run_centralised(clients, scaler, seed,
                                              rounds * local_epochs, lr, 128,
                                              input_len):
                    rows.append(dict(method="centralised", seed=seed, **e))
                done += 1
    bar.progress(1.0, "Done")
    st.session_state.result = dict(df=pd.DataFrame(rows), naive=naive,
                                   local_epochs=local_epochs, rounds=rounds)

tab_how, tab_data, tab_res = st.tabs(["How it works", "Data and clients", "Results"])

# ------------------------------ tab 1 --------------------------------------
with tab_how:
    st.markdown("""
**The problem.** Sending raw traffic data from every roadside sensor to a central cloud costs
bandwidth and exposes private travel logs. In **federated learning**, each intersection trains
a model on its own data and sends only the model weights to a server.

**What this app simulates** (all on one computer, no real network):
1. Each client is one sensor. Its history is split **by time**: first 70% to train, last 30% to test, so no future data leaks into training.
2. Every round, the server sends the global model to all clients. Each trains locally and sends back its weights.
3. The server averages the weights, weighted by how much data each client has (**FedAvg**).
4. We record prediction error (**MAE, RMSE**, in vehicles per 5 minutes) and the **bytes** sent.

**Two comparisons the task asks for**
- **Centralised vs federated:** how much accuracy does keeping data local cost?
- **IID vs non-IID:** IID clients all see a similar mix of traffic. Non-IID clients each see only their own sensor, so busy and quiet junctions differ a lot.

Everything runs for several **random seeds**, so results show average and spread, not luck.
""")
    st.code("""FedAvg (pseudocode)
global_model = new small MLP
for each round:
    for each client k:
        local_model = copy(global_model)
        train local_model on client k's data
    global_model = sum over k of (n_k / n_total) * local_model_k
    bytes_sent  += 2 * clients * parameters * 4     # download + upload
    record MAE, RMSE of global_model on test data""", language="text")

# ------------------------------ tab 2 --------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Time steps", f"{flow.shape[0]:,}")
    c2.metric("Sensors", flow.shape[1])
    c3.metric("Approx. days (5-min data)", f"{flow.shape[0] / 288:.0f}")
    names = [f"Client {i + 1} (sensor {s})" for i, s in enumerate(ids)]
    st.subheader("Traffic at the chosen clients (first 3 days)")
    st.line_chart(pd.DataFrame(flow[:864, ids], columns=names))
    st.subheader("Average flow per client")
    st.bar_chart(pd.Series(flow[:, ids].mean(axis=0), index=names))
    st.caption("Large differences between clients are what make the data non-IID. "
               "Clients are picked at evenly spaced traffic levels to show this clearly.")

# ------------------------------ tab 3 --------------------------------------
with tab_res:
    res = st.session_state.get("result")
    if not res:
        st.info("Press **Run experiment** in the sidebar.")
    else:
        df, naive, le = res["df"].copy(), res["naive"], res["local_epochs"]
        cen = df.method == "centralised"
        df = df[~cen | (df.step % le == 0)]           # one point per federated-round equivalent
        df.loc[df.method == "centralised", "step"] //= le
        last = df.groupby("method").step.transform("max") == df.step
        fin = df[last].groupby("method").agg(
            MAE=("mae", "mean"), MAE_std=("mae", "std"),
            RMSE=("rmse", "mean"), RMSE_std=("rmse", "std"),
            Bytes=("bytes_cum", "max")).reindex(list(LABELS)).fillna(0)

        cols = st.columns(4)
        cols[0].metric("Naive baseline MAE", f"{naive[0]:.2f}")
        for col, m in zip(cols[1:], LABELS):
            gain = 100 * (1 - fin.loc[m, "MAE"] / naive[0])
            col.metric(LABELS[m] + " MAE", f"{fin.loc[m, 'MAE']:.2f}",
                       f"{gain:.0f}% vs naive")

        st.subheader("Convergence (average over seeds)")
        st.line_chart(df.groupby(["step", "method"]).mae.mean().unstack()
                      .rename(columns=LABELS), x_label="Round (federated) / epoch (centralised)",
                      y_label="MAE")
        st.subheader("Accuracy vs communication cost")
        fed = df[df.method.str.startswith("fed")]
        per_round = fed[fed.step == 1].bytes_cum.iloc[0]
        pv = fed.groupby(["step", "method"]).mae.mean().unstack().rename(columns=LABELS)
        pv.index = pv.index * per_round / 1e6
        st.line_chart(pv, x_label="Data sent so far (MB)", y_label="MAE")
        st.caption(f"Each round moves {per_round:,} bytes (download plus upload, 4 bytes per parameter, "
                   "all clients every round).")

        st.subheader("Final results (mean and std over seeds)")
        show = fin.rename(index=LABELS).round(2)
        show["Bytes"] = show["Bytes"].map(lambda b: f"{int(b):,}" if b else "n/a")
        st.dataframe(show)
        gap = fin.loc["fed_noniid", "MAE"] - fin.loc["fed_iid", "MAE"]
        st.success(f"Non-IID vs IID gap in MAE: {gap:+.2f} vehicles per 5 min. "
                   f"Federated vs centralised (IID): "
                   f"{fin.loc['fed_iid', 'MAE'] - fin.loc['centralised', 'MAE']:+.2f}.")
        st.download_button("Download all results (CSV)", res["df"].to_csv(index=False),
                           "results.csv", "text/csv")
