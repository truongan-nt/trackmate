# frontend/app.py

import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

# Streamlit Layout
st.title("TrackMate - MLOps Assistant")

menu = ["Create Experiment", "View Experiments", "Manage Runs", "Ask Assistant"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Create Experiment":
    st.subheader("Create a New Experiment")

    with st.form("create_experiment_form"):
        exp_name = st.text_input("Experiment Name")
        exp_desc = st.text_area("Description")
        submit = st.form_submit_button("Create Experiment")

        if submit:
            if exp_name:
                payload = {
                    "name": exp_name,
                    "description": exp_desc
                }
                response = requests.post(f"{BACKEND_URL}/experiments/", json=payload)
                if response.status_code == 200:
                    res = response.json()
                    st.success(f"Experiment '{res['name']}' created with ID: {res['experiment_id']}")
                else:
                    st.error(f"Error: {response.json()['detail']}")
            else:
                st.error("Experiment name cannot be empty.")

elif choice == "View Experiments":
    st.subheader("List of Experiments")

    response = requests.get(f"{BACKEND_URL}/experiments/")
    if response.status_code == 200:
        experiments = response.json()["experiments"]
        if experiments:
            exp_df = pd.DataFrame(experiments)
            st.dataframe(exp_df)
        else:
            st.info("No experiments found.")
    else:
        st.error(f"Error: {response.json()['detail']}")

elif choice == "Manage Runs":
    st.subheader("Manage Runs")

    # Fetch experiments to select from
    exp_response = requests.get(f"{BACKEND_URL}/experiments/")
    if exp_response.status_code == 200:
        experiments = exp_response.json()["experiments"]
        if experiments:
            exp_names = [exp["name"] for exp in experiments]
            selected_exp = st.selectbox("Select Experiment", exp_names)
            selected_exp_id = next(exp["experiment_id"] for exp in experiments if exp["name"] == selected_exp)

            # Start a new run
            with st.expander("Start a New Run"):
                run_name = st.text_input("Run Name (optional)")
                if st.button("Start Run"):
                    payload = {"run_name": run_name}
                    run_response = requests.post(f"{BACKEND_URL}/experiments/{selected_exp_id}/runs/", json=payload)
                    if run_response.status_code == 200:
                        run_res = run_response.json()
                        st.success(f"Run started with ID: {run_res['run_id']} and Status: {run_res['status']}")
                    else:
                        st.error(f"Error: {run_response.json()['detail']}")

            # Log parameters and metrics
            st.subheader("Log Parameters and Metrics")

            run_id = st.text_input("Enter Run ID")
            if run_id:
                with st.form("log_form"):
                    param_key = st.text_input("Parameter Key")
                    param_value = st.text_input("Parameter Value")
                    metric_key = st.text_input("Metric Key")
                    metric_value = st.number_input("Metric Value", min_value=0.0)
                    submit = st.form_submit_button("Log Parameter and Metric")

                    if submit:
                        if param_key and param_value and metric_key:
                            # Log parameter
                            param_payload = {"key": param_key, "value": param_value}
                            param_response = requests.post(f"{BACKEND_URL}/runs/{run_id}/params/", json=param_payload)
                            if param_response.status_code == 200:
                                st.success(param_response.json()["message"])
                            else:
                                st.error(f"Error: {param_response.json()['detail']}")

                            # Log metric
                            metric_payload = {"key": metric_key, "value": metric_value}
                            metric_response = requests.post(f"{BACKEND_URL}/runs/{run_id}/metrics/", json=metric_payload)
                            if metric_response.status_code == 200:
                                st.success(metric_response.json()["message"])
                            else:
                                st.error(f"Error: {metric_response.json()['detail']}")
                        else:
                            st.error("All fields except Metric Value are required.")

            # View run details
            with st.expander("View Run Details"):
                if st.button("Get Run Details"):
                    run_details_response = requests.get(f"{BACKEND_URL}/experiments/{selected_exp_id}/runs/{run_id}/")
                    if run_details_response.status_code == 200:
                        run_data = run_details_response.json()["run"]
                        st.write(run_data)

                        # Plot metrics if available
                        if run_data["metrics"]:
                            metrics_df = pd.DataFrame(list(run_data["metrics"].items()), columns=["Metric", "Value"])
                            st.bar_chart(metrics_df.set_index("Metric"))
                    else:
                        st.error(f"Error: {run_details_response.json()['detail']}")

        else:
            st.info("No experiments found.")
    else:
        st.error(f"Error: {exp_response.json()['detail']}")

elif choice == "Ask Assistant":
    st.subheader("Ask the Assistant")

    st.markdown("**Examples of questions you can ask:**")
    st.markdown("""
        - How can I improve the accuracy of my model?
        - Why did run `run_id` perform worse than run `run_id`?
        - What are the most significant parameters affecting my model's performance?
    """)

    user_question = st.text_area("Enter your question about your experiments or runs:")
    if st.button("Get Answer"):
        if user_question:
            payload = {"prompt": user_question}
            response = requests.post(f"{BACKEND_URL}/assistant/query", json=payload)
            if response.status_code == 200:
                assistant_response = response.json()["response"]
                st.markdown("**Assistant:**")
                st.write(assistant_response)
            else:
                st.error(f"Error: {response.json()['detail']}")
        else:
            st.error("Please enter a question.")
