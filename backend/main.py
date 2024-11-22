# backend/main.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import mlflow
import os
from dotenv import load_dotenv
import openai
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize FastAPI app
app = FastAPI(title="TrackMate Backend")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None

class RunCreate(BaseModel):
    run_name: str | None = None

class Param(BaseModel):
    key: str
    value: str

class Metric(BaseModel):
    key: str
    value: float

class AssistantQuery(BaseModel):
    prompt: str

# Utility function to handle blocking MLflow calls asynchronously
def blocking_mlflow_call(func, *args, **kwargs):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, lambda: func(*args, **kwargs))

# Dependency to get OpenAI API key
def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
    return api_key

# Experiment endpoints
@app.post("/experiments/")
async def create_experiment(exp: ExperimentCreate):
    try:
        experiment_id = await blocking_mlflow_call(mlflow.create_experiment, exp.name, exp.description)
        return {"experiment_id": experiment_id, "name": exp.name}
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments/")
async def list_experiments():
    try:
        experiments = await blocking_mlflow_call(mlflow.list_experiments)
        exp_list = [
            {"experiment_id": exp.experiment_id, "name": exp.name, "artifact_location": exp.artifact_location}
            for exp in experiments
        ]
        return {"experiments": exp_list}
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Run endpoints
@app.post("/experiments/{experiment_id}/runs/")
async def start_run(experiment_id: str, run: RunCreate):
    try:
        with mlflow.start_run(experiment_id=experiment_id, run_name=run.run_name) as active_run:
            return {"run_id": active_run.info.run_id, "status": active_run.info.status}
    except Exception as e:
        logger.error(f"Error starting run: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/runs/{run_id}/params/")
async def log_param(run_id: str, param: Param):
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param(param.key, param.value)
        return {"message": f"Parameter '{param.key}' logged successfully."}
    except Exception as e:
        logger.error(f"Error logging parameter: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/runs/{run_id}/metrics/")
async def log_metric(run_id: str, metric: Metric):
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(metric.key, metric.value)
        return {"message": f"Metric '{metric.key}' logged successfully."}
    except Exception as e:
        logger.error(f"Error logging metric: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments/{experiment_id}/runs/{run_id}/")
async def get_run(experiment_id: str, run_id: str):
    try:
        run = await blocking_mlflow_call(mlflow.get_run, run_id)
        run_data = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "parameters": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
        }
        return {"run": run_data}
    except Exception as e:
        logger.error(f"Error getting run: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Assistant query endpoint
@app.post("/assistant/query")
async def assistant_query(query: AssistantQuery, api_key: str = Depends(get_openai_api_key)):
    try:
        # Fetch all experiments and runs
        experiments = await blocking_mlflow_call(mlflow.list_experiments)
        context = "Here is the current experiment and run data:\n"
        for exp in experiments:
            context += f"Experiment: {exp.name} (ID: {exp.experiment_id})\n"
            experiment_runs = await blocking_mlflow_call(mlflow.search_runs, exp.experiment_id)
            for run in experiment_runs.itertuples():
                context += f"  Run ID: {run.run_id}, Status: {run.status}, Metrics: {run.metrics}, Parameters: {run.params}\n"

        # Combine user prompt with context
        full_prompt = f"{context}\nUser Query: {query.prompt}\nAnswer:"

        # Send to OpenAI
        openai.api_key = api_key
        response = await blocking_mlflow_call(openai.Completion.create, engine="text-davinci-003", prompt=full_prompt, max_tokens=150, temperature=0.7)
        answer = response.choices[0].text.strip()
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error processing assistant query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
