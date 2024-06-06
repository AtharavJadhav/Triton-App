import subprocess
import os
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model state and metrics
model_names = set()  # Using a set to ensure unique model names
triton_server_process = None

# Metrics dictionary
metrics = {}

class DeployModelsRequest(BaseModel):
    models: List[str] = Field(..., example=["mnist_model_onnx", "mnist_model_openvino"])

class InferenceRequest(BaseModel):
    model_name: str
    correct: bool = None

@app.post("/deploy/")
async def deploy_models(request_body: DeployModelsRequest):
    models = request_body.models
    global triton_server_process, model_names
    if triton_server_process:
        triton_server_process.kill()

    required_models = {"mnist_model_onnx", "mnist_model_openvino", "mnist_model_pytorch", "bert_model_onnx"}
    if not all(model in required_models for model in models):
        return JSONResponse(status_code=422, content={"detail": "Invalid model names. Use the exact model names."})

    docker_run_command = [
        'docker', 'run', '--shm-size=256m', '--rm',
        '-p8000:8000', '-p8001:8001', '-p8002:8002',
        '-e', 'TRITON_SERVER_CPU_ONLY=1',
        '-v', f'{os.getcwd()}:/workspace/',
        '-v', f'{os.getcwd()}/model_repository:/models',
        'nvcr.io/nvidia/tritonserver:24.04-py3',
        'tritonserver', '--model-repository=/models',
        '--model-control-mode=explicit'
    ]

    for model in models:
        docker_run_command.extend(['--load-model', model])

    try:
        triton_server_process = subprocess.Popen(' '.join(docker_run_command), shell=True)
        model_names = set(models)  # Clear and set the new models
        # Initialize metrics for the deployed models
        for model in models:
            metrics[model] = {
                "accuracy": 0,
                "latency": 0,
                "throughput": 0,
                "request_count": 0,
                "correct_count": 0,
                "total_latency": 0,
            }
        return {"message": "Models deployed successfully"}
    except Exception as e:
        logger.error(f"Error deploying models: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Failed to deploy models"})

@app.post("/inference/")
async def inference(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in metrics:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Create the directory for storing uploaded files if it doesn't exist
        input_dir = "inference_inputs"
        os.makedirs(input_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(input_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Map model names to their corresponding scripts
        model_to_script = {
            "mnist_model_onnx": "client_mnist_onnx.py",
            "mnist_model_openvino": "client_mnist_openvino.py",
            "mnist_model_pytorch": "client_mnist_pytorch.py",
            "bert_model_onnx": "client_bert_onnx.py",
        }
        script_name = model_to_script.get(model_name)
        if not script_name:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Run the model-specific inference script
        start_time = time.time()
        script_path = f"clientfiles/{script_name}"
        subprocess.run(["python3", script_path, file_path])
        end_time = time.time()
        latency = end_time - start_time

        metrics[model_name]["request_count"] += 1
        metrics[model_name]["total_latency"] += latency
        metrics[model_name]["latency"] = metrics[model_name]["total_latency"] / metrics[model_name]["request_count"]
        metrics[model_name]["throughput"] = metrics[model_name]["request_count"] / (metrics[model_name]["latency"] or 1)

        return {"message": "Inference request processed"}

    finally:
        # Remove the temporary file
        os.remove(file_path)


@app.get("/results/")
async def get_results():
    results_file = "inference_results/results.txt"
    if os.path.exists(results_file):
        with open(results_file, "r") as file:
            result = file.read()
        # Clear the file after reading
        open(results_file, "w").close()
        return {"result": result}
    else:
        raise HTTPException(status_code=404, detail="Results not found")

@app.post("/feedback/")
async def submit_feedback(request_body: InferenceRequest):
    model_name = request_body.model_name
    correct = request_body.correct

    if model_name not in metrics:
        raise HTTPException(status_code=404, detail="Model not found")

    if correct is not None:
        if correct:
            metrics[model_name]["correct_count"] += 1
        metrics[model_name]["accuracy"] = metrics[model_name]["correct_count"] / metrics[model_name]["request_count"]

    return {"message": "Feedback submitted"}

@app.get("/metrics/")
async def get_metrics():
    return metrics
