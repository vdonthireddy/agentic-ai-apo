"""
FastAPI Web Application and WebSocket Server for GEPA Prompt Optimizer.
Binds to port 18435. Provides REST APIs, real-time WebSocket telemetry,
playground testing, and interactive dashboard UI.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

from app.config import settings
from app.core.ollama_client import OllamaClient
from app.core.gepa_optimizer import GEPAOptimizer
from app.examples import EXAMPLES_REGISTRY, list_examples, get_example
from app.examples.base import BaseExample, BenchmarkSample
from app.core.evaluator import ExecutionTrace, extract_json_from_text, compute_token_efficiency

# Configure internal logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GEPA.Server")

app = FastAPI(
    title="GEPA Automatic Prompt Optimizer",
    description="Genetic-Pareto Reflective Prompt Evolution for Local LLMs (Ollama)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and template mounting
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Global State & WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected to WebSocket. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected from WebSocket. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

active_optimizer: Optional[GEPAOptimizer] = None
active_optimizer_task: Optional[asyncio.Task] = None
last_run_result: Optional[Dict[str, Any]] = None
recent_logs: List[Dict[str, Any]] = []

async def optimizer_event_broadcaster(event: Dict[str, Any]):
    """Broadcasts events from GEPA optimizer to all connected UI clients."""
    if event.get("type") == "LOG":
        recent_logs.append(event["data"])
        if len(recent_logs) > 300:
            recent_logs.pop(0)
    await manager.broadcast(event)

# Custom Example wrapper for user-provided prompts
class CustomExample(BaseExample):
    def __init__(self, title: str, task_desc: str, initial_prompt: str, samples_data: List[Dict[str, Any]]):
        self.id = "custom"
        self.title = title or "Custom Optimization Task"
        self.domain = "User Defined"
        self.description = "User-supplied task prompt and validation examples."
        self.task_description = task_desc
        self.initial_prompt = initial_prompt
        self.objectives = ["accuracy", "token_efficiency"]
        self.train_samples = [
            BenchmarkSample(
                id=f"custom-train-{i+1}",
                input_text=s.get("input", ""),
                ground_truth=s.get("expected", {})
            )
            for i, s in enumerate(samples_data[:6])
        ]
        self.val_samples = [
            BenchmarkSample(
                id=f"custom-val-{i+1}",
                input_text=s.get("input", ""),
                ground_truth=s.get("expected", {})
            )
            for i, s in enumerate(samples_data[6:10])
        ] or self.train_samples[:2]

    def evaluate_output(self, sample: BenchmarkSample, model_output: str, latency_ms: float, token_count: int) -> ExecutionTrace:
        gt = str(sample.ground_truth).strip().lower()
        pred = model_output.strip().lower()
        exact_match = (gt in pred) or (pred in gt)
        acc = 1.0 if exact_match else 0.4
        eff = compute_token_efficiency(token_count, 20, 100)
        return ExecutionTrace(
            sample_id=sample.id,
            input_text=sample.input_text,
            ground_truth=sample.ground_truth,
            candidate_prompt="",
            model_output=model_output,
            metrics={"accuracy": acc, "token_efficiency": eff},
            passed=exact_match,
            failure_critique="Output did not match expected target substring." if not exact_match else None,
            latency_ms=latency_ms,
            token_count=token_count
        )

# Request Models
class OptimizeRequest(BaseModel):
    example_id: str = "customer_support"
    task_model: str = settings.DEFAULT_TASK_MODEL
    reflector_model: str = settings.DEFAULT_REFLECTOR_MODEL
    population_size: int = settings.DEFAULT_POPULATION_SIZE
    generations: int = settings.DEFAULT_GENERATIONS
    mutation_rate: float = settings.DEFAULT_MUTATION_RATE
    crossover_rate: float = settings.DEFAULT_CROSSOVER_RATE
    custom_task_description: Optional[str] = None
    custom_initial_prompt: Optional[str] = None
    custom_samples: Optional[List[Dict[str, Any]]] = None

class PlaygroundTestRequest(BaseModel):
    task_model: str = settings.DEFAULT_TASK_MODEL
    initial_prompt: str
    optimized_prompt: str
    test_input: str

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "port": settings.PORT,
            "default_model": settings.DEFAULT_TASK_MODEL
        }
    )

@app.get("/api/health")
async def health_check():
    client = OllamaClient()
    health_info = await client.check_health()
    return {
        "status": "HEALTHY",
        "port": settings.PORT,
        "ollama": health_info,
        "active_optimization": active_optimizer.is_running if active_optimizer else False
    }

@app.get("/api/examples")
async def get_examples():
    return list_examples()

@app.get("/api/models")
async def get_models():
    client = OllamaClient()
    models = await client.list_models()
    return {"models": models}

@app.get("/api/logs")
async def get_recent_logs():
    return {"logs": recent_logs[-100:]}

@app.get("/api/status")
async def get_status():
    global active_optimizer, last_run_result
    return {
        "is_running": active_optimizer.is_running if active_optimizer else False,
        "last_result": last_run_result
    }

@app.post("/api/stop")
async def stop_optimization():
    global active_optimizer
    if active_optimizer and active_optimizer.is_running:
        active_optimizer.stop()
        return {"status": "STOPPING", "message": "Signal sent to stop optimization after current step."}
    return {"status": "IDLE", "message": "No optimization is currently running."}

@app.post("/api/optimize")
async def start_optimization(req: OptimizeRequest):
    global active_optimizer, active_optimizer_task, last_run_result
    if active_optimizer and active_optimizer.is_running:
        raise HTTPException(status_code=400, detail="An optimization is already running. Please stop it first.")

    # Select example or custom
    if req.example_id == "custom":
        if not req.custom_initial_prompt or not req.custom_task_description:
            raise HTTPException(status_code=400, detail="Custom task requires custom_initial_prompt and custom_task_description.")
        example = CustomExample(
            title="Custom User Task",
            task_desc=req.custom_task_description,
            initial_prompt=req.custom_initial_prompt,
            samples_data=req.custom_samples or []
        )
    else:
        example = get_example(req.example_id)
        if not example:
            raise HTTPException(status_code=404, detail=f"Example '{req.example_id}' not found.")

    client = OllamaClient()
    optimizer = GEPAOptimizer(
        task_model=req.task_model,
        reflector_model=req.reflector_model,
        example=example,
        client=client,
        population_size=req.population_size,
        generations=req.generations,
        mutation_rate=req.mutation_rate,
        crossover_rate=req.crossover_rate,
        event_callback=optimizer_event_broadcaster
    )

    active_optimizer = optimizer

    async def run_wrapper():
        global last_run_result
        try:
            last_run_result = await optimizer.run()
        except Exception as e:
            logger.exception(f"Unhandled error in optimizer run: {e}")
            await manager.broadcast({
                "type": "OPTIMIZATION_ERROR",
                "data": {"error": str(e)}
            })

    active_optimizer_task = asyncio.create_task(run_wrapper())

    return {
        "status": "STARTED",
        "example_id": req.example_id,
        "task_model": req.task_model,
        "reflector_model": req.reflector_model,
        "generations": req.generations,
        "population_size": req.population_size
    }

@app.post("/api/playground/test")
async def playground_test(req: PlaygroundTestRequest):
    """Run side-by-side inference on initial prompt vs optimized prompt."""
    client = OllamaClient()

    # Rollout with Initial Prompt
    init_res = await client.generate(
        model=req.task_model,
        prompt=req.test_input,
        system=req.initial_prompt,
        temperature=0.1
    )

    # Rollout with Optimized Prompt
    opt_res = await client.generate(
        model=req.task_model,
        prompt=req.test_input,
        system=req.optimized_prompt,
        temperature=0.1
    )

    return {
        "initial": {
            "output": init_res["text"],
            "latency_ms": init_res["latency_ms"],
            "tokens": init_res["token_count"],
            "json_parsed": extract_json_from_text(init_res["text"])
        },
        "optimized": {
            "output": opt_res["text"],
            "latency_ms": opt_res["latency_ms"],
            "tokens": opt_res["token_count"],
            "json_parsed": extract_json_from_text(opt_res["text"])
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial welcome and recent logs
        await websocket.send_json({
            "type": "CONNECTED",
            "data": {
                "message": "Connected to GEPA Real-time Optimization Stream",
                "port": settings.PORT,
                "recent_logs": recent_logs[-20:]
            }
        })
        while True:
            data = await websocket.receive_text()
            # Handle client commands if sent via WS
            try:
                msg = json.loads(data)
                if msg.get("action") == "PING":
                    await websocket.send_json({"type": "PONG"})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
        manager.disconnect(websocket)
