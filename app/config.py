"""
Configuration settings for GEPA Automatic Prompt Optimizer.
Supports both local environment and containerized Docker environment.
"""

import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Web server settings
    PORT: int = int(os.getenv("PORT", "18435"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Ollama settings
    # When inside Docker container on macOS/Linux, host.docker.internal resolves to the host machine
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_TASK_MODEL: str = os.getenv("DEFAULT_TASK_MODEL", "llama3.2:latest")
    DEFAULT_REFLECTOR_MODEL: str = os.getenv("DEFAULT_REFLECTOR_MODEL", "llama3.2:latest")

    # Optimizer defaults
    DEFAULT_POPULATION_SIZE: int = int(os.getenv("POPULATION_SIZE", "5"))
    DEFAULT_GENERATIONS: int = int(os.getenv("GENERATIONS", "4"))
    DEFAULT_MUTATION_RATE: float = float(os.getenv("MUTATION_RATE", "0.7"))
    DEFAULT_CROSSOVER_RATE: float = float(os.getenv("CROSSOVER_RATE", "0.3"))
    MAX_EVAL_SAMPLES_PER_GEN: int = int(os.getenv("MAX_EVAL_SAMPLES", "8"))

settings = Settings()
