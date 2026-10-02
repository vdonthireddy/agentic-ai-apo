"""
Base class and data contracts for Real-World APO benchmark examples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from app.core.evaluator import ExecutionTrace

@dataclass
class BenchmarkSample:
    id: str
    input_text: str
    ground_truth: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class BaseExample(ABC):
    id: str
    title: str
    domain: str
    description: str
    task_description: str
    initial_prompt: str
    objectives: List[str]
    train_samples: List[BenchmarkSample]
    val_samples: List[BenchmarkSample]

    @abstractmethod
    def evaluate_output(
        self,
        sample: BenchmarkSample,
        model_output: str,
        latency_ms: float,
        token_count: int
    ) -> ExecutionTrace:
        """
        Evaluate a single model generation against ground truth.
        Returns an ExecutionTrace with metric scores and failure critique.
        """
        pass
