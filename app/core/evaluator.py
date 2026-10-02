"""
Evaluation Engine and Trace Collector for GEPA.
Calculates multi-objective scores and records execution trajectories
(including failure diagnostics and reasoning traces) for the Reflector model.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Dict, List, Optional

def normalize_str(value: Any, fallback: str = "") -> str:
    """
    Safely coerce a JSON field to a lowercase stripped string.
    LLMs sometimes return lists (['billing']), ints (2), or nested dicts
    instead of a plain string. This prevents AttributeError crashes on .lower().
    Examples:
        normalize_str(["billing"])   -> "billing"
        normalize_str("Billing ")    -> "billing"
        normalize_str(None)          -> ""
        normalize_str(2)             -> "2"
    """
    if value is None:
        return fallback
    if isinstance(value, list):
        # Take first element if list, e.g. ["billing"] -> "billing"
        value = value[0] if value else fallback
    return str(value).lower().strip()

def normalize_bool(value: Any) -> bool:
    """
    Safely coerce a JSON field to bool.
    Handles: True/False, 'true'/'false' strings, 1/0, 'yes'/'no', [True].
    """
    if isinstance(value, list):
        value = value[0] if value else False
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower().strip() in ("true", "1", "yes")
    return False

@dataclass
class ExecutionTrace:
    sample_id: str
    input_text: str
    ground_truth: Any
    candidate_prompt: str
    model_output: str
    metrics: Dict[str, float]
    passed: bool
    failure_critique: Optional[str] = None
    latency_ms: float = 0.0
    token_count: int = 0

@dataclass
class PromptCandidate:
    candidate_id: str
    generation: int
    prompt_text: str
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    # Aggregated objective scores (e.g. {"accuracy": 0.85, "schema_validity": 1.0, "token_efficiency": 0.9})
    scores: Dict[str, float] = field(default_factory=dict)
    average_latency_ms: float = 0.0
    average_token_count: float = 0.0
    traces: List[ExecutionTrace] = field(default_factory=list)
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    is_frontier: bool = False

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extractor that handles markdown blocks, preambles, and raw text."""
    if not text:
        return None
    # 1. Try direct json parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # 2. Extract from ```json ... ``` codeblock
    codeblock_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if codeblock_match:
        try:
            return json.loads(codeblock_match.group(1))
        except Exception:
            pass

    # 3. Extract the first outermost {...}
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    return None

def compute_token_efficiency(token_count: int, ideal_min: int = 40, ideal_max: int = 150) -> float:
    """
    Score token efficiency between 0.0 and 1.0.
    Penalizes extreme verbosity or unnecessary fluff.
    """
    if token_count <= 0:
        return 0.0
    if ideal_min <= token_count <= ideal_max:
        return 1.0
    if token_count < ideal_min:
        return max(0.2, token_count / ideal_min)
    # If token_count > ideal_max, decay smoothly
    excess = token_count - ideal_max
    decay = max(0.1, 1.0 - (excess / 300.0))
    return round(decay, 3)
