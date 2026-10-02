"""
Registry of benchmark examples for GEPA Automatic Prompt Optimization.
"""

from typing import Dict, List, Optional
from app.examples.base import BaseExample
from app.examples.customer_support import CustomerSupportExample
from app.examples.financial_risk import FinancialRiskExample
from app.examples.medical_triage import MedicalTriageExample

EXAMPLES_REGISTRY: Dict[str, BaseExample] = {
    "customer_support": CustomerSupportExample(),
    "financial_risk": FinancialRiskExample(),
    "medical_triage": MedicalTriageExample()
}

def get_example(example_id: str) -> Optional[BaseExample]:
    return EXAMPLES_REGISTRY.get(example_id)

def list_examples() -> List[Dict[str, str]]:
    return [
        {
            "id": ex.id,
            "title": ex.title,
            "domain": ex.domain,
            "description": ex.description,
            "objectives": ex.objectives,
            "train_count": len(ex.train_samples),
            "val_count": len(ex.val_samples),
            "initial_prompt": ex.initial_prompt
        }
        for ex in EXAMPLES_REGISTRY.values()
    ]
