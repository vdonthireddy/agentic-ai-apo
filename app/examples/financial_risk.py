"""
Real-World Example 2: Financial & Commercial Contract Risk Analyzer.
Optimizes a system prompt to scrutinize clauses from SaaS Master Service Agreements (MSAs),
NDAs, and vendor contracts to detect uncapped liability, non-standard indemnity, and termination terms.
"""

from typing import Any, Dict, List
from app.core.evaluator import ExecutionTrace, extract_json_from_text, compute_token_efficiency, normalize_str, normalize_bool
from app.examples.base import BaseExample, BenchmarkSample

class FinancialRiskExample(BaseExample):
    id = "financial_risk"
    title = "Commercial Contract Risk & Liability Extractor"
    domain = "LegalTech & Enterprise Procurement"
    description = (
        "Audits contractual clauses to flag liability caps, non-standard indemnification, "
        "and unilateral termination risks, outputting structured risk ratings."
    )
    task_description = """You are a specialized legal AI assistant for contract risk review.
Analyze the provided contract clause and output a JSON object with:
1. "risk_level": one of ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
2. "uncapped_liability": boolean (true if liability is unlimited or lacks standard aggregate cap)
3. "key_risk_factor": concise phrase identifying the primary exposure (e.g., "consequential damages carve-in", "unilateral convenience termination")
4. "clause_classification": one of ["indemnity", "limitation_of_liability", "termination", "confidentiality", "warranty"]

Output ONLY raw valid JSON."""

    initial_prompt = """Review the contract clause. Identify if it has high risk and return json with risk_level and uncapped_liability."""

    objectives = ["risk_detection_f1", "schema_compliance", "token_efficiency"]

    def __init__(self):
        self.train_samples = [
            BenchmarkSample(
                id="leg-01",
                input_text=(
                    "In no event shall either party's aggregate liability arising out of or related to this Agreement "
                    "exceed the total amount paid by Customer hereunder in the twelve (12) months preceding the incident."
                ),
                ground_truth={
                    "risk_level": "LOW",
                    "uncapped_liability": False,
                    "clause_classification": "limitation_of_liability"
                }
            ),
            BenchmarkSample(
                id="leg-02",
                input_text=(
                    "Vendor shall indemnify, defend, and hold harmless Customer against any and all third-party claims, "
                    "damages, losses, and legal fees without limitation or monetary cap arising from any breach of confidentiality or IP infringement."
                ),
                ground_truth={
                    "risk_level": "HIGH",
                    "uncapped_liability": True,
                    "clause_classification": "indemnity"
                }
            ),
            BenchmarkSample(
                id="leg-03",
                input_text=(
                    "Customer may terminate this Agreement immediately without cause upon written notice, and Vendor shall refund "
                    "all pre-paid unused fees within five (5) business days."
                ),
                ground_truth={
                    "risk_level": "MEDIUM",
                    "uncapped_liability": False,
                    "clause_classification": "termination"
                }
            ),
            BenchmarkSample(
                id="leg-04",
                input_text=(
                    "Either party disclaims all consequential, punitive, special, or indirect damages, except for gross negligence, willful misconduct, or data privacy breaches."
                ),
                ground_truth={
                    "risk_level": "MEDIUM",
                    "uncapped_liability": False,
                    "clause_classification": "limitation_of_liability"
                }
            )
        ]

        self.val_samples = [
            BenchmarkSample(
                id="leg-val-01",
                input_text=(
                    "Vendor agrees to be solely responsible for all direct and indirect commercial losses suffered by Customer, "
                    "waiving any statutory damage limits or statutory caps."
                ),
                ground_truth={
                    "risk_level": "CRITICAL",
                    "uncapped_liability": True,
                    "clause_classification": "limitation_of_liability"
                }
            ),
            BenchmarkSample(
                id="leg-val-02",
                input_text=(
                    "Confidential information shall be kept confidential for a period of three (3) years from disclosure, standard exclusions apply."
                ),
                ground_truth={
                    "risk_level": "LOW",
                    "uncapped_liability": False,
                    "clause_classification": "confidentiality"
                }
            ),
            BenchmarkSample(
                id="leg-val-03",
                input_text=(
                    "Customer shall indemnify Vendor for any claims arising from Customer's unauthorized modification of the Cloud Platform."
                ),
                ground_truth={
                    "risk_level": "LOW",
                    "uncapped_liability": False,
                    "clause_classification": "indemnity"
                }
            )
        ]

    def evaluate_output(
        self,
        sample: BenchmarkSample,
        model_output: str,
        latency_ms: float,
        token_count: int
    ) -> ExecutionTrace:
        parsed_json = extract_json_from_text(model_output)
        critiques = []

        schema_score = 0.0
        if parsed_json is not None:
            required_keys = ["risk_level", "uncapped_liability", "key_risk_factor", "clause_classification"]
            matched_keys = sum(1 for k in required_keys if k in parsed_json)
            schema_score = matched_keys / len(required_keys)
            if model_output.strip().startswith("{") and model_output.strip().endswith("}"):
                schema_score = min(1.0, schema_score + 0.1)
            else:
                critiques.append("Formatting contained non-JSON markdown.")
        else:
            critiques.append("Output could not be parsed as valid JSON.")

        risk_score = 0.0
        if parsed_json:
            gt = sample.ground_truth
            pred_risk = normalize_str(parsed_json.get("risk_level")).upper()
            pred_uncapped = normalize_bool(parsed_json.get("uncapped_liability"))
            pred_class = normalize_str(parsed_json.get("clause_classification"))

            risk_match = (pred_risk == gt["risk_level"])
            uncapped_match = (pred_uncapped == gt["uncapped_liability"])
            class_match = (pred_class == gt["clause_classification"])

            risk_score = round(sum([risk_match, uncapped_match, class_match]) / 3.0, 3)

            if not risk_match:
                critiques.append(f"Risk level mismatch: expected '{gt['risk_level']}', got '{pred_risk}'.")
            if not uncapped_match:
                critiques.append(f"Liability cap mismatch: expected uncapped={gt['uncapped_liability']}, got {pred_uncapped}.")
            if not class_match:
                critiques.append(f"Clause type mismatch: expected '{gt['clause_classification']}', got '{pred_class}'.")

        eff_score = compute_token_efficiency(token_count, ideal_min=30, ideal_max=90)
        passed = (schema_score >= 0.8 and risk_score >= 0.66)

        return ExecutionTrace(
            sample_id=sample.id,
            input_text=sample.input_text,
            ground_truth=sample.ground_truth,
            candidate_prompt="",
            model_output=model_output,
            metrics={
                "risk_detection_f1": risk_score,
                "schema_compliance": schema_score,
                "token_efficiency": eff_score
            },
            passed=passed,
            failure_critique="; ".join(critiques) if critiques else None,
            latency_ms=latency_ms,
            token_count=token_count
        )
