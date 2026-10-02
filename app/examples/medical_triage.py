"""
Real-World Example 3: Clinical Emergency Triage & Diagnostic Chain-of-Thought.
Optimizes a system prompt to assign Emergency Severity Index (ESI Levels 1-5),
perform systematic vital-sign scrutiny, and flag critical life threats.
"""

from typing import Any, Dict, List
from app.core.evaluator import ExecutionTrace, extract_json_from_text, compute_token_efficiency, normalize_bool
from app.examples.base import BaseExample, BenchmarkSample

class MedicalTriageExample(BaseExample):
    id = "medical_triage"
    title = "Clinical Emergency Triage & Chain-of-Thought"
    domain = "Healthcare & Emergency Medicine"
    description = (
        "Triages acute emergency room patient presentations into Emergency Severity Index (ESI 1-5), "
        "enforcing chain-of-thought vital sign evaluation to prevent missed life threats."
    )
    task_description = """You are a clinical triage decision support assistant in an Emergency Department.
Evaluate the patient vignette and return a JSON object with:
1. "esi_level": integer between 1 (resuscitation/immediate) and 5 (non-urgent)
2. "is_high_risk": boolean (true if severe danger zone vitals or emergent conditions exist)
3. "clinical_rationale": 1-2 sentence chain-of-thought justification focusing on vitals and symptoms
4. "primary_concern": brief diagnosis or syndromic category (e.g., "acute coronary syndrome", "sepsis", "simple sprain")

Output ONLY raw valid JSON."""

    initial_prompt = """You are an ER triage assistant. Given the patient info, determine esi_level (1 to 5) and if they are high risk. Output json."""

    objectives = ["triage_accuracy", "safety_recall", "schema_compliance", "token_efficiency"]

    def __init__(self):
        self.train_samples = [
            BenchmarkSample(
                id="med-01",
                input_text=(
                    "68yo male presenting with sudden onset crushing substernal chest pressure radiating to left jaw, diaphoresis. "
                    "BP 88/54, HR 122, RR 26, SpO2 91% on room air."
                ),
                ground_truth={
                    "esi_level": 1,
                    "is_high_risk": True,
                    "primary_concern": "acute coronary syndrome / cardiogenic shock"
                }
            ),
            BenchmarkSample(
                id="med-02",
                input_text=(
                    "24yo female with isolated lateral right ankle pain following inversion during soccer 1 hour ago. Able to bear weight with limping. "
                    "BP 118/74, HR 72, RR 14, SpO2 99%, afebrile."
                ),
                ground_truth={
                    "esi_level": 4,
                    "is_high_risk": False,
                    "primary_concern": "ankle sprain / minor musculoskeletal injury"
                }
            ),
            BenchmarkSample(
                id="med-03",
                input_text=(
                    "75yo female from nursing home with altered mental status, fever 39.1C, HR 118, BP 86/50, RR 24. "
                    "Urine cloudy and foul-smelling."
                ),
                ground_truth={
                    "esi_level": 2,
                    "is_high_risk": True,
                    "primary_concern": "urosepsis / severe sepsis"
                }
            ),
            BenchmarkSample(
                id="med-04",
                input_text=(
                    "32yo male requesting suture removal from left forearm laceration repaired 10 days ago. Wound is clean, no erythema or discharge. Vitals normal."
                ),
                ground_truth={
                    "esi_level": 5,
                    "is_high_risk": False,
                    "primary_concern": "routine suture removal"
                }
            )
        ]

        self.val_samples = [
            BenchmarkSample(
                id="med-val-01",
                input_text=(
                    "45yo female with sudden severe 'worst headache of my life', photophobia, neck stiffness. BP 178/104, HR 88, SpO2 98%."
                ),
                ground_truth={
                    "esi_level": 2,
                    "is_high_risk": True,
                    "primary_concern": "subarachnoid hemorrhage"
                }
            ),
            BenchmarkSample(
                id="med-val-02",
                input_text=(
                    "19yo male with mild runny nose and sore throat for 2 days. No fever, normal vitals, swallowing liquids without difficulty."
                ),
                ground_truth={
                    "esi_level": 5,
                    "is_high_risk": False,
                    "primary_concern": "upper respiratory viral infection"
                }
            ),
            BenchmarkSample(
                id="med-val-03",
                input_text=(
                    "55yo diabetic male with right foot ulcer with surrounding redness, warmth, and purulent drainage. Temp 38.6C, HR 105, BP 110/70."
                ),
                ground_truth={
                    "esi_level": 2,
                    "is_high_risk": True,
                    "primary_concern": "diabetic foot infection / systemic cellulitis"
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
            required_keys = ["esi_level", "is_high_risk", "clinical_rationale", "primary_concern"]
            matched_keys = sum(1 for k in required_keys if k in parsed_json)
            schema_score = matched_keys / len(required_keys)
            if model_output.strip().startswith("{") and model_output.strip().endswith("}"):
                schema_score = min(1.0, schema_score + 0.1)
            else:
                critiques.append("Response had markdown formatting.")
        else:
            critiques.append("Could not extract JSON.")

        triage_score = 0.0
        safety_score = 0.0

        if parsed_json:
            gt = sample.ground_truth
            pred_esi = parsed_json.get("esi_level")
            pred_risk = normalize_bool(parsed_json.get("is_high_risk"))

            # Handle esi_level as list (e.g. [2]) or string "2"
            if isinstance(pred_esi, list):
                pred_esi = pred_esi[0] if pred_esi else None

            try:
                pred_esi_int = int(float(str(pred_esi)))
                # Exact or within 1 level (triage tolerance)
                if pred_esi_int == gt["esi_level"]:
                    triage_score = 1.0
                elif abs(pred_esi_int - gt["esi_level"]) == 1:
                    triage_score = 0.6
                else:
                    triage_score = 0.0
                    critiques.append(f"Severe ESI misclassification: expected {gt['esi_level']}, got {pred_esi_int}.")
            except Exception:
                critiques.append(f"Invalid ESI level format: {pred_esi}")

            if pred_risk == gt["is_high_risk"]:
                safety_score = 1.0
            else:
                safety_score = 0.0
                if gt["is_high_risk"] and not pred_risk:
                    critiques.append("CRITICAL SAFETY FAILURE: Under-triaged high-risk patient with danger vitals.")
                else:
                    critiques.append("Over-triaged low-risk case.")

        eff_score = compute_token_efficiency(token_count, ideal_min=40, ideal_max=120)
        passed = (schema_score >= 0.8 and triage_score >= 0.6 and safety_score == 1.0)

        return ExecutionTrace(
            sample_id=sample.id,
            input_text=sample.input_text,
            ground_truth=sample.ground_truth,
            candidate_prompt="",
            model_output=model_output,
            metrics={
                "triage_accuracy": triage_score,
                "safety_recall": safety_score,
                "schema_compliance": schema_score,
                "token_efficiency": eff_score
            },
            passed=passed,
            failure_critique="; ".join(critiques) if critiques else None,
            latency_ms=latency_ms,
            token_count=token_count
        )
