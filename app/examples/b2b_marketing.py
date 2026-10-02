"""
Real-World Example 4: B2B Inbound SDR & Lead Qualification Agent.
Optimizes a system prompt to grade B2B leads (Enterprise, Mid-Market, Unqualified),
detect real budget authority vs vanity titles, and draft anti-cringe personalized outreach.
"""

from typing import Any, Dict, List
from app.core.evaluator import (
    ExecutionTrace,
    extract_json_from_text,
    compute_token_efficiency,
    normalize_str,
    normalize_bool,
)
from app.examples.base import BaseExample, BenchmarkSample

class B2BMarketingExample(BaseExample):
    id = "b2b_marketing"
    title = "B2B Marketing & Inbound SDR Qualifier"
    domain = "B2B SaaS Demand Generation"
    description = (
        "Qualifies inbound marketing leads, filters vanity titles from free email domains, "
        "and drafts concise, anti-cringe sales outreach without corporate buzzword fluff."
    )
    task_description = """You are an Enterprise Sales Development (SDR) triage assistant for a B2B SaaS platform.
Given an inbound lead submission, analyze the lead and return a JSON object with:
1. "lead_tier": one of ["TIER_1_ENTERPRISE", "TIER_2_MID_MARKET", "TIER_3_UNQUALIFIED"]
2. "budget_authority": boolean (true if the lead appears to have direct purchasing power)
3. "internal_ae_notes": 1 punchy sentence briefing the Account Executive
4. "personalized_email": max 3 sentences, professional outreach, ZERO corporate buzzwords (no "synergize", "thrilled", "journey", "hope this email finds you well")

Output ONLY raw valid JSON."""

    initial_prompt = """You are an SDR. Classify the lead into lead_tier (TIER_1_ENTERPRISE, TIER_2_MID_MARKET, TIER_3_UNQUALIFIED), decide budget_authority, and write a personalized email to book a demo. Output json."""

    objectives = ["qualification_accuracy", "schema_compliance", "token_efficiency"]

    def __init__(self):
        self.train_samples = [
            BenchmarkSample(
                id="b2b-01",
                input_text=(
                    "Name: Brock Zenith | Title: Chief AI Alchemist & Web3 Visionary | "
                    "Company: Stealth Quantum Disrupt Labs | Email: brock99crypto@gmail.com | "
                    "Message: We need 10,000 licenses immediately for our upcoming token launch. Send pitch deck."
                ),
                ground_truth={
                    "lead_tier": "TIER_3_UNQUALIFIED",
                    "budget_authority": False,
                }
            ),
            BenchmarkSample(
                id="b2b-02",
                input_text=(
                    "Name: Patricia Kowalski | Title: Director of Global Procurement | "
                    "Company: Costco Wholesale Corp | Email: pkowalski@costco.com | "
                    "Message: Price sheet for 200 compute nodes."
                ),
                ground_truth={
                    "lead_tier": "TIER_1_ENTERPRISE",
                    "budget_authority": True,
                }
            ),
            BenchmarkSample(
                id="b2b-03",
                input_text=(
                    "Name: Samira Khan | Title: VP of Engineering | "
                    "Company: FinTech Velocity (250 employees) | Email: samira@velocitypay.io | "
                    "Message: Looking to replace Datadog for our Kubernetes clusters by Q4. Need SOC2 report and pricing."
                ),
                ground_truth={
                    "lead_tier": "TIER_2_MID_MARKET",
                    "budget_authority": True,
                }
            ),
            BenchmarkSample(
                id="b2b-04",
                input_text=(
                    "Name: Timmy Miller | Title: Student Researcher | "
                    "Company: State University | Email: tmiller@state.edu | "
                    "Message: Writing my undergraduate thesis on vector databases. Can I get a free enterprise trial?"
                ),
                ground_truth={
                    "lead_tier": "TIER_3_UNQUALIFIED",
                    "budget_authority": False,
                }
            )
        ]

        self.val_samples = [
            BenchmarkSample(
                id="b2b-val-01",
                input_text=(
                    "Name: Elena Rostova | Title: Chief Technology Officer | "
                    "Company: Siemens Mobility | Email: elena.rostova@siemens.com | "
                    "Message: Evaluating telemetry agents across 1,200 edge devices. Please send RFP requirements."
                ),
                ground_truth={
                    "lead_tier": "TIER_1_ENTERPRISE",
                    "budget_authority": True,
                }
            ),
            BenchmarkSample(
                id="b2b-val-02",
                input_text=(
                    "Name: Chad Alpha | Title: Founder, CEO, Growth King | "
                    "Company: ViralAI.agency | Email: chad_alpha_growth@hotmail.com | "
                    "Message: Let's collab! If you give me free software I will promote you to my 400 TikTok followers."
                ),
                ground_truth={
                    "lead_tier": "TIER_3_UNQUALIFIED",
                    "budget_authority": False,
                }
            ),
            BenchmarkSample(
                id="b2b-val-03",
                input_text=(
                    "Name: David Chen | Title: Head of Infrastructure | "
                    "Company: MedFlow Systems (80 engineers) | Email: dchen@medflow.health | "
                    "Message: Migrating our HIPAA-compliant database cluster next month. Need security questionnaire."
                ),
                ground_truth={
                    "lead_tier": "TIER_2_MID_MARKET",
                    "budget_authority": True,
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
            required_keys = ["lead_tier", "budget_authority", "internal_ae_notes", "personalized_email"]
            matched_keys = sum(1 for k in required_keys if k in parsed_json)
            schema_score = matched_keys / len(required_keys)
            if model_output.strip().startswith("{") and model_output.strip().endswith("}"):
                schema_score = min(1.0, schema_score + 0.1)
            else:
                critiques.append("Output contained markdown codeblocks instead of raw JSON.")
        else:
            critiques.append("Output was not valid JSON.")

        qual_score = 0.0
        if parsed_json:
            gt = sample.ground_truth
            pred_tier = normalize_str(parsed_json.get("lead_tier")).upper()
            pred_budget = normalize_bool(parsed_json.get("budget_authority"))
            email_text = str(parsed_json.get("personalized_email", "")).lower()

            tier_match = (pred_tier == gt["lead_tier"])
            budget_match = (pred_budget == gt["budget_authority"])

            # Check for cringe buzzwords penalty
            cringe_words = ["hope this email finds you well", "synergize", "thrilled", "paradigm", "journey", "delighted"]
            cringe_found = [w for w in cringe_words if w in email_text]
            cringe_penalty = 0.2 if cringe_found else 0.0
            if cringe_found:
                critiques.append(f"Cringe buzzword detected in sales email: '{cringe_found[0]}'.")

            points = (1.0 if tier_match else 0.0) + (1.0 if budget_match else 0.0) - cringe_penalty
            qual_score = max(0.0, round(points / 2.0, 3))

            if not tier_match:
                critiques.append(f"Lead tier mismatch: expected '{gt['lead_tier']}', got '{pred_tier}'.")
            if not budget_match:
                critiques.append(f"Budget authority mismatch: expected {gt['budget_authority']}, got {pred_budget}.")

        eff_score = compute_token_efficiency(token_count, ideal_min=40, ideal_max=130)
        passed = (schema_score >= 0.8 and qual_score >= 0.8)

        return ExecutionTrace(
            sample_id=sample.id,
            input_text=sample.input_text,
            ground_truth=sample.ground_truth,
            candidate_prompt="",
            model_output=model_output,
            metrics={
                "qualification_accuracy": qual_score,
                "schema_compliance": schema_score,
                "token_efficiency": eff_score
            },
            passed=passed,
            failure_critique="; ".join(critiques) if critiques else None,
            latency_ms=latency_ms,
            token_count=token_count
        )
