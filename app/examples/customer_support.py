"""
Real-World Example 1: Enterprise Customer Support Triage & Intent Routing.
Optimizes a system prompt to categorize support tickets, assign urgency,
detect customer sentiment, and extract structured JSON compliant with backend schemas.
"""

from typing import Any, Dict, List
from app.core.evaluator import ExecutionTrace, extract_json_from_text, compute_token_efficiency, normalize_str
from app.examples.base import BaseExample, BenchmarkSample

class CustomerSupportExample(BaseExample):
    id = "customer_support"
    title = "Customer Support Intent & Urgency Routing"
    domain = "SaaS & E-commerce Operations"
    description = (
        "Triages chaotic inbound customer tickets into strict JSON schema containing category, "
        "urgency level, customer sentiment, and recommended next action."
    )
    task_description = """You are an automated customer support triage system.
Given a customer ticket, analyze the message and output a JSON object with:
1. "category": one of ["billing", "technical", "account", "cancellation", "general"]
2. "urgency": one of ["low", "medium", "high", "critical"]
3. "sentiment": one of ["positive", "neutral", "negative"]
4. "suggested_action": brief one-sentence instruction for the support agent

You must output ONLY valid raw JSON with no extra conversational text or markdown codeblocks."""

    initial_prompt = """You are customer support AI. Look at the customer ticket, decide the category, urgency, and sentiment. Return json."""

    objectives = ["accuracy", "schema_compliance", "token_efficiency"]

    def __init__(self):
        self.train_samples = [
            BenchmarkSample(
                id="cs-01",
                input_text="I was double billed for my subscription this morning ($199 charged twice)! Fix this immediately or I am reporting fraud to my bank.",
                ground_truth={
                    "category": "billing",
                    "urgency": "critical",
                    "sentiment": "negative",
                }
            ),
            BenchmarkSample(
                id="cs-02",
                input_text="Hey team, loving the product so far! Just wanted to know if you support dark mode on iOS yet?",
                ground_truth={
                    "category": "general",
                    "urgency": "low",
                    "sentiment": "positive",
                }
            ),
            BenchmarkSample(
                id="cs-03",
                input_text="Our production webhook is returning 504 Gateway Timeout errors intermittently since 2 PM UTC. We are losing transactions.",
                ground_truth={
                    "category": "technical",
                    "urgency": "critical",
                    "sentiment": "negative",
                }
            ),
            BenchmarkSample(
                id="cs-04",
                input_text="I want to cancel my account. We are migrating to another provider at the end of this month.",
                ground_truth={
                    "category": "cancellation",
                    "urgency": "high",
                    "sentiment": "negative",
                }
            ),
            BenchmarkSample(
                id="cs-05",
                input_text="Can someone help me update the email address linked to our team admin profile?",
                ground_truth={
                    "category": "account",
                    "urgency": "medium",
                    "sentiment": "neutral",
                }
            ),
            BenchmarkSample(
                id="cs-06",
                input_text="My invoice for September shows the wrong VAT number. Can you please re-issue it under VAT ID GB123456789?",
                ground_truth={
                    "category": "billing",
                    "urgency": "medium",
                    "sentiment": "neutral",
                }
            )
        ]

        self.val_samples = [
            BenchmarkSample(
                id="cs-val-01",
                input_text="This software is completely broken. Nothing loads and I have a client presentation in 10 minutes. CANCEL MY SUBSCRIPTION IMMEDIATELY AND REFUND ME.",
                ground_truth={
                    "category": "cancellation",
                    "urgency": "critical",
                    "sentiment": "negative",
                }
            ),
            BenchmarkSample(
                id="cs-val-02",
                input_text="Where can I download our SOC2 compliance report for our annual audit?",
                ground_truth={
                    "category": "general",
                    "urgency": "low",
                    "sentiment": "neutral",
                }
            ),
            BenchmarkSample(
                id="cs-val-03",
                input_text="Getting HTTP 401 Unauthorized on our API keys even though the dashboard says the token is active.",
                ground_truth={
                    "category": "technical",
                    "urgency": "high",
                    "sentiment": "negative",
                }
            ),
            BenchmarkSample(
                id="cs-val-04",
                input_text="We have a 20-person team and need to transfer ownership from Alice to Bob since Alice left the company.",
                ground_truth={
                    "category": "account",
                    "urgency": "medium",
                    "sentiment": "neutral",
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

        # 1. Schema Compliance Metric
        schema_score = 0.0
        if parsed_json is not None:
            required_keys = ["category", "urgency", "sentiment", "suggested_action"]
            matched_keys = sum(1 for k in required_keys if k in parsed_json)
            schema_score = matched_keys / len(required_keys)
            # Bonus check: raw JSON vs wrapped in markdown
            if model_output.strip().startswith("{") and model_output.strip().endswith("}"):
                schema_score = min(1.0, schema_score + 0.1)
            else:
                critiques.append("Output was wrapped in markdown/chat commentary instead of raw JSON.")
        else:
            critiques.append("Failed to parse valid JSON from model output.")

        # 2. Accuracy Metric
        accuracy_score = 0.0
        if parsed_json:
            gt = sample.ground_truth
            cat_match = normalize_str(parsed_json.get("category")) == gt["category"]
            urg_match = normalize_str(parsed_json.get("urgency")) == gt["urgency"]
            sent_match = normalize_str(parsed_json.get("sentiment")) == gt["sentiment"]

            acc_points = sum([cat_match, urg_match, sent_match])
            accuracy_score = round(acc_points / 3.0, 3)

            if not cat_match:
                critiques.append(f"Category mismatch: expected '{gt['category']}', got '{parsed_json.get('category')}'.")
            if not urg_match:
                critiques.append(f"Urgency mismatch: expected '{gt['urgency']}', got '{parsed_json.get('urgency')}'.")
            if not sent_match:
                critiques.append(f"Sentiment mismatch: expected '{gt['sentiment']}', got '{parsed_json.get('sentiment')}'.")

        # 3. Token Efficiency Metric
        eff_score = compute_token_efficiency(token_count, ideal_min=30, ideal_max=90)

        passed = (schema_score >= 0.8 and accuracy_score >= 0.66)
        failure_critique = "; ".join(critiques) if critiques else None

        return ExecutionTrace(
            sample_id=sample.id,
            input_text=sample.input_text,
            ground_truth=sample.ground_truth,
            candidate_prompt="",
            model_output=model_output,
            metrics={
                "accuracy": accuracy_score,
                "schema_compliance": schema_score,
                "token_efficiency": eff_score
            },
            passed=passed,
            failure_critique=failure_critique,
            latency_ms=latency_ms,
            token_count=token_count
        )
