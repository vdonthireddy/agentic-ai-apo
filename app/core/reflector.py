"""
Natural Language Reflection Engine for GEPA.
Inspects full execution traces (reasoning logs, tool calls, failure critiques)
and instructs an LLM to diagnose root causes and propose targeted prompt mutations.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from app.core.evaluator import ExecutionTrace, PromptCandidate
from app.core.ollama_client import OllamaClient

logger = logging.getLogger("GEPA.Reflector")

REFLECTOR_SYSTEM_PROMPT = """You are an expert Prompt Optimization Reflector in the GEPA (Genetic-Pareto) framework.
Your job is to analyze failure traces from an AI system, diagnose why the current prompt failed, and generate a significantly improved version of the prompt.

Guidelines:
1. Conduct a rigorous ROOT CAUSE DIAGNOSIS: Why did the model make mistakes on the failed examples? Was the prompt ambiguous? Did it fail to handle edge cases? Did it fail to enforce strict output formats (e.g. JSON)?
2. Formulate TARGETED RULES: Add explicit instructions, boundary definitions, disambiguation rules, or negative constraints to eliminate those errors.
3. PRESERVE STRENGTHS: Do not break what was already working. Keep the prompt clear, concise, and structured.
4. Output your response in the exact format:
<diagnosis>
[Detailed analysis of failure patterns and root causes]
</diagnosis>

<strategy>
[Key additions or adjustments made to fix the errors]
</strategy>

<new_prompt>
[The complete, revised system prompt]
</new_prompt>
"""

class ReflectorEngine:
    def __init__(self, client: OllamaClient, reflector_model: str):
        self.client = client
        self.reflector_model = reflector_model

    async def reflect_and_mutate(
        self,
        candidate: PromptCandidate,
        task_description: str,
        objectives: List[str],
        failure_traces: List[ExecutionTrace],
        success_traces: List[ExecutionTrace],
    ) -> Tuple[str, str, str]:
        """
        Run LLM reflection on failure traces and return (new_prompt, diagnosis, strategy).
        """
        # Format the failure traces into an informative summary
        failure_text_blocks = []
        for i, t in enumerate(failure_traces[:4], 1):
            failure_text_blocks.append(
                f"--- Failure Example {i} ---\n"
                f"Input: {t.input_text}\n"
                f"Expected: {t.ground_truth}\n"
                f"Actual Model Output: {t.model_output}\n"
                f"Critique / Error: {t.failure_critique or 'Failed evaluation criteria'}\n"
            )

        success_text_blocks = []
        for i, t in enumerate(success_traces[:2], 1):
            success_text_blocks.append(
                f"--- Successful Example {i} ---\n"
                f"Input: {t.input_text}\n"
                f"Actual Output: {t.model_output}\n"
            )

        failures_str = "\n".join(failure_text_blocks) if failure_text_blocks else "No severe failures recorded."
        successes_str = "\n".join(success_text_blocks) if success_text_blocks else "None provided."

        prompt_for_reflector = f"""### TASK DESCRIPTION:
{task_description}

### CURRENT OPTIMIZATION OBJECTIVES:
{', '.join(objectives)}

### CURRENT PROMPT UNDER EVALUATION:
\"\"\"
{candidate.prompt_text}
\"\"\"

### EXECUTION TRACES - FAILED CASES:
{failures_str}

### EXECUTION TRACES - SUCCESSFUL CASES (PRESERVE THESE):
{successes_str}

Please perform deep reflection on the failed cases. Diagnose why the current prompt allowed these mistakes, and write an improved new prompt.
Follow the format with <diagnosis>, <strategy>, and <new_prompt> tags.
"""
        logger.info(f"Reflecting on candidate {candidate.candidate_id} with {len(failure_traces)} failure traces...")

        resp = await self.client.generate(
            model=self.reflector_model,
            prompt=prompt_for_reflector,
            system=REFLECTOR_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=1500
        )

        if not resp["success"] or not resp["text"]:
            logger.warning(f"Reflector LLM failed: {resp.get('error')}. Using heuristic evolutionary reflection.")
            return self._heuristic_fallback_reflection(candidate, failure_traces)

        text = resp["text"]
        diagnosis = self._extract_tag(text, "diagnosis") or "Identified edge-case boundary errors and formatting deviations."
        strategy = self._extract_tag(text, "strategy") or "Added explicit disambiguation and schema enforcement."
        new_prompt = self._extract_tag(text, "new_prompt")

        if not new_prompt or len(new_prompt.strip()) < 15:
            # Fallback if tags weren't followed strictly
            if "```" in text:
                matches = re.findall(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
                if matches:
                    new_prompt = matches[-1].strip()
            if not new_prompt:
                new_prompt = text.strip()

        return new_prompt.strip(), diagnosis.strip(), strategy.strip()

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _heuristic_fallback_reflection(
        self,
        candidate: PromptCandidate,
        failures: List[ExecutionTrace]
    ) -> Tuple[str, str, str]:
        """Algorithmic fallback reflection if Ollama LLM is unavailable."""
        reasons = [f.failure_critique for f in failures if f.failure_critique]
        has_json_error = any("json" in r.lower() for r in reasons)
        has_classification_error = any("classif" in r.lower() or "intent" in r.lower() for r in reasons)

        additions = []
        if has_json_error:
            additions.append("CRITICAL: Respond ONLY with valid, raw JSON. Do not include markdown codeblocks or explanatory commentary.")
        if has_classification_error:
            additions.append("RULES: Carefully evaluate edge-case boundaries before selecting category. Disambiguate overlapping intents.")

        additions_str = "\n".join(additions)
        new_prompt = f"{candidate.prompt_text.strip()}\n\n{additions_str}".strip()
        diagnosis = f"Detected {len(failures)} execution trace failures related to formatting and edge-case boundaries."
        strategy = "Injected strict boundary constraints and raw formatting invariants."
        return new_prompt, diagnosis, strategy
