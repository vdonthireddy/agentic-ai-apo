"""
Genetic Operators for GEPA:
Parent selection from Pareto frontier, reflective crossover, and elitism preservation.
"""

import logging
import random
from typing import List, Tuple
from app.core.evaluator import PromptCandidate
from app.core.ollama_client import OllamaClient
from app.core.reflector import ReflectorEngine

logger = logging.getLogger("GEPA.GeneticOps")

CROSSOVER_SYSTEM_PROMPT = """You are an expert Prompt Synthesizer in the GEPA Genetic-Pareto framework.
Your task is to perform genetic crossover on two high-performing parent prompts from the Pareto frontier.
Synthesize a single superior prompt that combines the distinct strengths of both parents while removing redundancy.

Output format:
<crossover_rationale>
[Brief rationale explaining how the strengths of Parent A and Parent B were combined]
</crossover_rationale>

<new_prompt>
[The final combined system prompt]
</new_prompt>
"""

class GeneticEngine:
    def __init__(self, reflector: ReflectorEngine, client: OllamaClient, reflector_model: str):
        self.reflector = reflector
        self.client = client
        self.reflector_model = reflector_model

    def select_parent(self, frontier: List[PromptCandidate]) -> PromptCandidate:
        """
        Sample a parent from the Pareto frontier.
        Favors candidates with higher crowding distance to preserve genetic diversity.
        """
        if not frontier:
            raise ValueError("Frontier is empty, cannot select parent.")
        if len(frontier) == 1:
            return frontier[0]

        # Candidates with inf crowding distance (extremes) get high weight
        weights = []
        for c in frontier:
            if c.crowding_distance == float("inf"):
                weights.append(5.0)
            else:
                weights.append(max(0.5, c.crowding_distance))

        return random.choices(frontier, weights=weights, k=1)[0]

    async def crossover(
        self,
        parent_a: PromptCandidate,
        parent_b: PromptCandidate,
        task_description: str,
        objectives: List[str]
    ) -> Tuple[str, str]:
        """
        Synthesize a child prompt by crossing over two Pareto-optimal parents.
        """
        logger.info(f"Performing crossover between {parent_a.candidate_id} and {parent_b.candidate_id}...")

        prompt = f"""### TASK:
{task_description}

### OBJECTIVES:
{', '.join(objectives)}

### PARENT PROMPT A (Scores: {parent_a.scores}):
\"\"\"
{parent_a.prompt_text}
\"\"\"

### PARENT PROMPT B (Scores: {parent_b.scores}):
\"\"\"
{parent_b.prompt_text}
\"\"\"

Synthesize a single child prompt that merges Parent A's advantages with Parent B's advantages.
Follow the required output format with <crossover_rationale> and <new_prompt> tags.
"""
        resp = await self.client.generate(
            model=self.reflector_model,
            prompt=prompt,
            system=CROSSOVER_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=1500
        )

        if resp["success"] and resp["text"]:
            text = resp["text"]
            rationale = self.reflector._extract_tag(text, "crossover_rationale") or "Combined complementary rules from both parents."
            new_prompt = self.reflector._extract_tag(text, "new_prompt")
            if new_prompt and len(new_prompt.strip()) > 15:
                return new_prompt.strip(), rationale.strip()

        # Fallback combination
        merged = f"{parent_a.prompt_text.strip()}\n\n# Additional Guidelines:\n{parent_b.prompt_text.strip()}"
        return merged, "Heuristic merge of parent prompts."
