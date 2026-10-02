"""
Master GEPA (Genetic-Pareto) Automatic Prompt Optimizer Engine.
Implements the full evolutionary loop:
  1. Multi-candidate evaluation & trace collection
  2. Pareto dominance ranking & crowding distance calculation
  3. LLM Natural Language Reflection on failure traces
  4. Genetic crossover and reflective mutation
  5. Real-time logging and telemetry emission
"""

import asyncio
from datetime import datetime
import logging
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.core.evaluator import ExecutionTrace, PromptCandidate
from app.core.genetic_ops import GeneticEngine
from app.core.ollama_client import OllamaClient
from app.core.pareto import compute_pareto_frontiers, get_pareto_frontier
from app.core.reflector import ReflectorEngine
from app.examples.base import BaseExample, BenchmarkSample

logger = logging.getLogger("GEPA.Optimizer")

class GEPAOptimizer:
    def __init__(
        self,
        task_model: str,
        reflector_model: str,
        example: BaseExample,
        client: Optional[OllamaClient] = None,
        population_size: int = 5,
        generations: int = 4,
        mutation_rate: float = 0.7,
        crossover_rate: float = 0.3,
        event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.task_model = task_model
        self.reflector_model = reflector_model
        self.example = example
        self.client = client or OllamaClient()
        self.population_size = max(3, population_size)
        self.generations = max(1, generations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.event_callback = event_callback

        self.reflector = ReflectorEngine(self.client, self.reflector_model)
        self.genetic = GeneticEngine(self.reflector, self.client, self.reflector_model)

        self.all_candidates: List[PromptCandidate] = []
        self.frontier: List[PromptCandidate] = []
        self.generation_history: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.should_stop: bool = False

    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit structured event for WebSocket clients and UI telemetry."""
        payload = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        if self.event_callback:
            try:
                if asyncio.iscoroutinefunction(self.event_callback):
                    await self.event_callback(payload)
                else:
                    self.event_callback(payload)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    async def log_internal(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send human-readable diagnostic log to UI and standard logger."""
        getattr(logger, level.lower(), logger.info)(message)
        await self.emit("LOG", {
            "level": level.upper(),
            "message": message,
            "details": details or {}
        })

    async def evaluate_candidate(
        self,
        candidate: PromptCandidate,
        samples: List[BenchmarkSample]
    ) -> PromptCandidate:
        """
        Roll out prompt candidate on evaluation samples and aggregate multi-objective metrics.
        """
        await self.log_internal(
            "INFO",
            f"⚡ Evaluating Candidate [{candidate.candidate_id}] (Gen {candidate.generation}) on {len(samples)} samples...",
            {"prompt_snippet": candidate.prompt_text[:120] + "..."}
        )

        traces: List[ExecutionTrace] = []
        total_latency = 0.0
        total_tokens = 0

        for sample in samples:
            if self.should_stop:
                break

            gen_result = await self.client.generate(
                model=self.task_model,
                prompt=sample.input_text,
                system=candidate.prompt_text,
                temperature=0.1
            )

            latency = gen_result["latency_ms"]
            tokens = gen_result["token_count"]
            model_text = gen_result["text"]

            trace = self.example.evaluate_output(
                sample=sample,
                model_output=model_text,
                latency_ms=latency,
                token_count=tokens
            )
            trace.candidate_prompt = candidate.prompt_text
            traces.append(trace)
            total_latency += latency
            total_tokens += tokens

            await self.emit("SAMPLE_EVALUATED", {
                "candidate_id": candidate.candidate_id,
                "sample_id": sample.id,
                "passed": trace.passed,
                "metrics": trace.metrics,
                "failure_critique": trace.failure_critique
            })

        if not traces:
            return candidate

        candidate.traces = traces
        candidate.average_latency_ms = round(total_latency / len(traces), 2)
        candidate.average_token_count = round(total_tokens / len(traces), 1)

        # Aggregate objective scores across samples
        scores: Dict[str, float] = {}
        for obj in self.example.objectives:
            vals = [t.metrics.get(obj, 0.0) for t in traces]
            scores[obj] = round(sum(vals) / len(vals), 3) if vals else 0.0

        candidate.scores = scores

        await self.log_internal(
            "INFO",
            f"✅ Completed evaluation of [{candidate.candidate_id}] -> Scores: {candidate.scores} | Latency: {candidate.average_latency_ms}ms",
            {"scores": candidate.scores, "candidate_id": candidate.candidate_id}
        )

        return candidate

    async def _initialize_population(self) -> List[PromptCandidate]:
        """
        Create generation 0 with baseline prompt plus initial heuristic variations.
        """
        population: List[PromptCandidate] = []

        # Candidate 0: Initial Seed Prompt
        seed_candidate = PromptCandidate(
            candidate_id="cand-gen0-seed",
            generation=0,
            prompt_text=self.example.initial_prompt.strip(),
            mutation_history=["Initial baseline seed prompt"]
        )
        population.append(seed_candidate)

        # Seed Variation 1: Strict JSON constraint injection
        var1 = PromptCandidate(
            candidate_id="cand-gen0-var1",
            generation=0,
            prompt_text=(
                f"{self.example.initial_prompt.strip()}\n\n"
                "CRITICAL: Output must be strictly valid JSON without any markdown formatting, preambles, or conversational text."
            ),
            mutation_history=["Seed variant: Strict JSON constraint"]
        )
        population.append(var1)

        # Seed Variation 2: Chain of Thought / Step-by-Step reasoning
        var2 = PromptCandidate(
            candidate_id="cand-gen0-var2",
            generation=0,
            prompt_text=(
                f"{self.example.initial_prompt.strip()}\n\n"
                "Think step-by-step to analyze all nuances and edge-cases before deciding the final classification. Return strictly the required output."
            ),
            mutation_history=["Seed variant: Step-by-step reasoning"]
        )
        population.append(var2)

        # If population size is larger, add few-shot or structured layout variations
        if self.population_size > 3:
            var3 = PromptCandidate(
                candidate_id="cand-gen0-var3",
                generation=0,
                prompt_text=(
                    f"### SYSTEM INSTRUCTION:\n{self.example.initial_prompt.strip()}\n\n"
                    "### GUIDELINES:\n- Maintain high accuracy and verify edge conditions.\n- Provide compact, structured output."
                ),
                mutation_history=["Seed variant: Structured Markdown layout"]
            )
            population.append(var3)

        return population[:self.population_size]

    async def run(self) -> Dict[str, Any]:
        """
        Execute the full GEPA optimization loop across all generations.
        """
        self.is_running = True
        self.should_stop = False
        start_time = time.time()

        await self.log_internal(
            "INFO",
            f"🚀 Launching GEPA Optimizer for [{self.example.title}] using Task Model='{self.task_model}', Reflector='{self.reflector_model}'",
            {
                "generations": self.generations,
                "population_size": self.population_size,
                "objectives": self.example.objectives
            }
        )

        # Step 1: Initialize population
        current_pop = await self._initialize_population()

        # Step 2: Evaluate Generation 0
        for cand in current_pop:
            if self.should_stop:
                break
            await self.evaluate_candidate(cand, self.example.train_samples)
            self.all_candidates.append(cand)

        # Update Pareto frontier for Gen 0
        self.frontier = get_pareto_frontier(self.all_candidates, self.example.objectives)
        await self._record_generation_snapshot(0)

        # Step 3: Evolution Loop
        for gen in range(1, self.generations + 1):
            if self.should_stop:
                await self.log_internal("WARNING", "⏹ Optimization stopped early by user request.")
                break

            await self.log_internal(
                "INFO",
                f"🧬 --- Commencing Generation {gen}/{self.generations} (Current Pareto Frontier Size: {len(self.frontier)}) ---"
            )

            offspring_population: List[PromptCandidate] = []

            # Generate new candidates through reflective mutation and crossover
            while len(offspring_population) < self.population_size:
                if self.should_stop:
                    break

                roll = random.random()
                if roll < self.mutation_rate or len(self.frontier) < 2:
                    # Reflective Mutation
                    parent = self.genetic.select_parent(self.frontier)
                    failed_traces = [t for t in parent.traces if not t.passed]
                    success_traces = [t for t in parent.traces if t.passed]

                    if not failed_traces:
                        # If parent had no failures, mutate based on token efficiency or subtle improvement
                        failed_traces = parent.traces[:2]

                    await self.log_internal(
                        "INFO",
                        f"🧠 [Reflector] Diagnosing {len(failed_traces)} failure traces of Parent [{parent.candidate_id}]...",
                        {"parent_scores": parent.scores}
                    )

                    new_prompt, diagnosis, strategy = await self.reflector.reflect_and_mutate(
                        candidate=parent,
                        task_description=self.example.task_description,
                        objectives=self.example.objectives,
                        failure_traces=failed_traces,
                        success_traces=success_traces
                    )

                    cand_id = f"cand-gen{gen}-mut{uuid.uuid4().hex[:4]}"
                    child = PromptCandidate(
                        candidate_id=cand_id,
                        generation=gen,
                        prompt_text=new_prompt,
                        parent_ids=[parent.candidate_id],
                        mutation_history=parent.mutation_history + [f"Gen {gen} Mutation: {strategy}"]
                    )

                    await self.emit("REFLECTION_COMPLETED", {
                        "generation": gen,
                        "parent_id": parent.candidate_id,
                        "candidate_id": cand_id,
                        "diagnosis": diagnosis,
                        "strategy": strategy,
                        "mutated_prompt": new_prompt
                    })

                    offspring_population.append(child)

                else:
                    # Reflective Crossover
                    parent_a = self.genetic.select_parent(self.frontier)
                    # Pick distinct parent b if available
                    other_parents = [c for c in self.frontier if c.candidate_id != parent_a.candidate_id]
                    parent_b = random.choice(other_parents) if other_parents else parent_a

                    new_prompt, rationale = await self.genetic.crossover(
                        parent_a=parent_a,
                        parent_b=parent_b,
                        task_description=self.example.task_description,
                        objectives=self.example.objectives
                    )

                    cand_id = f"cand-gen{gen}-cross{uuid.uuid4().hex[:4]}"
                    child = PromptCandidate(
                        candidate_id=cand_id,
                        generation=gen,
                        prompt_text=new_prompt,
                        parent_ids=[parent_a.candidate_id, parent_b.candidate_id],
                        mutation_history=[f"Gen {gen} Crossover of [{parent_a.candidate_id}] & [{parent_b.candidate_id}]: {rationale}"]
                    )

                    await self.emit("CROSSOVER_COMPLETED", {
                        "generation": gen,
                        "parent_a": parent_a.candidate_id,
                        "parent_b": parent_b.candidate_id,
                        "candidate_id": cand_id,
                        "rationale": rationale,
                        "crossed_prompt": new_prompt
                    })

                    offspring_population.append(child)

            # Evaluate all offspring
            for child in offspring_population:
                if self.should_stop:
                    break
                await self.evaluate_candidate(child, self.example.train_samples)
                self.all_candidates.append(child)

            # Update Pareto frontier with all candidates evaluated so far (Elitism guaranteed)
            self.frontier = get_pareto_frontier(self.all_candidates, self.example.objectives)
            await self._record_generation_snapshot(gen)

        # Final validation evaluation of the Pareto Frontier
        await self.log_internal("INFO", f"🏁 Final Evaluation: Validating top Pareto candidates on unseen holdout test set...")
        best_candidate = self.get_best_candidate()
        if best_candidate and self.example.val_samples:
            val_trace_results = []
            for sample in self.example.val_samples:
                res = await self.client.generate(
                    model=self.task_model,
                    prompt=sample.input_text,
                    system=best_candidate.prompt_text,
                    temperature=0.1
                )
                t = self.example.evaluate_output(sample, res["text"], res["latency_ms"], res["token_count"])
                val_trace_results.append(t)

            val_scores = {}
            for obj in self.example.objectives:
                vals = [t.metrics.get(obj, 0.0) for t in val_trace_results]
                val_scores[obj] = round(sum(vals) / len(vals), 3)

            await self.log_internal(
                "INFO",
                f"🏆 Final Best Candidate [{best_candidate.candidate_id}] Validation Scores: {val_scores}",
                {"val_scores": val_scores, "candidate_id": best_candidate.candidate_id}
            )

        self.is_running = False
        duration = round(time.time() - start_time, 2)

        summary = {
            "status": "COMPLETED",
            "duration_seconds": duration,
            "total_candidates": len(self.all_candidates),
            "frontier_size": len(self.frontier),
            "best_candidate": self._candidate_to_dict(best_candidate) if best_candidate else None,
            "initial_candidate": self._candidate_to_dict(self.all_candidates[0]) if self.all_candidates else None,
            "frontier": [self._candidate_to_dict(c) for c in self.frontier],
            "generation_history": self.generation_history
        }

        await self.emit("OPTIMIZATION_FINISHED", summary)
        return summary

    def get_best_candidate(self) -> Optional[PromptCandidate]:
        """
        Return the top candidate on the Pareto frontier.
        Uses sum of normalized scores or primary objective.
        """
        if not self.frontier:
            if not self.all_candidates:
                return None
            return self.all_candidates[0]

        # Rank by primary objective first, then secondary
        primary_obj = self.example.objectives[0]
        sorted_frontier = sorted(
            self.frontier,
            key=lambda c: (c.scores.get(primary_obj, 0.0), sum(c.scores.values())),
            reverse=True
        )
        return sorted_frontier[0]

    async def _record_generation_snapshot(self, gen: int):
        """Record and emit current generation state and Pareto frontier."""
        best = self.get_best_candidate()
        snapshot = {
            "generation": gen,
            "frontier_count": len(self.frontier),
            "total_candidates_so_far": len(self.all_candidates),
            "best_scores": best.scores if best else {},
            "frontier_candidates": [self._candidate_to_dict(c) for c in self.frontier]
        }
        self.generation_history.append(snapshot)
        await self.emit("GENERATION_SNAPSHOT", snapshot)

    def _candidate_to_dict(self, c: PromptCandidate) -> Dict[str, Any]:
        return {
            "candidate_id": c.candidate_id,
            "generation": c.generation,
            "prompt_text": c.prompt_text,
            "parent_ids": c.parent_ids,
            "mutation_history": c.mutation_history,
            "scores": c.scores,
            "average_latency_ms": c.average_latency_ms,
            "average_token_count": c.average_token_count,
            "pareto_rank": c.pareto_rank,
            "crowding_distance": c.crowding_distance,
            "is_frontier": c.is_frontier,
            "trace_summary": [
                {
                    "sample_id": t.sample_id,
                    "passed": t.passed,
                    "metrics": t.metrics,
                    "failure_critique": t.failure_critique,
                    "latency_ms": t.latency_ms,
                    "token_count": t.token_count,
                    "output_snippet": t.model_output[:100]
                }
                for t in c.traces
            ]
        }

    def stop(self):
        """Signal optimizer to halt after current rollout."""
        self.should_stop = True
