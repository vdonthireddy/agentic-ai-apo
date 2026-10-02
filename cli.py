#!/usr/bin/env python3
"""
CLI Runner for GEPA (Genetic-Pareto) Automatic Prompt Optimizer.
Allows developers to run prompt optimization from terminal with rich progress formatting.
"""

import argparse
import asyncio
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

from app.config import settings
from app.core.ollama_client import OllamaClient
from app.core.gepa_optimizer import GEPAOptimizer
from app.examples import get_example, list_examples

console = Console()

async def run_cli(args):
    console.print(Panel.fit(
        "[bold cyan]GEPA (Genetic-Pareto) Automatic Prompt Optimizer[/bold cyan]\n"
        "[dim]Reflective Prompt Evolution with Local Ollama Models[/dim]",
        border_style="cyan"
    ))

    # Check Ollama
    client = OllamaClient(base_url=args.ollama_url)
    health = await client.check_health()
    if not health["connected"]:
        console.print(f"[bold red]❌ Failed to connect to Ollama at {args.ollama_url}[/bold red]")
        console.print("[yellow]Please ensure Ollama is running (`ollama serve`).[/yellow]")
        sys.exit(1)

    console.print(f"[green]✓ Connected to Ollama at {health['url']}[/green]")
    console.print(f"[dim]Available models: {', '.join(health['models'])}[/dim]\n")

    example = get_example(args.example)
    if not example:
        console.print(f"[bold red]Unknown example: {args.example}[/bold red]")
        console.print(f"Available examples: {[e['id'] for e in list_examples()]}")
        sys.exit(1)

    console.print(f"[bold green]Task:[/bold green] {example.title}")
    console.print(f"[dim]{example.description}[/dim]")
    console.print(f"[bold yellow]Objectives:[/bold yellow] {', '.join(example.objectives)}")
    console.print(f"[bold blue]Task Model:[/bold blue] {args.task_model} | [bold blue]Reflector:[/bold blue] {args.reflector_model}")
    console.print(f"[dim]Population: {args.population}, Generations: {args.generations}[/dim]\n")

    console.print(Panel(
        f"[bold]Initial Baseline Prompt:[/bold]\n{example.initial_prompt}",
        title="[bold yellow]Seed Prompt[/bold yellow]",
        border_style="yellow"
    ))

    def cli_callback(event):
        ev_type = event.get("type")
        data = event.get("data", {})
        if ev_type == "LOG":
            lvl = data.get("level", "INFO")
            msg = data.get("message", "")
            if lvl == "WARNING":
                console.print(f"[yellow]⚠ {msg}[/yellow]")
            elif lvl == "ERROR":
                console.print(f"[red]✖ {msg}[/red]")
            else:
                console.print(f"[dim]{msg}[/dim]")
        elif ev_type == "REFLECTION_COMPLETED":
            console.print(Panel(
                f"[bold red]Diagnosis:[/bold red] {data.get('diagnosis')}\n\n"
                f"[bold green]Strategy:[/bold green] {data.get('strategy')}",
                title=f"[bold magenta]🧠 Gen {data.get('generation')} Reflector Thought (Parent: {data.get('parent_id')})[/bold magenta]",
                border_style="magenta"
            ))
        elif ev_type == "GENERATION_SNAPSHOT":
            gen = data.get("generation")
            frontier = data.get("frontier_candidates", [])
            table = Table(title=f"Generation {gen} Pareto Frontier (Non-Dominated Candidates)")
            table.add_column("Candidate ID", style="cyan")
            table.add_column("Generation", style="dim")
            table.add_column("Scores", style="green")
            table.add_column("Avg Latency", style="yellow")
            table.add_column("Avg Tokens", style="blue")

            for c in frontier:
                scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in c.get("scores", {}).items())
                table.add_row(
                    c["candidate_id"],
                    str(c["generation"]),
                    scores_str,
                    f"{c['average_latency_ms']}ms",
                    str(c["average_token_count"])
                )
            console.print(table)
            console.print("")

    optimizer = GEPAOptimizer(
        task_model=args.task_model,
        reflector_model=args.reflector_model,
        example=example,
        client=client,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        event_callback=cli_callback
    )

    with console.status("[bold green]Running GEPA Evolutionary Optimization..."):
        summary = await optimizer.run()

    best = summary.get("best_candidate")
    initial = summary.get("initial_candidate")

    console.print("\n" + "=" * 70)
    console.print("[bold green]🎉 OPTIMIZATION COMPLETED SUCCESSFULLY![/bold green]")
    console.print("=" * 70 + "\n")

    if best and initial:
        comp_table = Table(title="Before vs. After Optimization Summary")
        comp_table.add_column("Metric", style="bold")
        comp_table.add_column("Initial Seed Prompt", style="red")
        comp_table.add_column("GEPA Optimized Prompt", style="green")

        all_objs = set(list(initial.get("scores", {}).keys()) + list(best.get("scores", {}).keys()))
        for obj in sorted(all_objs):
            s_init = initial.get("scores", {}).get(obj, 0.0)
            s_best = best.get("scores", {}).get(obj, 0.0)
            diff = s_best - s_init
            diff_str = f"(+{diff:.2f})" if diff >= 0 else f"({diff:.2f})"
            comp_table.add_row(obj, f"{s_init:.3f}", f"{s_best:.3f} {diff_str}")

        comp_table.add_row("Avg Latency", f"{initial.get('average_latency_ms', 0)}ms", f"{best.get('average_latency_ms', 0)}ms")
        comp_table.add_row("Avg Tokens", f"{initial.get('average_token_count', 0)}", f"{best.get('average_token_count', 0)}")
        console.print(comp_table)

        console.print(Panel(
            f"[bold green]{best['prompt_text']}[/bold green]",
            title=f"[bold green]Best Pareto-Optimal Prompt ({best['candidate_id']})[/bold green]",
            border_style="green"
        ))

def main():
    parser = argparse.ArgumentParser(description="GEPA Automatic Prompt Optimizer CLI")
    parser.add_argument("--example", default="customer_support", choices=["customer_support", "financial_risk", "medical_triage", "b2b_marketing"], help="Benchmark example")
    parser.add_argument("--task-model", default="llama3.2:latest", help="Ollama model for task inference")
    parser.add_argument("--reflector-model", default="llama3.2:latest", help="Ollama model for reflection")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--population", type=int, default=4, help="Population size")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.7, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.3, help="Crossover rate")

    args = parser.parse_args()
    asyncio.run(run_cli(args))

if __name__ == "__main__":
    main()
