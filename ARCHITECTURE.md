# System Architecture & Technical Specification

## GEPA: Genetic-Pareto Reflective Automatic Prompt Optimizer

This document provides a comprehensive technical reference for the GEPA Prompt Optimization engine, its multi-objective evolutionary loop, LLM reflection mechanisms, and containerized deployment architecture.

---

## 1. High-Level System Architecture & Flow

The GEPA framework is structured into four primary connected tiers, with a dedicated observability and delivery pipeline:

1. **Presentation & Client Tier**: Web Dashboard (Port 18435) and rich terminal CLI (`cli.py`) for configuring optimization targets, monitoring Pareto convergence, and comparing prompts side-by-side.
2. **API & Orchestration Tier**: FastAPI application (`app/main.py`) managing asynchronous background optimization workers and real-time WebSocket telemetry streaming.
3. **Core GEPA Algorithmic Engine**: Evaluator & trace collector (`evaluator.py`), multi-objective Pareto dominance engine (`pareto.py`), and natural language LLM reflector & genetic mutator (`reflector.py`, `genetic_ops.py`).
4. **Local LLM Inference Tier**: Host Ollama instance (Port 11434) running task models (e.g. `llama3.2`) for evaluation rollouts and reflector models (e.g. `mistral`) for failure diagnosis.
5. **Observability & Delivery Tier**: WebSocket event dispatcher delivering live Pareto frontier updates, metric convergence charts, and production-ready optimal prompts.

### High-Level System Connectivity & Evolutionary Flow

```mermaid
flowchart TD
    subgraph TIER1 ["1. Presentation & Client Tier"]
        CLIENT["🖥️ Web Dashboard (Port 18435) / Rich CLI (cli.py)<br/>• Configure seed prompt, benchmark dataset & target objectives<br/>• Live Pareto frontier visualization & candidate comparisons"]
    end

    subgraph TIER2 ["2. API & Orchestration Tier (FastAPI)"]
        API["⚡ FastAPI Application (app/main.py)<br/>• Job Lifecycle Manager: async optimization workers<br/>• WebSocket Telemetry Hub: real-time streaming broadcaster"]
    end

    subgraph TIER3 ["3. GEPA Optimization Engine (app/core)"]
        direction LR
        EVAL["🔍 Step 1: Evaluator<br/>Run benchmark tests & collect traces"]
        PARETO["📊 Step 2: Pareto Sorter<br/>Rank Accuracy vs Schema vs Cost"]
        REFLECT["🧬 Step 3: Reflector & Mutator<br/>LLM diagnosis & genetic breeding"]

        EVAL -->|"Candidate Traces"| PARETO
        PARETO -->|"Frontier Parents"| REFLECT
        REFLECT -->|"Mutated Prompts"| EVAL
    end

    subgraph TIER4 ["4. Local Inference Tier (Ollama Port 11434)"]
        direction LR
        TASK_LLM["🦙 Task Model (llama3.2)<br/>Test rollouts"]
        REF_LLM["🧠 Reflector Model (mistral)<br/>Error diagnosis"]
    end

    subgraph TIER5 ["5. Observability & Delivery"]
        direction LR
        LIVE["📡 Real-Time Telemetry<br/>Live WebSocket updates"]
        BEST["🏆 Optimal Pareto Prompt<br/>Production export"]
    end

    CLIENT -->|"1. Submit Optimization Config"| API
    API -->|"2. Dispatch Optimization Run"| EVAL

    EVAL <-->|"Rollouts"| TASK_LLM
    REFLECT <-->|"Diagnosis"| REF_LLM

    PARETO -.->|"3. Stream Telemetry Events"| LIVE
    PARETO ==>|"4. Deliver Optimal Frontier"| BEST
```

### Component Connectivity & Data Flow Summary

| Connection / Link | Source → Target | Protocol / Mechanism | Description |
| :--- | :--- | :--- | :--- |
| **1. Job Submission** | Web UI / CLI → FastAPI | HTTP `POST /api/optimize` | User configures seed prompt, benchmark dataset, and optimization hyper-parameters. |
| **2. Worker Dispatch** | FastAPI → GEPA Core | Async Task Runner | Background thread initializes Population Generation 0 and kicks off the evolutionary loop. |
| **3. Benchmark Rollouts** | Evaluator ↔ Ollama | REST `POST /api/generate` | Local Ollama runs test cases against candidate prompts, recording accuracy, schema validity, and latency. |
| **4. Failure Reflection** | Reflector ↔ Ollama | REST `POST /api/generate` | Reflector LLM diagnoses root-cause error traces and synthesizes targeted instruction fixes. |
| **5. Live Telemetry** | Pareto Sorter → Web UI | WebSocket `/ws` | Streaming events broadcast live frontier coordinates, candidate scores, and diagnostic reflections. |
| **6. Prompt Delivery** | GEPA Core → Web UI | REST / WebSocket / File | Production-ready non-dominated prompt candidate delivered with side-by-side playground testing. |

---

## 2. End-to-End Sequence Diagram

The following sequence diagram illustrates the complete execution flow from when a user clicks "Start GEPA Optimization" to when the final Pareto-optimal prompt is delivered:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Web UI
    participant API as FastAPI Server
    participant GEPA as GEPA Optimizer Core
    participant Ollama as Local Ollama (Port 11434)

    User->>API: 1. POST /api/optimize (Seed prompt, benchmark, hyperparams)
    API->>GEPA: 2. Launch background optimization job
    API-->>User: 3. Return Job ID (200 OK)

    loop Generation 0 to N (Evolutionary Loop)
        GEPA->>Ollama: 4. Rollout candidates on benchmark samples
        Ollama-->>GEPA: Execution outputs, latency & token usage
        GEPA->>GEPA: 5. Calculate Pareto Frontier (Accuracy, Schema, Cost)
        GEPA-->>User: 6. Stream live telemetry snapshot (WebSocket)
        GEPA->>Ollama: 7. Request LLM reflection on failure traces
        Ollama-->>GEPA: Diagnosis, fix strategy & mutated prompt
    end

    GEPA->>API: 8. Optimization complete (Best Pareto frontier candidates)
    API-->>User: 9. Final notification & deliver optimal prompts
    User->>API: 10. POST /api/playground/test (Side-by-side verification)
    API->>Ollama: Test Baseline vs Optimized prompt
    Ollama-->>API: Outputs & comparative metrics
    API-->>User: Side-by-side diff & verification report
```

---

## 3. Data Flow Diagram (DFD)

The data flow highlights how raw failure critiques are transformed into refined prompt instructions without losing schema validity or token efficiency:

```mermaid
flowchart LR
    subgraph IN ["1. Inputs"]
        direction TB
        SEED["Seed Prompt"]
        DATA["Benchmark Dataset"]
        OBJ["Objective Criteria<br/>(Acc, Schema, Cost)"]
    end

    subgraph EVAL ["2. Candidate Evaluation"]
        direction TB
        ROLLOUT["Ollama Rollouts<br/>Task Model Inference"]
        TRACE["Trace Collector<br/>Metrics & Error Logs"]
        ROLLOUT --> TRACE
    end

    subgraph PARETO ["3. Multi-Objective Selection"]
        direction TB
        DOM["Dominance Check<br/>A dominates B?"]
        FRONT["Pareto Frontier Pool<br/>Rank 1 Non-Dominated"]
        DOM --> FRONT
    end

    subgraph REFLECT ["4. Natural Language Reflection"]
        direction TB
        FILTER["Failure Trace Filter<br/>Extract Failed Cases"]
        COACH["Reflector LLM<br/>Root-Cause Diagnosis"]
        MUTATE["Mutated / Crossed<br/>Candidate Prompts"]
        FILTER --> COACH --> MUTATE
    end

    subgraph OUT ["5. Outputs"]
        direction TB
        BEST["Optimal Pareto Prompts"]
        STREAM["WebSocket Telemetry"]
    end

    IN --> ROLLOUT
    TRACE --> DOM
    FRONT --> FILTER
    MUTATE -->|"New Generation"| ROLLOUT
    FRONT --> BEST
    TRACE -.-> STREAM
```

---

## 4. Algorithmic Formulation

### 4.1 Pareto Dominance Formulation
Let $C = \{c_1, c_2, \dots, c_m\}$ be candidate prompts. For each candidate $c$, the evaluator computes an objective vector:
$$\mathbf{F}(c) = \left( f_1(c), f_2(c), \dots, f_k(c) \right)$$
where:
- $f_1(c) \in [0, 1]$: Core Task Accuracy / F1
- $f_2(c) \in [0, 1]$: Strict Schema Compliance (JSON validity, key presence)
- $f_3(c) \in [0, 1]$: Token Efficiency ($1.0 - \text{penalty for verbosity}$)

Candidate $c_a$ **Pareto-dominates** candidate $c_b$ ($c_a \succ c_b$) if and only if:
$$\forall i \in \{1, \dots, k\}, \quad f_i(c_a) \ge f_i(c_b) \quad \land \quad \exists j \in \{1, \dots, k\}, \quad f_j(c_a) > f_j(c_b)$$

The **Pareto Frontier** $\mathcal{P}^*$ is the subset of all candidates not dominated by any other candidate:
$$\mathcal{P}^* = \left\lbrace c \in C \;\middle|\; \nexists c' \in C : c' \succ c \right\rbrace$$

### 4.2 Crowding Distance Diversity Metric
To prevent all prompts from converging to a single point along the trade-off curve, GEPA calculates crowding distance $d_i$ for each solution on the frontier:
$$d_i = \sum_{m=1}^{k} \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$
Candidates with higher crowding distance are prioritized during parent selection, preserving diverse prompt styles (e.g. ultra-compact vs highly-detailed reasoning).

### 4.3 Natural Language Reflection Formulation
Unlike Reinforcement Learning where feedback is a scalar reward $R \in \mathbb{R}$, GEPA constructs a natural language diagnostic context $\mathcal{D}$:
$$\mathcal{D} = \left\lbrace (x_i, y_i, \hat{y}_i, \mathcal{C}_i) \;\middle|\; \text{trace } i \text{ failed} \right\rbrace$$
where:
- $x_i$: input query
- $y_i$: target ground truth
- $\hat{y}_i$: model output
- $\mathcal{C}_i$: specific failure critique

The Reflector LLM evaluates:
$$(\text{Diagnosis}, \text{Strategy}, c_{\text{new}}) \sim P_{\text{reflector}}\left( \cdot \;\middle|\; \mathcal{D}, c_{\text{parent}}, \text{TaskSpec} \right)$$

---

## 5. Structured Internal Log Taxonomy

The system emits typed telemetry events over WebSockets and stdout:

| Event Type | Description | Key Payload Attributes |
| :--- | :--- | :--- |
| `LOG` | Internal execution logs | `level`, `message`, `details` |
| `SAMPLE_EVALUATED` | Single sample rollout | `candidate_id`, `sample_id`, `passed`, `metrics` |
| `REFLECTION_COMPLETED` | Reflector LLM analysis | `generation`, `parent_id`, `diagnosis`, `strategy`, `mutated_prompt` |
| `CROSSOVER_COMPLETED` | Hybrid prompt synthesis | `generation`, `parent_a`, `parent_b`, `rationale`, `crossed_prompt` |
| `GENERATION_SNAPSHOT` | Frontier state snapshot | `generation`, `frontier_count`, `best_scores`, `frontier_candidates` |
| `OPTIMIZATION_FINISHED` | Final run summary | `duration_seconds`, `frontier_size`, `best_candidate`, `initial_candidate` |
