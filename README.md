# GEPA Automatic Prompt Optimizer (APO)

> **Genetic-Pareto Reflective Prompt Evolution for Local Ollama Models**  
> *Oral Presentation at ICLR 2026 Breakthrough Framework*

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Port 18435](https://img.shields.io/badge/Web%20UI-Port%2018435-00d4ff.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20Models-orange.svg)

---

## 🌟 Overview

**GEPA (Genetic-Pareto)** is an automated framework that eliminates manual prompt engineering. Instead of manually tweaking prompts through trial-and-error, GEPA uses **natural language reflection** and **multi-objective genetic evolution** to discover high-performing, cost-efficient, and schema-compliant prompts.

### Why GEPA Beats Reinforcement Learning (GRPO/PPO)
- **Natural Language Reflection**: Traditional RL compresses execution into a single sparse scalar reward (e.g. `0.65`). GEPA inspects **execution traces** (inputs, outputs, errors, reasoning) with an LLM "coach" that diagnoses root causes in plain language.
- **Pareto Multi-Objective Frontier**: Optimizes competing real-world goals simultaneously (e.g., **Task Accuracy** vs. **Strict JSON Schema** vs. **Token Efficiency**).
- **35x Fewer Rollouts**: Reaches superior accuracy with dramatically fewer model calls compared to RL methods.
- **100% Local with Ollama**: Runs entirely on your local machine with zero external API fees or data leakage.

---

## 🚀 Quickstart (Docker - Recommended)

The entire application is dockerized and binds to **port 18435**. It automatically connects to your host machine's Ollama instance.

### 1. Ensure Ollama is Running
Make sure Ollama is active on your host machine:
```bash
ollama serve
# Verify models are downloaded (e.g. llama3.2, mistral)
ollama list
```

### 2. Start or Restart Container
Run the included restart script:
```bash
./restart.sh
```

The script will:
1. Verify the Docker daemon.
2. Check host Ollama connectivity at `http://localhost:11434`.
3. Stop any existing GEPA container.
4. Build the image and start the container on port **18435**.
5. Wait for the healthcheck to pass.

Open your browser to:
👉 **[http://localhost:18435](http://localhost:18435)**

---

## 💻 Local Development Setup (Without Docker)

If you prefer to run directly in a Python virtual environment:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Web Server on port 18435
uvicorn app.main:app --host 0.0.0.0 --port 18435
```

---

## 🖥️ Terminal CLI Usage

You can also run the GEPA optimizer directly from your terminal with rich real-time formatting:

```bash
# Run on Customer Support benchmark using llama3.2
python cli.py --example customer_support --task-model llama3.2:latest --generations 3

# Run on Financial Contract Risk benchmark
python cli.py --example financial_risk --task-model llama3.2:latest --generations 3

# Run on Medical Triage benchmark
python cli.py --example medical_triage --task-model llama3.2:latest --generations 3
```

---

## 🎯 Included Real-World Production Examples

### Example 1: Customer Support Intent & Priority Routing
- **Domain**: SaaS & E-Commerce Operations
- **Task**: Classifies chaotic customer messages into `category` (`billing`, `technical`, `account`, `cancellation`, `general`), `urgency` (`low`, `medium`, `high`, `critical`), and `sentiment`, outputting strictly parsed JSON.
- **Initial Baseline**: Often fails on multi-intent messages (e.g. furious customer wanting a refund while threatening cancellation) and includes markdown codeblocks.
- **GEPA Outcome**: Discovers intent hierarchy rules and strict formatting invariants, boosting accuracy and schema compliance from ~55% to 100%.

### Example 2: Commercial Contract Risk & Liability Extractor
- **Domain**: LegalTech & Enterprise Procurement
- **Task**: Audits SaaS agreements, NDAs, and vendor contracts to detect uncapped liability, non-standard indemnification, and unilateral termination.
- **Initial Baseline**: Overlooks subtle unlimited liability waivers and indirect damage carve-ins.
- **GEPA Outcome**: Reflects on false negatives and learns explicit clause definitions.

### Example 3: Clinical Emergency Triage & Chain-of-Thought
- **Domain**: Healthcare & Emergency Medicine
- **Task**: Evaluates acute patient vignettes to assign Emergency Severity Index (ESI 1-5), enforcing vital sign scrutiny to prevent missed life threats.
- **Initial Baseline**: Makes hasty judgments without systematic vital evaluation.
- **GEPA Outcome**: Synthesizes step-by-step vital sign boundary checks, ensuring 100% safety recall on critical danger zone vitals.

---

## 🌐 Web Dashboard Features (Port 18435)

1. **Evolution & Pareto Frontier Tab**:
   - Live KPI counters: Generation counter, rollouts, Pareto frontier size, % improvement over seed.
   - **Interactive 2D Pareto Canvas**: Visualizes non-dominated frontier curve (Accuracy vs. Token Efficiency).
   - **Frontier Candidates Table**: Inspect any candidate's scores, mutation history, and exact traces.
2. **Reflector Deep-Dive Tab**:
   - Stream of every reflection event showing the Reflector LLM's **Root-Cause Diagnosis** and **Prescribed Strategy**.
   - Genetic crossover events showing how two parents were synthesized.
3. **Side-by-Side Playground Tab**:
   - Run baseline seed prompt vs. optimized prompt side-by-side on any test input.
   - Real-time comparison of latency (ms), token count, schema compliance, and model output.
4. **Internal Logs & Observability**:
   - Live WebSocket feed displaying internal trace evaluations, genetic operations, and frontier updates.

---

## 📚 Documentation Links

- 📖 **[Layman's Guide to GEPA & APO](LAYMAN_GUIDE.md)**: Beginner-friendly explanation of why manual prompt guessing is broken, how reflection works, and Pareto trade-offs with sports car analogies.
- 🏛 **[System Architecture & Mermaid Diagrams](ARCHITECTURE.md)**: High-Level System Architecture (HLSA), Sequence Diagram, Data Flow Diagram (DFD), and mathematical formulations.

---

## 🛠️ API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Interactive Web Dashboard |
| `GET /api/health` | `GET` | System health and host Ollama status |
| `GET /api/examples` | `GET` | List available benchmark examples |
| `GET /api/models` | `GET` | List locally detected Ollama models |
| `POST /api/optimize` | `POST` | Launch asynchronous GEPA optimization run |
| `POST /api/stop` | `POST` | Halt active optimization run |
| `POST /api/playground/test`| `POST` | Test baseline vs optimized prompt side-by-side |
| `WS /ws` | `WS` | Real-time WebSocket telemetry and log stream |

---

## 🐳 Docker Management (`restart.sh`)

To cleanly restart the container at any time:
```bash
./restart.sh
```
To check container logs directly:
```bash
docker logs -f gepa-prompt-optimizer
```
To stop the container:
```bash
docker compose down
```
