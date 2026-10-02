// GEPA Automatic Prompt Optimizer - Modern Interactive Frontend
document.addEventListener("DOMContentLoaded", () => {
    // State
    let ws = null;
    let examplesData = {};
    let currentExample = null;
    let allCandidates = [];
    let frontierCandidates = [];
    let initialSeedCandidate = null;
    let bestCandidate = null;
    let activeTab = "tab-evolution";
    let isOptimizing = false;

    // DOM Elements
    const ollamaDot = document.getElementById("ollama-dot");
    const ollamaStatusText = document.getElementById("ollama-status-text");
    const optStateBadge = document.getElementById("opt-state-badge");
    const optStateText = document.getElementById("opt-state-text");
    const exampleSelect = document.getElementById("example-select");
    const taskDomain = document.getElementById("task-domain");
    const taskObjectives = document.getElementById("task-objectives");
    const taskDesc = document.getElementById("task-desc");
    const taskModelSelect = document.getElementById("task-model-select");
    const reflectorModelSelect = document.getElementById("reflector-model-select");
    const btnRefreshModels = document.getElementById("btn-refresh-models");
    const seedPromptPreview = document.getElementById("seed-prompt-preview");
    const customSection = document.getElementById("custom-task-section");
    const customTaskDesc = document.getElementById("custom-task-desc");
    const customSeedPrompt = document.getElementById("custom-seed-prompt");

    const paramPopSize = document.getElementById("param-pop-size");
    const paramGenerations = document.getElementById("param-generations");
    const paramMutationRate = document.getElementById("param-mutation-rate");
    const paramCrossoverRate = document.getElementById("param-crossover-rate");

    const btnStartOpt = document.getElementById("btn-start-opt");
    const btnStopOpt = document.getElementById("btn-stop-opt");

    // KPIs
    const kpiGen = document.getElementById("kpi-gen");
    const kpiGenSub = document.getElementById("kpi-gen-sub");
    const kpiCandidates = document.getElementById("kpi-candidates");
    const kpiFrontier = document.getElementById("kpi-frontier");
    const kpiImprovement = document.getElementById("kpi-improvement");
    const kpiImprovementSub = document.getElementById("kpi-improvement-sub");

    // Table & Canvas
    const frontierTableBody = document.getElementById("frontier-table-body");
    const frontierCountBadge = document.getElementById("frontier-count-badge");
    const paretoCanvas = document.getElementById("pareto-canvas");
    const chartTooltip = document.getElementById("chart-tooltip");

    // Reflector stream
    const reflectionStream = document.getElementById("reflection-stream");

    // Playground
    const playgroundInput = document.getElementById("playground-input");
    const btnRunPlayground = document.getElementById("btn-run-playground");
    const compInitPromptText = document.getElementById("comp-init-prompt-text");
    const compOptPromptText = document.getElementById("comp-opt-prompt-text");
    const compInitOutput = document.getElementById("comp-init-output");
    const compOptOutput = document.getElementById("comp-opt-output");
    const initMetrics = document.getElementById("init-metrics");
    const optMetrics = document.getElementById("opt-metrics");
    const pill1 = document.getElementById("pill-test-1");
    const pill2 = document.getElementById("pill-test-2");
    const pill3 = document.getElementById("pill-test-3");

    // Terminal
    const terminalLogs = document.getElementById("terminal-logs");
    const chkAutoscroll = document.getElementById("chk-autoscroll");
    const btnClearLogs = document.getElementById("btn-clear-logs");

    // Modal
    const modal = document.getElementById("candidate-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const modalTitle = document.getElementById("modal-candidate-title");
    const modalScoresGrid = document.getElementById("modal-scores-grid");
    const modalPromptText = document.getElementById("modal-prompt-text");
    const btnCopyPrompt = document.getElementById("btn-copy-prompt");
    const modalLineageList = document.getElementById("modal-lineage-list");
    const modalTracesList = document.getElementById("modal-traces-list");

    // 1. Initialization
    async function init() {
        setupTabs();
        setupCanvas();
        await checkHealth();
        await loadExamples();
        await loadModels();
        connectWebSocket();
        setupEventListeners();
    }

    // Tabs Setup
    function setupTabs() {
        document.querySelectorAll(".tab-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
                document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                activeTab = btn.dataset.tab;
                document.getElementById(activeTab).classList.add("active");
                if (activeTab === "tab-evolution") {
                    drawParetoChart();
                }
            });
        });
    }

    // Health Check
    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            if (data.ollama && data.ollama.connected) {
                ollamaDot.className = "indicator-dot online";
                ollamaStatusText.textContent = `Ollama: Online (${data.ollama.models.length} models)`;
            } else {
                ollamaDot.className = "indicator-dot";
                ollamaStatusText.textContent = "Ollama: Offline (Simulation Mode)";
            }
        } catch (e) {
            ollamaDot.className = "indicator-dot";
            ollamaStatusText.textContent = "Ollama: Unreachable";
        }
    }

    // Load Examples
    async function loadExamples() {
        try {
            const res = await fetch("/api/examples");
            const examples = await res.json();
            examples.forEach(ex => {
                examplesData[ex.id] = ex;
            });
            updateSelectedExample();
        } catch (e) {
            console.error("Error loading examples:", e);
        }
    }

    function updateSelectedExample() {
        const id = exampleSelect.value;
        if (id === "custom") {
            customSection.classList.remove("hidden");
            taskDomain.textContent = "User Defined";
            taskObjectives.textContent = "accuracy, token_efficiency";
            taskDesc.textContent = "Custom user-defined task and validation dataset.";
            seedPromptPreview.textContent = customSeedPrompt.value || "Enter your initial prompt above...";
            return;
        }

        customSection.classList.add("hidden");
        const ex = examplesData[id];
        if (!ex) return;
        currentExample = ex;

        taskDomain.textContent = ex.domain;
        taskObjectives.textContent = ex.objectives.join(", ");
        taskDesc.textContent = ex.description;
        seedPromptPreview.textContent = ex.initial_prompt;

        compInitPromptText.textContent = ex.initial_prompt;
        setupPillsForExample(id);
    }

    function setupPillsForExample(id) {
        if (id === "customer_support") {
            pill1.textContent = "Double Billed";
            pill1.dataset.val = "I was double billed for my subscription this morning ($199 charged twice)! Fix this immediately.";
            pill2.textContent = "Feature Request";
            pill2.dataset.val = "Hey team, loving the product! Do you support dark mode on iOS yet?";
            pill3.textContent = "Severe Outage";
            pill3.dataset.val = "Our production webhook is returning 504 Gateway Timeout errors intermittently. We are losing transactions.";
            playgroundInput.value = pill1.dataset.val;
        } else if (id === "financial_risk") {
            pill1.textContent = "Liability Cap";
            pill1.dataset.val = "In no event shall aggregate liability exceed total amounts paid in preceding 12 months.";
            pill2.textContent = "Uncapped Indemnity";
            pill2.dataset.val = "Vendor shall indemnify Customer against all claims and damages without limitation or monetary cap.";
            pill3.textContent = "Termination Notice";
            pill3.dataset.val = "Customer may terminate this Agreement immediately without cause upon written notice.";
            playgroundInput.value = pill2.dataset.val;
        } else if (id === "medical_triage") {
            pill1.textContent = "Chest Pain (ESI 1)";
            pill1.dataset.val = "68yo male with sudden crushing substernal chest pressure, diaphoresis. BP 88/54, HR 122, RR 26, SpO2 91%.";
            pill2.textContent = "Ankle Sprain (ESI 4)";
            pill2.dataset.val = "24yo female with isolated lateral right ankle pain after soccer. Bearing weight with limp. Vitals normal.";
            pill3.textContent = "Severe Headache (ESI 2)";
            pill3.dataset.val = "45yo female with sudden 'worst headache of life', photophobia, neck stiffness. BP 178/104, HR 88.";
            playgroundInput.value = pill1.dataset.val;
        }
    }

    // Load Local Ollama Models
    async function loadModels() {
        try {
            const res = await fetch("/api/models");
            const data = await res.json();
            const models = data.models || [];

            taskModelSelect.innerHTML = "";
            reflectorModelSelect.innerHTML = "";

            if (models.length === 0) {
                const opt = document.createElement("option");
                opt.value = "llama3.2:latest";
                opt.textContent = "llama3.2:latest (Default / Fallback)";
                taskModelSelect.appendChild(opt.cloneNode(true));
                reflectorModelSelect.appendChild(opt.cloneNode(true));
                return;
            }

            models.forEach(m => {
                const opt1 = document.createElement("option");
                opt1.value = m;
                opt1.textContent = m;
                taskModelSelect.appendChild(opt1);

                const opt2 = document.createElement("option");
                opt2.value = m;
                opt2.textContent = m;
                reflectorModelSelect.appendChild(opt2);
            });

            // Set smart defaults
            if (models.includes("llama3.2:latest")) {
                taskModelSelect.value = "llama3.2:latest";
                reflectorModelSelect.value = "llama3.2:latest";
            }
        } catch (e) {
            console.error("Error loading models:", e);
        }
    }

    // 2. WebSocket Telemetry
    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            logTerminal("INFO", "Connected to GEPA telemetry stream.");
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleStreamEvent(msg);
            } catch (e) {
                console.error("WS Parse error:", e);
            }
        };

        ws.onclose = () => {
            setTimeout(connectWebSocket, 3000);
        };
    }

    let sampleEvalCount = 0;

    function handleStreamEvent(event) {
        const type = event.type;
        const data = event.data || {};

        try {
            if (type === "LOG") {
                logTerminal(data.level, data.message);
            } else if (type === "SAMPLE_EVALUATED") {
                sampleEvalCount++;
                kpiCandidates.textContent = sampleEvalCount;
            } else if (type === "REFLECTION_COMPLETED") {
                // Also switch to the reflector tab automatically on first reflection
                appendReflectionCard(data);
            } else if (type === "CROSSOVER_COMPLETED") {
                appendCrossoverCard(data);
            } else if (type === "GENERATION_SNAPSHOT") {
                updateGenerationSnapshot(data);
            } else if (type === "OPTIMIZATION_FINISHED") {
                onOptimizationFinished(data);
            } else if (type === "OPTIMIZATION_ERROR") {
                onOptimizationError(data);
            }
        } catch (err) {
            console.error("[GEPA WS handler] Error processing event type:", type, err);
            logTerminal("ERROR", `UI handler failed for event '${type}': ${err.message}`);
        }
    }

    // Append Reflector card
    function appendReflectionCard(data) {
        const emptyBox = reflectionStream.querySelector(".empty-state-box");
        if (emptyBox) emptyBox.remove();

        const card = document.createElement("div");
        card.className = "reflection-card";
        card.innerHTML = `
            <div class="reflection-header">
                <span>🧠 Gen ${data.generation} Reflection | Mutating Parent [${data.parent_id}]</span>
                <span>Offspring: ${data.candidate_id}</span>
            </div>
            <div class="reflection-body">
                <div class="reflection-box">
                    <h5>🔍 Root Cause Diagnosis</h5>
                    <p>${data.diagnosis}</p>
                </div>
                <div class="reflection-box">
                    <h5>🛠 Prescribed Strategy</h5>
                    <p>${data.strategy}</p>
                </div>
            </div>
            <div class="comp-subheading">Mutated Prompt:</div>
            <pre class="code-view" style="max-height: 100px;">${data.mutated_prompt}</pre>
        `;
        reflectionStream.prepend(card);
    }

    // Append Crossover card
    function appendCrossoverCard(data) {
        const emptyBox = reflectionStream.querySelector(".empty-state-box");
        if (emptyBox) emptyBox.remove();

        const card = document.createElement("div");
        card.className = "reflection-card";
        card.style.borderLeftColor = "var(--primary)";
        card.innerHTML = `
            <div class="reflection-header" style="color: var(--primary);">
                <span>🔀 Gen ${data.generation} Genetic Crossover | Parents [${data.parent_a}] & [${data.parent_b}]</span>
                <span>Offspring: ${data.candidate_id}</span>
            </div>
            <div class="reflection-box">
                <h5>🧬 Synthesis Rationale</h5>
                <p>${data.rationale}</p>
            </div>
            <div class="comp-subheading">Synthesized Prompt:</div>
            <pre class="code-view" style="max-height: 100px;">${data.crossed_prompt}</pre>
        `;
        reflectionStream.prepend(card);
    }

    // Update generation state
    function updateGenerationSnapshot(data) {
        const gen = data.generation;
        const totalGens = paramGenerations.value;
        kpiGen.textContent = `${gen} / ${totalGens}`;
        kpiGenSub.textContent = `Gen ${gen} frontier converged`;

        frontierCandidates = data.frontier_candidates || [];
        kpiFrontier.textContent = frontierCandidates.length;
        frontierCountBadge.textContent = `${frontierCandidates.length} Candidates`;

        // Store candidates
        frontierCandidates.forEach(fc => {
            if (!allCandidates.find(c => c.candidate_id === fc.candidate_id)) {
                allCandidates.push(fc);
            }
        });

        // Compute improvement over seed
        if (allCandidates.length > 0) {
            initialSeedCandidate = allCandidates[0];
            const primaryObj = Object.keys(initialSeedCandidate.scores || {})[0] || "accuracy";
            const seedScore = initialSeedCandidate.scores?.[primaryObj] || 0.5;

            // Find best
            bestCandidate = frontierCandidates.sort((a,b) => (b.scores?.[primaryObj] || 0) - (a.scores?.[primaryObj] || 0))[0];
            if (bestCandidate) {
                const bestScore = bestCandidate.scores?.[primaryObj] || seedScore;
                const imp = seedScore > 0 ? ((bestScore - seedScore) / seedScore) * 100 : 0;
                kpiImprovement.textContent = `+${imp.toFixed(1)}%`;
                kpiImprovementSub.textContent = `${primaryObj}: ${seedScore.toFixed(2)} → ${bestScore.toFixed(2)}`;

                // Update playground prompt preview
                compOptPromptText.textContent = bestCandidate.prompt_text;
            }
        }

        renderFrontierTable();
        drawParetoChart();
    }

    // Render Frontier Table
    function renderFrontierTable() {
        if (!frontierCandidates || frontierCandidates.length === 0) {
            frontierTableBody.innerHTML = `<tr><td colspan="7" class="empty-state">No frontier candidates yet.</td></tr>`;
            return;
        }

        frontierTableBody.innerHTML = "";
        frontierCandidates.forEach(c => {
            const tr = document.createElement("tr");
            const scoresStr = Object.entries(c.scores || {})
                .map(([k, v]) => `<strong>${k}:</strong> ${v}`)
                .join(" | ");

            const historyStr = (c.mutation_history && c.mutation_history.length > 0)
                ? c.mutation_history[c.mutation_history.length - 1]
                : "Baseline Seed";

            tr.innerHTML = `
                <td><strong style="color: var(--primary); font-family: var(--font-mono);">${c.candidate_id}</strong></td>
                <td><span class="badge">Gen ${c.generation}</span></td>
                <td>${scoresStr}</td>
                <td>${c.average_latency_ms}ms</td>
                <td>${c.average_token_count}</td>
                <td style="font-size: 0.75rem; color: var(--text-muted); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${historyStr}</td>
                <td>
                    <button class="btn-xs btn-inspect" data-id="${c.candidate_id}">Inspect</button>
                </td>
            `;

            tr.querySelector(".btn-inspect").addEventListener("click", () => {
                openCandidateModal(c);
            });

            frontierTableBody.appendChild(tr);
        });
    }

    // Modal Inspector
    function openCandidateModal(c) {
        modalTitle.textContent = `Candidate Details: ${c.candidate_id} (Generation ${c.generation})`;
        modalPromptText.textContent = c.prompt_text;

        modalScoresGrid.innerHTML = "";
        Object.entries(c.scores || {}).forEach(([k, v]) => {
            const div = document.createElement("div");
            div.className = "kpi-card";
            div.innerHTML = `<span class="kpi-label">${k}</span><div class="kpi-value" style="font-size: 1.2rem;">${v}</div>`;
            modalScoresGrid.appendChild(div);
        });

        modalLineageList.innerHTML = "";
        (c.mutation_history || []).forEach(step => {
            const li = document.createElement("li");
            li.textContent = step;
            modalLineageList.appendChild(li);
        });

        modalTracesList.innerHTML = "";
        (c.trace_summary || []).forEach(t => {
            const div = document.createElement("div");
            div.className = `trace-item ${t.passed ? '' : 'failed'}`;
            const metricsStr = Object.entries(t.metrics || {}).map(([k, v]) => `${k}:${v}`).join(", ");
            div.innerHTML = `
                <div style="display:flex; justify-content:space-between; font-weight:600; margin-bottom:4px;">
                    <span>Sample: ${t.sample_id}</span>
                    <span style="color: ${t.passed ? 'var(--accent-green)' : 'var(--accent-red)'}">${t.passed ? 'PASSED' : 'FAILED'} (${metricsStr})</span>
                </div>
                ${t.failure_critique ? `<div style="color:var(--accent-red); margin-bottom:4px;"><strong>Critique:</strong> ${t.failure_critique}</div>` : ''}
                <div style="color:var(--text-muted); font-size:0.75rem;"><strong>Output:</strong> ${t.output_snippet}...</div>
            `;
            modalTracesList.appendChild(div);
        });

        modal.classList.remove("hidden");
    }

    // Modal Events
    btnCloseModal.addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.add("hidden");
    });
    btnCopyPrompt.addEventListener("click", () => {
        navigator.clipboard.writeText(modalPromptText.textContent);
        btnCopyPrompt.textContent = "✓ Copied!";
        setTimeout(() => btnCopyPrompt.textContent = "📋 Copy Prompt", 2000);
    });

    // 3. Pareto Frontier Chart (Canvas 2D)
    function setupCanvas() {
        const resize = () => {
            const rect = paretoCanvas.parentElement.getBoundingClientRect();
            paretoCanvas.width = rect.width - 20;
            paretoCanvas.height = 280;
            drawParetoChart();
        };
        window.addEventListener("resize", resize);
        resize();
    }

    function drawParetoChart() {
        const ctx = paretoCanvas.getContext("2d");
        const w = paretoCanvas.width;
        const h = paretoCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const pad = { top: 30, right: 30, bottom: 40, left: 60 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        // Axes Lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.lineTo(pad.left + plotW, pad.top + plotH);
        ctx.stroke();

        // Grid lines & Axis Labels
        ctx.fillStyle = "#64748b";
        ctx.font = "10px JetBrains Mono";
        ctx.textAlign = "right";

        for (let i = 0; i <= 5; i++) {
            const val = (i / 5).toFixed(1);
            const y = pad.top + plotH - (i / 5) * plotH;
            ctx.fillText(val, pad.left - 8, y + 4);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(pad.left + plotW, y);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
            ctx.stroke();
        }

        ctx.textAlign = "center";
        for (let i = 0; i <= 5; i++) {
            const val = (i / 5).toFixed(1);
            const x = pad.left + (i / 5) * plotW;
            ctx.fillText(val, x, pad.top + plotH + 18);
        }

        // Labels
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Plus Jakarta Sans";
        ctx.fillText("Primary Objective Score (Accuracy / F1)", pad.left + plotW / 2, h - 8);

        ctx.save();
        ctx.translate(14, pad.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Secondary Objective (Token Efficiency / Schema)", 0, 0);
        ctx.restore();

        if (allCandidates.length === 0) return;

        // Extract primary and secondary objective keys
        const sampleScores = allCandidates[0].scores || {};
        const keys = Object.keys(sampleScores);
        const xKey = keys[0] || "accuracy";
        const yKey = keys[1] || (keys.length > 2 ? keys[2] : "token_efficiency");

        // Sort Pareto frontier to draw line
        const sortedFrontier = [...frontierCandidates].sort((a, b) => {
            const xA = a.scores?.[xKey] || 0;
            const xB = b.scores?.[xKey] || 0;
            return xA - xB;
        });

        // Draw Frontier Connection Curve
        if (sortedFrontier.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);

            sortedFrontier.forEach((c, idx) => {
                const xVal = Math.min(1.0, Math.max(0.0, c.scores?.[xKey] || 0));
                const yVal = Math.min(1.0, Math.max(0.0, c.scores?.[yKey] || 0));
                const cx = pad.left + xVal * plotW;
                const cy = pad.top + plotH - yVal * plotH;
                if (idx === 0) ctx.moveTo(cx, cy);
                else ctx.lineTo(cx, cy);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Plot All Candidates
        allCandidates.forEach(c => {
            const xVal = Math.min(1.0, Math.max(0.0, c.scores?.[xKey] || 0));
            const yVal = Math.min(1.0, Math.max(0.0, c.scores?.[yKey] || 0));
            const cx = pad.left + xVal * plotW;
            const cy = pad.top + plotH - yVal * plotH;

            const isSeed = c.generation === 0 && c.candidate_id.includes("seed");
            const isFrontier = frontierCandidates.some(fc => fc.candidate_id === c.candidate_id);

            ctx.beginPath();
            if (isSeed) {
                ctx.fillStyle = "#f59e0b";
                ctx.arc(cx, cy, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            } else if (isFrontier) {
                ctx.fillStyle = "#10b981";
                ctx.arc(cx, cy, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "rgba(16, 185, 129, 0.5)";
                ctx.lineWidth = 4;
                ctx.stroke();
            } else {
                ctx.fillStyle = "#475569";
                ctx.arc(cx, cy, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }

    // 4. Start & Stop Optimization
    btnStartOpt.addEventListener("click", async () => {
        const payload = {
            example_id: exampleSelect.value,
            task_model: taskModelSelect.value,
            reflector_model: reflectorModelSelect.value,
            population_size: parseInt(paramPopSize.value),
            generations: parseInt(paramGenerations.value),
            mutation_rate: parseFloat(paramMutationRate.value),
            crossover_rate: parseFloat(paramCrossoverRate.value)
        };

        if (exampleSelect.value === "custom") {
            payload.custom_task_description = customTaskDesc.value;
            payload.custom_initial_prompt = customSeedPrompt.value;
        }

        btnStartOpt.classList.add("hidden");
        btnStopOpt.classList.remove("hidden");
        optStateBadge.className = "status-badge opt-running";
        optStateText.textContent = "RUNNING";
        isOptimizing = true;

        allCandidates = [];
        frontierCandidates = [];
        sampleEvalCount = 0;
        kpiCandidates.textContent = "0";
        kpiFrontier.textContent = "0";
        kpiImprovement.textContent = "0.0%";
        if (kpiImprovementSub) kpiImprovementSub.textContent = "vs baseline seed";
        frontierTableBody.innerHTML = `<tr><td colspan="7" class="empty-state">Optimization started. Waiting for first rollout...</td></tr>`;

        try {
            const res = await fetch("/api/optimize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) {
                alert(`Error: ${data.detail}`);
                resetRunControls();
            }
        } catch (e) {
            alert(`Error starting run: ${e}`);
            resetRunControls();
        }
    });

    btnStopOpt.addEventListener("click", async () => {
        try {
            await fetch("/api/stop", { method: "POST" });
            logTerminal("WARNING", "Stop signal sent to optimizer.");
        } catch (e) {
            console.error("Stop error:", e);
        }
    });

    function resetRunControls() {
        btnStartOpt.classList.remove("hidden");
        btnStopOpt.classList.add("hidden");
        optStateBadge.className = "status-badge opt-idle";
        optStateText.textContent = "IDLE";
        isOptimizing = false;
    }

    function onOptimizationFinished(data) {
        resetRunControls();
        optStateBadge.className = "status-badge opt-completed";
        optStateText.textContent = "COMPLETED";
        logTerminal("INFO", `🎉 Optimization finished successfully in ${data.duration_seconds}s. Total candidates: ${data.total_candidates}`);
    }

    function onOptimizationError(data) {
        resetRunControls();
        logTerminal("ERROR", `✖ Optimization error: ${data.error}`);
    }

    // 5. Playground Testing
    btnRunPlayground.addEventListener("click", async () => {
        const testInput = playgroundInput.value.trim();
        if (!testInput) {
            alert("Please enter a test input.");
            return;
        }

        const initPrompt = compInitPromptText.textContent.trim();
        const optPrompt = compOptPromptText.textContent.trim();

        btnRunPlayground.disabled = true;
        btnRunPlayground.textContent = "Testing...";
        compInitOutput.textContent = "Running baseline model rollout...";
        compOptOutput.textContent = "Running optimized model rollout...";

        try {
            const res = await fetch("/api/playground/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    task_model: taskModelSelect.value,
                    initial_prompt: initPrompt,
                    optimized_prompt: optPrompt,
                    test_input: testInput
                })
            });
            const data = await res.json();

            compInitOutput.textContent = data.initial.output || "(empty response)";
            compOptOutput.textContent = data.optimized.output || "(empty response)";

            initMetrics.innerHTML = `<span>Latency: ${data.initial.latency_ms}ms</span><span>Tokens: ${data.initial.tokens}</span>`;
            optMetrics.innerHTML = `<span>Latency: ${data.optimized.latency_ms}ms</span><span>Tokens: ${data.optimized.tokens}</span>`;
        } catch (e) {
            compInitOutput.textContent = `Error: ${e}`;
            compOptOutput.textContent = `Error: ${e}`;
        } finally {
            btnRunPlayground.disabled = false;
            btnRunPlayground.innerHTML = "<span>▶</span> Test Both Prompts Side-by-Side";
        }
    });

    // Pills
    [pill1, pill2, pill3].forEach(p => {
        p.addEventListener("click", () => {
            if (p.dataset.val) {
                playgroundInput.value = p.dataset.val;
            }
        });
    });

    // 6. Terminal Logging
    function logTerminal(level, message) {
        const entry = document.createElement("div");
        entry.className = `log-entry log-${level}`;
        const time = new Date().toLocaleTimeString();
        entry.innerHTML = `
            <span class="log-time">[${time}]</span>
            <span class="log-badge badge-${level}">${level}</span>
            <span class="log-msg">${escapeHtml(message)}</span>
        `;
        terminalLogs.appendChild(entry);
        if (chkAutoscroll.checked) {
            terminalLogs.scrollTop = terminalLogs.scrollHeight;
        }
    }

    btnClearLogs.addEventListener("click", () => {
        terminalLogs.innerHTML = "";
    });

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function setupEventListeners() {
        exampleSelect.addEventListener("change", updateSelectedExample);
        btnRefreshModels.addEventListener("click", async () => {
            btnRefreshModels.style.transform = "rotate(360deg)";
            await checkHealth();
            await loadModels();
            setTimeout(() => btnRefreshModels.style.transform = "none", 500);
        });
    }

    init();
});
