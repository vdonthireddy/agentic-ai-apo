#!/usr/bin/env bash
set -e

# ==============================================================================
# GEPA Automatic Prompt Optimizer - Docker Container Restart Script
# Service Port: 18435
# Ollama Gateway: http://host.docker.internal:11434
# ==============================================================================

PORT=18435
CONTAINER_NAME="gepa-prompt-optimizer"
HEALTH_URL="http://localhost:${PORT}/api/health"

echo "============================================================"
echo "🧬 Restarting GEPA Automatic Prompt Optimizer (Port ${PORT})"
echo "============================================================"

# 1. Verify Docker Daemon
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker daemon is not running. Please start Docker Desktop and retry."
    exit 1
fi
echo "✓ Docker daemon is active."

# 2. Check Host Ollama Status
echo -n "Checking Host Ollama status (http://localhost:11434)... "
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✓ Online"
else
    echo "⚠ Warning: Local Ollama is not responding at http://localhost:11434."
    echo "  (Make sure to run 'ollama serve' so local models can be utilized)"
fi

# 3. Stop and Remove Existing Container / Services
echo "Stopping existing GEPA container if running..."
if docker compose version >/dev/null 2>&1; then
    docker compose down --remove-orphans 2>/dev/null || true
else
    docker-compose down --remove-orphans 2>/dev/null || true
fi

# Also check for any standalone container with the same name
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Removing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
fi

# 4. Build and Start Container
echo "Building and launching GEPA container on port ${PORT}..."
if docker compose version >/dev/null 2>&1; then
    docker compose up -d --build
else
    docker-compose up -d --build
fi

# 5. Wait for Healthcheck
echo -n "Waiting for GEPA server to become healthy on port ${PORT}..."
MAX_ATTEMPTS=30
ATTEMPT=0
HEALTHY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    if curl -s -f "${HEALTH_URL}" >/dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

if [ "$HEALTHY" = true ]; then
    echo "============================================================"
    echo "🎉 GEPA Prompt Optimizer is LIVE and HEALTHY!"
    echo "🌐 Access Web UI:     http://localhost:${PORT}"
    echo "🩺 Health Endpoint:   http://localhost:${PORT}/api/health"
    echo "============================================================"
    curl -s "${HEALTH_URL}" | python3 -m json.tool 2>/dev/null || curl -s "${HEALTH_URL}"
    echo ""
else
    echo "❌ Error: Timed out waiting for ${HEALTH_URL}"
    echo "Displaying container logs for troubleshooting:"
    docker logs --tail 50 "${CONTAINER_NAME}"
    exit 1
fi
