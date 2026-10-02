"""
Ollama HTTP API Client for Local LLM Inference.
Interacts with Ollama's REST endpoints (/api/generate, /api/chat, /api/tags).
Provides automatic fallback detection and error diagnostics.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import httpx
from app.config import settings

logger = logging.getLogger("GEPA.OllamaClient")

class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = httpx.Timeout(60.0, connect=5.0)

    async def check_health(self) -> Dict[str, Any]:
        """Verify connection to Ollama and report reachable status and models."""
        urls_to_try = [self.base_url]
        if "localhost" in self.base_url:
            urls_to_try.append("http://host.docker.internal:11434")
            urls_to_try.append("http://127.0.0.1:11434")
        elif "host.docker.internal" in self.base_url:
            urls_to_try.append("http://localhost:11434")
            urls_to_try.append("http://127.0.0.1:11434")

        for url in urls_to_try:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{url}/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("name") for m in data.get("models", [])]
                        self.base_url = url
                        logger.info(f"Connected to Ollama at {url}. Available models: {models}")
                        return {
                            "connected": True,
                            "url": url,
                            "models": models,
                            "error": None
                        }
            except Exception as e:
                logger.debug(f"Could not connect to {url}: {e}")

        logger.warning(f"Ollama not reachable at {urls_to_try}. Will allow simulation/mock mode if requested.")
        return {
            "connected": False,
            "url": self.base_url,
            "models": [],
            "error": f"Failed to connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
        }

    async def list_models(self) -> List[str]:
        """Fetch list of locally installed models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Error fetching models from {self.base_url}: {e}")
        return []

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Execute completion using local Ollama model.
        Returns response text, token usage, and latency metrics.
        """
        start_time = time.time()
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                elapsed = time.time() - start_time
                if resp.status_code == 200:
                    data = resp.json()
                    response_text = data.get("response", "").strip()
                    eval_count = data.get("eval_count", len(response_text.split()))
                    return {
                        "text": response_text,
                        "token_count": eval_count,
                        "latency_ms": round(elapsed * 1000, 2),
                        "success": True,
                        "error": None
                    }
                else:
                    return {
                        "text": "",
                        "token_count": 0,
                        "latency_ms": round(elapsed * 1000, 2),
                        "success": False,
                        "error": f"Ollama HTTP {resp.status_code}: {resp.text}"
                    }
        except httpx.ConnectError:
            elapsed = time.time() - start_time
            logger.error(f"Ollama connection refused at {self.base_url}")
            return {
                "text": "",
                "token_count": 0,
                "latency_ms": round(elapsed * 1000, 2),
                "success": False,
                "error": f"Connection refused to {self.base_url}. Is Ollama running?"
            }
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error during Ollama inference: {e}")
            return {
                "text": "",
                "token_count": 0,
                "latency_ms": round(elapsed * 1000, 2),
                "success": False,
                "error": str(e)
            }
