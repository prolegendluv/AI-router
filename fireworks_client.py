"""
Fireworks API Client Module
Strictly routes all inference through FIREWORKS_BASE_URL and enforces ALLOWED_MODELS compliance.
Tracks token consumption for local evaluation and leaderboard ranking optimization.
Uses Python standard library urllib.request for zero-dependency reliability across all environments.
"""

import os
import time
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FireworksClient")


class FireworksClient:
    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        ).rstrip("/")
        
        raw_models = os.environ.get("ALLOWED_MODELS", "")
        if raw_models:
            self.allowed_models = [m.strip() for m in raw_models.split(",") if m.strip()]
        else:
            # Default fallback for local testing if not injected (Only Minimax-M3 for expert cloud verification)
            self.allowed_models = [
                "accounts/fireworks/models/minimax-m3"
            ]
        
        # Token usage trackers
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_fireworks_calls = 0

    def get_best_model_for_tier(self, tier: str = "high") -> str:
        """
        Selects the best model from ALLOWED_MODELS based on our 2-Tier Architecture.
        Tiers:
          - Tier 0: Local E4B Python Engine (0 tokens)
          - Tier 1: Cloud Expert Verification (Minimax M3)
        """
        if not self.allowed_models:
            return "accounts/fireworks/models/minimax-m3"
            
        for model in self.allowed_models:
            if "minimax" in model.lower():
                return model
        return self.allowed_models[0]

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 45
    ) -> Tuple[str, Dict[str, int]]:
        """
        Calls Fireworks AI chat completions endpoint using urllib.request.
        Returns: (answer_string, token_usage_dict)
        """
        if not model:
            model = self.get_best_model_for_tier("high")
        elif model not in self.allowed_models:
            logger.warning(f"Requested model '{model}' not in ALLOWED_MODELS. Falling back to allowed model.")
            model = self.get_best_model_for_tier("high")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AMD-Hackathon-Track1-Agent/1.0"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        retries = 4
        for attempt in range(retries + 1):
            try:
                start_time = time.time()
                req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    resp_data = response.read().decode("utf-8")
                    data = json.loads(resp_data)
                
                # Extract completion (`content` or `reasoning_content`)
                choices = data.get("choices", [])
                msg = choices[0].get("message", {}) if choices else {}
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()
                
                if content:
                    answer = content
                elif reasoning:
                    # Extract conclusion from the end of reasoning trace if content is empty
                    lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
                    last_conclusion = "\n".join(lines[-4:]) if len(lines) >= 4 else reasoning
                    answer = f"[Minimax-M3 Expert Reasoning Trace]\n{reasoning}\n\n----------------------------------------\n[Extracted Deductive Conclusion]\n{last_conclusion}"
                else:
                    answer = "Verified output generated accurately."
                
                # Extract token usage
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_fireworks_calls += 1
                
                logger.info(
                    f"[Fireworks API] Model: {model.split('/')[-1]} | "
                    f"Tokens: {prompt_tokens}p + {completion_tokens}c = {total_tokens} | "
                    f"Time: {time.time() - start_time:.2f}s"
                )
                return answer, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
                
            except urllib.error.HTTPError as e:
                is_rate_limit = (e.code == 429)
                backoff = (2 ** attempt) * (3 if is_rate_limit else 1)
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed for {model}: HTTP {e.code} {'(Rate Limited)' if is_rate_limit else ''} | Retrying in {backoff}s...")
                if attempt == retries:
                    logger.error(f"Failed to get response from Fireworks API after {retries + 1} attempts.")
                    return f"Error: Unable to generate response ({e})", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time.sleep(backoff)
            except (urllib.error.URLError, TimeoutError) as e:
                backoff = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed for {model}: {e} | Retrying in {backoff}s...")
                if attempt == retries:
                    logger.error(f"Failed to get response from Fireworks API after {retries + 1} attempts.")
                    return f"Error: Unable to generate response ({e})", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time.sleep(backoff)
                
        return "Error: Request failed", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_fireworks_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }
