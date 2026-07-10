"""
BhaluRouter Backend API Server
Lightweight HTTP server that exposes the Python HybridRouter to the Next.js frontend.
Runs on port 8000. The Next.js /api/route endpoint proxies requests here.
"""

import os
import sys
import json
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ.setdefault("FIREWORKS_API_KEY", "fw_MNn3ToL3W4niBx4YR1ftUx")

from router import HybridRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("BackendAPI")

# Initialize router once at startup
router = HybridRouter()
task_counter = 0


class RouteHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global task_counter

        if self.path == "/api/route":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            prompt = data.get("prompt", "").strip()
            category_override = data.get("category", "Auto Detect")

            if not prompt:
                self._send_json(400, {"error": "No prompt provided"})
                return

            task_counter += 1
            task_id = f"web-{task_counter}"

            logger.info(f"[Web Request] Task {task_id}: {prompt[:80]}...")

            start_time = time.time()
            try:
                answer, metadata = router.route_and_execute(task_id, prompt, category_override)
                elapsed_ms = int((time.time() - start_time) * 1000)

                route_type = metadata.get("route", "remote_fireworks")
                tokens = metadata.get("token_usage", {}).get("total_tokens", 0)
                category = metadata.get("category", "unknown")
                complexity = metadata.get("complexity", "medium")

                # Map exact capability route and model name
                if route_type in ("deterministic", "local_python", "python_math", "relation_rules", "local_deterministic", "unit_converter", "json_validator", "kinship_rule_engine", "encyclopedic_ast"):
                    out_route = "deterministic"
                    out_model = metadata.get("model_used", "Deterministic Rule/AST Engine (Local Zero-Token)")
                elif route_type in ("local_llm", "E4B_local", "E4B_local_verified"):
                    out_route = "local_llm"
                    out_model = metadata.get("model_used", "gemma-4-e4b-it (Local Gemma 4B Zero-Token)")
                else:
                    out_route = "remote_fireworks"
                    out_model = metadata.get("model_used", "accounts/fireworks/models/minimax-m3")

                # Estimate cost (minimax-m3 pricing ~$0.01/1K tokens)
                cost = tokens * 0.00001

                response = {
                    "route": out_route,
                    "model": out_model,
                    "confidence": metadata.get("confidence", 0.95),
                    "complexity": complexity,
                    "tokens": tokens,
                    "latency": elapsed_ms,
                    "cost": cost,
                    "reason": metadata.get("reason", f"Capability Router decision: {out_route}"),
                    "deterministic_rule": metadata.get("deterministic_rule", "None"),
                    "token_savings": metadata.get("token_savings", 0),
                    "answer": answer,
                    "category": category
                }

                logger.info(f"[Web Response] Task {task_id}: Route={out_route} | Tokens={tokens} | Time={elapsed_ms}ms")
                self._send_json(200, response)

            except Exception as e:
                logger.error(f"[Web Error] Task {task_id}: {e}")
                self._send_json(500, {"error": str(e)})

        elif self.path == "/api/health":
            self._send_json(200, {"status": "ok", "tasks_processed": task_counter})

        else:
            self._send_json(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok", "tasks_processed": task_counter})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress default BaseHTTPRequestHandler access logs (we have our own)
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", 8000)))
    server = HTTPServer(("0.0.0.0", port), RouteHandler)
    logger.info(f"BhaluRouter Backend API running on http://localhost:{port}")
    logger.info(f"  POST /api/route  -> Route a task prompt")
    logger.info(f"  GET  /api/health -> Health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.server_close()
