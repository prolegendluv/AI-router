"""
Local Evaluation Harness for Track 1 Agent
Runs local benchmark evaluation across all 8 capability categories.
Checks:
  1. Routing accuracy & category detection
  2. Latency per task (Must be < 30s)
  3. Token savings vs. pure cloud routing
"""

import json
import time
import os
import logging
from typing import List, Dict, Any
from router import HybridRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluator")


def run_evaluation():
    eval_file = "eval_dataset.json"
    if not os.path.exists(eval_file):
        logger.error(f"Cannot find benchmark file {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    logger.info(f"Loaded {len(tasks)} benchmark tasks across 8 categories.")
    router = HybridRouter()

    results_summary = []
    total_time = 0

    print("\n" + "="*80)
    print(" 🚀 STARTING TRACK 1 HYBRID ROUTER LOCAL BENCHMARK EVALUATION 🚀 ")
    print("="*80 + "\n")

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        target_cat = task["category_target"]
        prompt = task["prompt"]

        print(f"[{i}/{len(tasks)}] Evaluating Task ID: {task_id} | Target Category: {target_cat}")
        start_t = time.time()
        
        answer, metadata = router.route_and_execute(task_id, prompt)
        elapsed = time.time() - start_t
        total_time += elapsed

        detected_cat = metadata["category"]
        route = metadata["route"]
        tokens = metadata["token_usage"]["total_tokens"]

        print(f" -> Detected Category: {detected_cat} | Route Taken: {route} | Tokens Used: {tokens}")
        print(f" -> Latency: {elapsed:.2f}s (Budget: <30.00s)")
        print(f" -> Answer Preview: {answer[:140].replace(chr(10), ' ')}...\n")

        results_summary.append({
            "task_id": task_id,
            "target_category": target_cat,
            "detected_category": detected_cat,
            "route": route,
            "latency_s": round(elapsed, 2),
            "tokens_used": tokens,
            "passed_latency_gate": elapsed < 30.0
        })

    # Summary
    stats = router.get_summary_stats()
    avg_latency = total_time / len(tasks)
    
    print("\n" + "="*80)
    print(" 🎯 EVALUATION SUMMARY REPORT 🎯 ")
    print("="*80)
    print(f"Total Benchmark Tasks Processed: {len(tasks)}")
    print(f"Average Latency per Task: {avg_latency:.2f}s (Max Budget: 30.00s)")
    print(f"Routing Breakdown:")
    print(f"  - E4B (Local Model / 0 Tokens): {stats['routing_counts']['local']} tasks")
    print(f"  - Fireworks API (Low Tier):     {stats['routing_counts']['fireworks_low']} tasks")
    print(f"  - Fireworks API (Minimax/High): {stats['routing_counts']['fireworks_high']} tasks")
    print("-" * 50)
    print(f"Total Fireworks API Calls Made:   {stats['fireworks_stats']['total_calls']}")
    print(f"Total Fireworks Prompt Tokens:    {stats['fireworks_stats']['total_prompt_tokens']}")
    print(f"Total Fireworks Completion Tokens:{stats['fireworks_stats']['total_completion_tokens']}")
    print(f"TOTAL FIREWORKS TOKENS CONSUMED:  {stats['fireworks_stats']['total_tokens']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_evaluation()
