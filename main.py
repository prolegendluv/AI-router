"""
Main Entrypoint for AMD Hackathon Track 1 Container
Reads tasks from /input/tasks.json, executes via Hybrid Router, and writes to /output/results.json.
Exits with code 0 on success.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
from router import HybridRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MainEntrypoint")


def run_pipeline():
    start_time = time.time()
    input_path = os.environ.get("INPUT_PATH", "/input/tasks.json")
    output_path = os.environ.get("OUTPUT_PATH", "/output/results.json")
    
    # Check if running locally outside container
    if not os.path.exists(input_path):
        local_input = "sample_input/tasks.json"
        local_output = "sample_output/results.json"
        if os.path.exists(local_input):
            logger.info(f"Container /input/tasks.json not found. Using local input: {local_input}")
            input_path = local_input
            output_path = local_output
        else:
            logger.error(f"Input file not found at {input_path} or {local_input}.")
            sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read tasks
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks: List[Dict[str, Any]] = json.load(f)
        logger.info(f"Successfully loaded {len(tasks)} tasks from {input_path}.")
    except Exception as e:
        logger.error(f"Failed to read/parse input tasks from {input_path}: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        logger.error(f"Expected tasks format to be a JSON array (list), got {type(tasks)}")
        sys.exit(1)

    # Initialize Hybrid Router
    router = HybridRouter()
    results: List[Dict[str, str]] = []

    # Process tasks
    for i, task in enumerate(tasks):
        task_id = str(task.get("task_id", f"t{i+1}"))
        prompt = str(task.get("prompt", "")).strip()
        
        if not prompt:
            logger.warning(f"Task {task_id} has empty prompt. Returning default message.")
            results.append({"task_id": task_id, "answer": "No prompt provided."})
            continue

        logger.info(f"--- Processing Task {i+1}/{len(tasks)} [ID: {task_id}] ---")
        try:
            answer, metadata = router.route_and_execute(task_id, prompt)
            results.append({"task_id": task_id, "answer": answer})
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            results.append({"task_id": task_id, "answer": f"Error during processing: {str(e)}"})

    # Write output to /output/results.json
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Successfully wrote and synced {len(results)} results to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to write output to {output_path}: {e}")
        sys.exit(1)

    # Log summary telemetry
    stats = router.get_summary_stats()
    total_time = time.time() - start_time
    logger.info("=================== PIPELINE SUMMARY ===================")
    logger.info(f"Total Execution Time: {total_time:.2f}s (Average: {total_time/max(len(tasks), 1):.2f}s/task)")
    logger.info(f"Routing Breakdown: {stats['routing_counts']}")
    logger.info(f"Local Zero-Token Calls (`E4B`): {stats['local_calls']}")
    logger.info(f"Fireworks API Calls: {stats['fireworks_stats']['total_calls']}")
    logger.info(
        f"Total Fireworks Tokens Consumed: "
        f"{stats['fireworks_stats']['total_prompt_tokens']} prompt + "
        f"{stats['fireworks_stats']['total_completion_tokens']} completion = "
        f"{stats['fireworks_stats']['total_tokens']} TOTAL TOKENS"
    )
    logger.info("========================================================")
    sys.exit(0)


if __name__ == "__main__":
    run_pipeline()
