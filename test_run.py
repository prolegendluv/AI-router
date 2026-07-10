"""
Quick End-to-End Test Runner for Container Entrypoint (main.py)
Creates sample_input/tasks.json, runs main.py, and verifies sample_output/results.json.
"""

import os
import json
import subprocess
import sys

def prepare_and_test():
    os.makedirs("sample_input", exist_ok=True)
    os.makedirs("sample_output", exist_ok=True)
    
    sample_tasks = [
        {
            "task_id": "t1",
            "prompt": "Summarise the following text in one sentence: 'The AMD Ryzen and Epyc processors have revolutionized modern cloud computing and AI workloads with unprecedented core density and energy efficiency.'"
        },
        {
            "task_id": "t2",
            "prompt": "If x + 15 = 45, what is the exact numerical value of 3 * x?"
        },
        {
            "task_id": "t3",
            "prompt": "Classify the sentiment of: 'This router works extremely well and saves thousands of API tokens without sacrificing accuracy!'"
        }
    ]
    
    input_file = "sample_input/tasks.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sample_tasks, f, indent=2)
        
    print(f"[TestRunner] Created {input_file} with {len(sample_tasks)} tasks.")
    print("[TestRunner] Executing main.py...")
    
    env = os.environ.copy()
    # Use real FIREWORKS_API_KEY if passed in environment, otherwise fallback to mock_test_key
    env["FIREWORKS_API_KEY"] = os.environ.get("FIREWORKS_API_KEY", "mock_test_key")
    env["FIREWORKS_BASE_URL"] = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    env["ALLOWED_MODELS"] = os.environ.get("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/llama-v3p1-8b-instruct")
    
    ret = subprocess.run([sys.executable, "main.py"], env=env)
    if ret.returncode == 0:
        output_file = "sample_output/results.json"
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            print(f"[TestRunner] SUCCESS! Output verified at {output_file} ({len(res)} results generated).")
        else:
            print(f"[TestRunner] FAILED: Output file {output_file} was not created.")
    else:
        print(f"[TestRunner] FAILED: main.py exited with return code {ret.returncode}")

if __name__ == "__main__":
    prepare_and_test()
