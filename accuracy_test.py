"""
Accuracy Test Suite for BhaluRouter
Tests all 8 task categories with known expected answers.
Reports per-category and overall accuracy.
"""

import os
import json
import re
import sys
import time

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ.setdefault("FIREWORKS_API_KEY", "fw_MNn3ToL3W4niBx4YR1ftUx")

from router import HybridRouter

# ── Test cases: (task_id, prompt, expected_keywords_or_answer, category_label) ──
# Each test has a list of "must contain" keywords/patterns that a correct answer should have.

TEST_CASES = [
    # ═══════════════════════ TEXT SUMMARISATION ═══════════════════════
    {
        "task_id": "acc-sum-1",
        "prompt": "Summarise the following text in one sentence: 'Artificial intelligence has transformed healthcare by enabling early disease detection, personalized treatment plans, and efficient drug discovery processes.'",
        "expected_keywords": ["artificial intelligence", "healthcare"],
        "category": "Text Summarisation"
    },
    {
        "task_id": "acc-sum-2",
        "prompt": "Summarise the following text in one sentence: 'The Great Wall of China, stretching over 13,000 miles, was built over centuries to protect Chinese states from invasions by nomadic groups from the north.'",
        "expected_keywords": ["wall", "china"],
        "category": "Text Summarisation"
    },
    {
        "task_id": "acc-sum-3",
        "prompt": "Summarise in one sentence: 'Electric vehicles are gaining popularity worldwide due to lower operating costs, reduced emissions, and advances in battery technology that extend driving range.'",
        "expected_keywords": ["electric", "vehicle"],
        "category": "Text Summarisation"
    },

    # ═══════════════════════ SENTIMENT CLASSIFICATION ═══════════════════════
    {
        "task_id": "acc-sent-1",
        "prompt": "Classify the sentiment of the following review: 'This product is absolutely amazing! Best purchase I have ever made.'",
        "expected_keywords": ["positive"],
        "category": "Sentiment Classification"
    },
    {
        "task_id": "acc-sent-2",
        "prompt": "Classify the sentiment: 'The service was terrible and the food was awful. I will never come back.'",
        "expected_keywords": ["negative"],
        "category": "Sentiment Classification"
    },
    {
        "task_id": "acc-sent-3",
        "prompt": "Classify the sentiment of: 'The weather today is 72 degrees Fahrenheit with partly cloudy skies.'",
        "expected_keywords": ["neutral"],
        "category": "Sentiment Classification"
    },

    # ═══════════════════════ NAMED ENTITY RECOGNITION ═══════════════════════
    {
        "task_id": "acc-ner-1",
        "prompt": "Extract all named entities from: 'On March 15, 2024, Dr. Sarah Chen presented her findings at Microsoft headquarters in Redmond.'",
        "expected_keywords": ["sarah chen", "microsoft", "march"],
        "category": "Named Entity Recognition"
    },
    {
        "task_id": "acc-ner-2",
        "prompt": "Identify all persons, organizations, locations, and dates from: 'Prof. James Wilson joined Google in January 2023 at their office in Mountain View, California.'",
        "expected_keywords": ["james wilson", "google"],
        "category": "Named Entity Recognition"
    },

    # ═══════════════════════ MATHEMATICAL REASONING ═══════════════════════
    {
        "task_id": "acc-math-1",
        "prompt": "If x + 15 = 45, what is the exact numerical value of 3 * x?",
        "expected_keywords": ["90"],
        "category": "Mathematical Reasoning"
    },
    {
        "task_id": "acc-math-2",
        "prompt": "A store sells an item for $200. If they offer a 25% discount, what is the final sale price?",
        "expected_keywords": ["150"],
        "category": "Mathematical Reasoning"
    },
    {
        "task_id": "acc-math-3",
        "prompt": "Calculate: If a car travels at 60 miles per hour for 2.5 hours, how many miles does it travel in total?",
        "expected_keywords": ["150"],
        "category": "Mathematical Reasoning"
    },

    # ═══════════════════════ LOGICAL DEDUCTIVE REASONING ═══════════════════════
    {
        "task_id": "acc-logic-1",
        "prompt": "Alice, Bob, and Carol are standing in a line. Alice is not first. Carol is not last. Bob is not next to Alice. What is the order from first to last?",
        "expected_keywords": ["carol", "bob", "alice"],
        "category": "Logical Deductive Reasoning"
    },
    {
        "task_id": "acc-logic-2",
        "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "expected_keywords": ["no", "cannot"],
        "category": "Logical Deductive Reasoning"
    },

    # ═══════════════════════ CODE DEBUGGING ═══════════════════════
    {
        "task_id": "acc-debug-1",
        "prompt": "Debug this Python code and explain the error:\n```python\ndef add_numbers(a, b):\n    return a + b\n\nresult = add_numbers(5)\nprint(result)\n```",
        "expected_keywords": ["argument", "missing"],
        "category": "Code Debugging"
    },
    {
        "task_id": "acc-debug-2",
        "prompt": "Find the bug in this code:\n```python\nmy_list = [1, 2, 3, 4, 5]\nfor i in range(len(my_list)):\n    if my_list[i] % 2 == 0:\n        my_list.remove(my_list[i])\n```",
        "expected_keywords": ["modif", "remov"],  # modifying/removing during iteration
        "category": "Code Debugging"
    },

    # ═══════════════════════ CODE GENERATION ═══════════════════════
    {
        "task_id": "acc-code-1",
        "prompt": "Write a Python function called `is_palindrome` that takes a string and returns True if the string is a palindrome (reads the same forwards and backwards), False otherwise.",
        "expected_keywords": ["def is_palindrome", "return"],
        "category": "Code Generation"
    },
    {
        "task_id": "acc-code-2",
        "prompt": "Write a Python function called `fibonacci` that takes an integer n and returns the nth Fibonacci number. Use iteration, not recursion.",
        "expected_keywords": ["def fibonacci", "return"],
        "category": "Code Generation"
    },

    # ═══════════════════════ FACTUAL KNOWLEDGE ═══════════════════════
    {
        "task_id": "acc-fact-1",
        "prompt": "What is the capital of France?",
        "expected_keywords": ["paris"],
        "category": "Factual Knowledge"
    },
    {
        "task_id": "acc-fact-2",
        "prompt": "What is photosynthesis?",
        "expected_keywords": ["light", "plant"],
        "category": "Factual Knowledge"
    },
]


def check_answer(answer: str, expected_keywords: list) -> bool:
    """Check if the answer contains all expected keywords (case-insensitive)."""
    answer_lower = answer.lower()
    for kw in expected_keywords:
        if kw.lower() not in answer_lower:
            return False
    return True


def main():
    print("=" * 70)
    print("🐻 BhaluRouter Accuracy Test Suite")
    print("=" * 70)

    router = HybridRouter()
    results = []
    category_stats = {}

    for i, tc in enumerate(TEST_CASES):
        task_id = tc["task_id"]
        prompt = tc["prompt"]
        expected = tc["expected_keywords"]
        cat = tc["category"]

        print(f"\n[{i+1}/{len(TEST_CASES)}] {cat} | {task_id}")
        print(f"  Prompt: {prompt[:80]}...")

        try:
            answer, metadata = router.route_and_execute(task_id, prompt)
            route = metadata.get("route", "unknown")
            tokens = metadata.get("token_usage", {}).get("total_tokens", 0)
        except Exception as e:
            answer = f"ERROR: {e}"
            route = "error"
            tokens = 0

        passed = check_answer(answer, expected)
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"  Route: {route} | Tokens: {tokens}")
        print(f"  Answer: {answer[:120]}...")
        print(f"  Expected keywords: {expected}")
        print(f"  Result: {status}")

        results.append({
            "task_id": task_id,
            "category": cat,
            "route": route,
            "tokens": tokens,
            "passed": passed,
            "answer_preview": answer[:200]
        })

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "tokens": 0}
        category_stats[cat]["total"] += 1
        if passed:
            category_stats[cat]["passed"] += 1
        category_stats[cat]["tokens"] += tokens

        # Pause between queries to avoid Fireworks API rate limits
        if i < len(TEST_CASES) - 1:
            time.sleep(3)

    # ── Summary ──
    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    total_tokens = sum(r["tokens"] for r in results)

    print("\n" + "=" * 70)
    print("📊 ACCURACY RESULTS BY CATEGORY")
    print("=" * 70)
    print(f"{'Category':<30} {'Passed':<10} {'Accuracy':<12} {'Tokens':<10}")
    print("-" * 70)

    for cat, stats in sorted(category_stats.items()):
        acc = stats["passed"] / stats["total"] * 100
        color = "🟢" if acc == 100 else ("🟡" if acc >= 50 else "🔴")
        print(f"{color} {cat:<28} {stats['passed']}/{stats['total']:<8} {acc:>6.1f}%      {stats['tokens']}")

    print("-" * 70)
    overall_acc = total_passed / total * 100
    print(f"{'OVERALL':<30} {total_passed}/{total:<8} {overall_acc:>6.1f}%      {total_tokens} total tokens")
    print("=" * 70)

    # ── Failures detail ──
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n⚠️  {len(failures)} FAILED TESTS:")
        for f in failures:
            print(f"  - {f['task_id']} ({f['category']}): {f['answer_preview'][:100]}...")
    else:
        print("\n🎉 ALL TESTS PASSED! 100% accuracy across all 8 categories.")

    print(f"\n💰 Total Fireworks tokens used: {total_tokens}")
    print(f"🐻 Local zero-token solves: {sum(1 for r in results if r['tokens'] == 0)}/{total}")


if __name__ == "__main__":
    main()
