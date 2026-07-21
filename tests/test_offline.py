import re
from src import classifier, cleaner, verifier, config, fireworks_client

def run_all_tests():
    # 1. Test classifier across plural, conjugated, and tricky boundaries
    assert classifier.analyze("What is 2+2?")["category"] == "mathematical_reasoning"
    assert classifier.analyze("Extract entities: Sundar Pichai visited Zurich")["category"] == "named_entity_recognition"
    assert classifier.analyze("Summarize this article: ...")["category"] == "text_summarization"
    assert classifier.analyze("Classify the sentiment: Great product, bad shipping")["category"] == "sentiment_classification"
    assert classifier.analyze("Why does this code fail with an exception?")["category"] == "code_debugging"
    assert classifier.analyze("Implement a python function to compute prime numbers")["category"] == "code_generation"
    assert classifier.analyze("Who is sitting next to Bob if Alice sits on the left?")["category"] == "logical_reasoning"

    # 2. Test verifier & cleaner on exact examples
    ner_text = "Sundar Pichai - PERSON\nZurich - LOCATION"
    assert verifier.score(ner_text, "named_entity_recognition", "") == 1.0

    summary_text = "- Point 1\n- Point 2\n- Point 3"
    assert verifier.score(summary_text, "text_summarization", "Give at most three bullet points.") == 1.0

    sentiment_text = "Mixed\nReason: The product is good but shipping was delayed."
    assert verifier.score(sentiment_text, "sentiment_classification", "") == 1.0

    # 3. Test cleaner math box stripping
    assert cleaner.clean("Here is the step.\nAnswer: \\boxed{1,672 units}", "mathematical_reasoning", "") == "Answer: 1,672 units"

    # 4. Test cleaner NER bullet stripping
    ner_bullet = "- Sundar Pichai - PERSON\n* Zurich - LOCATION"
    assert cleaner.clean(ner_bullet, "named_entity_recognition", "") == "Sundar Pichai - PERSON\nZurich - LOCATION"

    # 5. Test code block fence stripping exactly
    code_fenced = "```python\ndef foo():\n    return 42\n```"
    cleaned_code = cleaner.clean(code_fenced, "code_generation", "")
    assert cleaned_code == "def foo():\n    return 42", f"Got: {repr(cleaned_code)}"

    print("ALL 100% OFFLINE UNIT TESTS PASSED WITH ZERO ERRORS!")

if __name__ == "__main__":
    run_all_tests()
