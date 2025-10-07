"""
Evaluation functions to score LLM reasoning and answers.
"""

import re
from typing import Tuple, Dict


def extract_number(text: str) -> float:
    """
    Extract a number from text.

    Args:
        text: Text potentially containing a number

    Returns:
        Extracted number, or None if not found
    """
    # Remove common words
    text = text.lower().replace("answer:", "").replace("final answer:", "")

    # Try to find numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)

    if numbers:
        return float(numbers[-1])  # Return last number found
    return None


def evaluate_math_answer(problem_answer: str, generated_answer: str) -> Tuple[bool, float]:
    """
    Evaluate if a generated answer is correct.

    Args:
        problem_answer: Correct answer
        generated_answer: LLM generated answer

    Returns:
        (is_correct, reward)
    """
    try:
        correct = float(problem_answer.strip())
        generated = extract_number(generated_answer)

        if generated is None:
            return False, -0.5  # Penalty for no answer

        # Check if close enough (within 1% or exact)
        if abs(correct - generated) < 0.01 * abs(correct) or abs(correct - generated) < 0.5:
            return True, 1.0
        else:
            return False, -1.0

    except (ValueError, TypeError):
        return False, -0.5


def evaluate_reasoning_quality(reasoning: str) -> float:
    """
    Heuristic evaluation of reasoning quality.

    Args:
        reasoning: Chain of reasoning steps

    Returns:
        Quality score between 0 and 1
    """
    score = 0.5  # Baseline

    # Positive indicators
    if len(reasoning) > 20:  # Has some content
        score += 0.1

    if any(word in reasoning.lower() for word in ['first', 'then', 'next', 'finally']):
        score += 0.1  # Structured reasoning

    if any(word in reasoning.lower() for word in ['calculate', 'add', 'subtract', 'multiply', 'divide']):
        score += 0.1  # Math operations mentioned

    # Negative indicators
    if "i don't know" in reasoning.lower() or "cannot" in reasoning.lower():
        score -= 0.2

    if len(reasoning) < 10:  # Too short
        score -= 0.2

    return max(0.0, min(1.0, score))


def evaluate_solution(problem_answer: str,
                      reasoning_chain: str,
                      final_answer: str) -> Dict:
    """
    Complete evaluation of a solution.

    Args:
        problem_answer: Correct answer
        reasoning_chain: Full reasoning process
        final_answer: Final answer from LLM

    Returns:
        Dictionary with evaluation metrics
    """
    is_correct, answer_reward = evaluate_math_answer(
        problem_answer, final_answer)
    reasoning_quality = evaluate_reasoning_quality(reasoning_chain)

    # Combined reward
    if is_correct:
        total_reward = 1.0 + 0.2 * reasoning_quality  # Bonus for good reasoning
    else:
        total_reward = -0.5 + 0.3 * reasoning_quality  # Some credit for process

    return {
        'correct': is_correct,
        'answer_reward': answer_reward,
        'reasoning_quality': reasoning_quality,
        'total_reward': total_reward,
        'extracted_answer': extract_number(final_answer),
        'expected_answer': float(problem_answer)
    }


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluators...")

    # Test number extraction
    test_texts = [
        "The answer is 42",
        "Final answer: 3.14",
        "I calculated 25 apples",
        "No clear answer here"
    ]

    for text in test_texts:
        num = extract_number(text)
        print(f"'{text}' -> {num}")

    # Test answer evaluation
    print("\nTesting answer evaluation:")
    print(evaluate_math_answer("42", "The answer is 42"))
    print(evaluate_math_answer("42", "I think it's 43"))
    print(evaluate_math_answer("42", "No idea"))
