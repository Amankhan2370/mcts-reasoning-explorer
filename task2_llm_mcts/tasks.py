"""
Problem definitions for LLM-MCTS experiments.
"""

from typing import List, Dict
import json


class MathProblem:
    """A math word problem."""
    
    def __init__(self, question: str, answer: str, category: str = "arithmetic"):
        self.question = question
        self.answer = answer
        self.category = category
    
    def __repr__(self):
        return f"MathProblem({self.category}): {self.question[:50]}..."


def get_math_problems() -> List[MathProblem]:
    """
    Get a set of math word problems for testing.
    Curated set with known correct answers.
    """
    problems = [
        # Simple arithmetic
        MathProblem(
            "Sarah has 8 cookies. She eats 3 cookies. How many cookies does she have left?",
            "5",
            "simple_arithmetic"
        ),
        MathProblem(
            "A store has 15 red shirts and 23 blue shirts. How many shirts in total?",
            "38",
            "simple_arithmetic"
        ),
        MathProblem(
            "Tom saved $45 in January, $32 in February, and $28 in March. How much did he save in total?",
            "105",
            "simple_arithmetic"
        ),
        
        # Multi-step arithmetic
        MathProblem(
            "Emma has 12 apples. She buys 5 more, then gives 4 to her friend. How many apples does she have now?",
            "13",
            "multi_step"
        ),
        MathProblem(
            "A classroom has 25 students. 8 students leave for lunch, then 3 new students join. How many students are in the classroom?",
            "20",
            "multi_step"
        ),
        MathProblem(
            "John starts with $50. He spends $18 on lunch and $12 on a book. How much money does he have left?",
            "20",
            "multi_step"
        ),
        
        # Multiplication/Division
        MathProblem(
            "There are 4 boxes with 6 pencils in each box. How many pencils in total?",
            "24",
            "multiplication"
        ),
        MathProblem(
            "A pizza is cut into 8 slices. If 3 people share it equally, how many slices does each person get? Round to nearest whole number.",
            "3",
            "division"
        ),
        MathProblem(
            "A car travels 60 miles per hour. How far does it travel in 3 hours?",
            "180",
            "multiplication"
        ),
        
        # Comparison/Reasoning
        MathProblem(
            "Lisa is 12 years old. Her brother is 3 years younger. How old is her brother?",
            "9",
            "comparison"
        ),
        MathProblem(
            "A rope is 45 feet long. Another rope is 28 feet long. How much longer is the first rope?",
            "17",
            "comparison"
        ),
        
        # Slightly harder
        MathProblem(
            "A store sells notebooks for $3 each. If you buy 5 notebooks and pay with a $20 bill, how much change do you get?",
            "5",
            "multi_step_with_multiplication"
        ),
        MathProblem(
            "There are 36 students going on a field trip. Each bus holds 9 students. How many buses are needed?",
            "4",
            "division_with_rounding"
        ),
    ]
    
    return problems


def get_problems_by_difficulty(difficulty: str = "easy") -> List[MathProblem]:
    """Get problems filtered by difficulty."""
    all_problems = get_math_problems()
    
    if difficulty == "easy":
        categories = ["simple_arithmetic"]
    elif difficulty == "medium":
        categories = ["multi_step", "multiplication", "division", "comparison"]
    elif difficulty == "hard":
        categories = ["multi_step_with_multiplication", "division_with_rounding"]
    else:
        return all_problems
    
    return [p for p in all_problems if p.category in categories]


def load_gsm8k_sample(n: int = 10) -> List[MathProblem]:
    """
    Load sample problems from GSM8K dataset format.
    (Placeholder - would load from actual dataset file)
    """
    # For now, return subset of curated problems
    return get_math_problems()[:n]


if __name__ == "__main__":
    # Test problem loading
    problems = get_math_problems()
    print(f"Loaded {len(problems)} problems")
    
    for i, p in enumerate(problems[:3]):
        print(f"\nProblem {i+1}:")
        print(f"  Q: {p.question}")
        print(f"  A: {p.answer}")
        print(f"  Category: {p.category}")