"""
MCTS adapted for LLM reasoning.
"""

import random
import math
from typing import List, Dict, Any, Optional
from llm_interface import LLMInterface
from evaluators import evaluate_solution


class ReasoningNode:
    """Node in LLM-MCTS reasoning tree."""
    
    def __init__(self, 
                 state: str,  # Current reasoning chain
                 parent: Optional['ReasoningNode'] = None,
                 action: str = ""):  # Last reasoning step added
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['ReasoningNode'] = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.is_terminal = False
    
    def uct_value(self, c: float = 1.4) -> float:
        """Compute UCT value."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visit_count
        
        if self.parent is None:
            return exploitation
        
        exploration = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration
    
    def best_child(self, c: float = 1.4) -> 'ReasoningNode':
        """Select best child using UCT."""
        return max(self.children, key=lambda child: child.uct_value(c))
    
    def update(self, reward: float):
        """Update statistics."""
        self.visit_count += 1
        self.total_reward += reward
    
    def get_avg_reward(self) -> float:
        """Get average reward."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count


class LLM_MCTS:
    """MCTS for LLM reasoning."""
    
    def __init__(self, 
                 llm: LLMInterface,
                 max_depth: int = 5,
                 num_simulations: int = 10,
                 exploration_constant: float = 1.4):
        self.llm = llm
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.c = exploration_constant
    
    def search(self, problem: str, correct_answer: str) -> Dict:
        """
        Run MCTS to solve a problem.
        
        Args:
            problem: Problem statement
            correct_answer: Correct answer (for evaluation)
        
        Returns:
            Dictionary with results
        """
        # Create root node
        root = ReasoningNode(state="")
        
        # Run simulations
        for i in range(self.num_simulations):
            # Selection + Expansion
            node = self._select_and_expand(root, problem)
            
            # Simulation
            reward = self._simulate(node, problem, correct_answer)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Get best path
        best_path = self._get_best_path(root)
        reasoning_chain = "\n".join([node.action for node in best_path if node.action])
        
        # Get final answer
        final_answer = self.llm.complete_solution(problem, reasoning_chain)
        
        # Evaluate
        evaluation = evaluate_solution(correct_answer, reasoning_chain, final_answer)
        
        return {
            'reasoning_chain': reasoning_chain,
            'final_answer': final_answer,
            'evaluation': evaluation,
            'tree_visits': root.visit_count,
            'best_path_length': len(best_path)
        }
    
    def _select_and_expand(self, node: ReasoningNode, problem: str) -> ReasoningNode:
        """Select and expand node."""
        current = node
        depth = 0
        
        # Selection: traverse to leaf
        while current.children and depth < self.max_depth:
            current = current.best_child(self.c)
            depth += 1
        
        # Expansion: add new child if not at max depth
        if depth < self.max_depth and not current.is_terminal:
            new_step = self.llm.generate_reasoning_step(problem, current.state)
            
            if new_step:
                new_state = current.state + ("\n" if current.state else "") + new_step
                child = ReasoningNode(state=new_state, parent=current, action=new_step)
                current.children.append(child)
                return child
        
        return current
    
    def _simulate(self, node: ReasoningNode, problem: str, correct_answer: str) -> float:
        """Simulate from node to get reward."""
        # Complete reasoning from this point
        reasoning = node.state
        
        # If we have some reasoning, try to get final answer
        if reasoning:
            final_answer = self.llm.complete_solution(problem, reasoning)
            evaluation = evaluate_solution(correct_answer, reasoning, final_answer)
            return evaluation['total_reward']
        else:
            # No reasoning yet, small penalty
            return -0.3
    
    def _backpropagate(self, node: ReasoningNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def _get_best_path(self, root: ReasoningNode) -> List[ReasoningNode]:
        """Extract best path from root."""
        path = [root]
        current = root
        
        while current.children:
            # Follow most-visited child
            current = max(current.children, key=lambda c: c.visit_count)
            path.append(current)
        
        return path


def baseline_llm_solve(llm: LLMInterface, problem: str, correct_answer: str) -> Dict:
    """
    Baseline: Direct LLM solution without MCTS.
    
    Args:
        llm: LLM interface
        problem: Problem to solve
        correct_answer: Correct answer for evaluation
    
    Returns:
        Results dictionary
    """
    prompt = f"""Solve this problem step by step:

{problem}

Provide your reasoning, then give the final answer."""
    
    response = llm.generate(prompt, temperature=0.7, max_tokens=300)
    
    # Try to extract final answer
    final_answer = llm.generate(f"Based on this solution, what is the final numerical answer? Just give the number.\n\n{response}", 
                                temperature=0.3, max_tokens=50)
    
    evaluation = evaluate_solution(correct_answer, response, final_answer)
    
    return {
        'reasoning_chain': response,
        'final_answer': final_answer,
        'evaluation': evaluation,
        'method': 'baseline'
    }


if __name__ == "__main__":
    print("Testing LLM-MCTS...")
    
    from tasks import get_math_problems
    
    # Initialize
    llm = LLMInterface(provider="anthropic")
    mcts = LLM_MCTS(llm, max_depth=3, num_simulations=5)
    
    # Test on one problem
    problems = get_math_problems()
    problem = problems[0]
    
    print(f"\nProblem: {problem.question}")
    print(f"Correct answer: {problem.answer}")
    
    # Run MCTS
    print("\nRunning LLM-MCTS...")
    result = mcts.search(problem.question, problem.answer)
    
    print(f"\nReasoning:\n{result['reasoning_chain']}")
    print(f"\nFinal answer: {result['final_answer']}")
    print(f"\nCorrect: {result['evaluation']['correct']}")