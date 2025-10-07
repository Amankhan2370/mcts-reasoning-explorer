# Summary of MCTS-UCT with LLM Assignment

## Overview
This assignment implements a simplified MCTS-UCT algorithm inspired by recent research combining MCTS with Large Language Models (LLMs) for task planning. The approach uses a heuristic LLM proxy to guide search decisions.

## Key Concepts
- **MCTS (Monte Carlo Tree Search):** A search algorithm balancing exploration and exploitation using simulations.
- **UCT (Upper Confidence Bound for Trees):** A formula guiding node selection during search.
- **LLM Proxy:** A heuristic function simulating the guidance of an LLM by scoring how close a state is to the goal.

## Implementation Details
- The environment consists of numeric states incremented by +1 or +2 actions.
- The goal is to reach state 10 for a reward of 1.
- MCTS performs simulations by selecting, expanding, rolling out, and backpropagating rewards in the tree.
- The LLM proxy biases rollout action choices towards promising states.

## Outcome
The code efficiently finds a path reaching the goal state, demonstrating MCTS's capacity to plan in guided environments, aligning with the research's emphasis on LLM-assisted planning.

## Significance
This assignment captures the essence of integrating LLM guidance with classical search algorithms, providing a foundation for future work using real LLM APIs and complex tasks.

