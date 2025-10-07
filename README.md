# MCTS-UCT with LLM: Assignment Implementation

## Overview
This repository contains a simple implementation of Monte Carlo Tree Search with Upper Confidence Bound applied to Trees (MCTS-UCT), guided by a mock Large Language Model (LLM) proxy scoring function. The implementation replicates a key idea from the NeurIPS 2023 paper *"Large Language Models as Commonsense Knowledge for Large-Scale Task Planning"*.

## Paper Reference
- Title: Large Language Models as Commonsense Knowledge for Large-Scale Task Planning  
- Authors: Zirui Zhao, Wee Sun Lee, David Hsu  
- Conference: NeurIPS 2023  
- Link: [https://arxiv.org/abs/2305.14078](https://arxiv.org/abs/2305.14078)

## What This Project Does
- Implements MCTS-UCT for a toy sequential decision task where the state increments by +1 or +2 actions.
- Uses a heuristic LLM proxy function to score states during rollouts, biasing MCTS towards promising paths.
- Demonstrates the interaction of classical search algorithms with LLM guidance in task planning.

## Files
- `mcts_uct_llm.py`: Python code for the MCTS-UCT with LLM proxy implementation.
- `MCTS_UCT_LLM_Assignment_Report.md`: Detailed assignment report explaining the approach, implementation, and results.
- `MCTS_UCT_LLM_Summary.md`: A concise summary of the project and key concepts.

## How to Run
1. Ensure you have Python 3 installed.
2. Run the main script:

3. Observe the sequence of chosen actions towards the goal state printed to the terminal.

## Notes
- The LLM proxy is a simplified heuristic and not a real language model.
- The toy environment is minimal to keep the implementation straightforward and understandable.
- This work serves as a foundation for integrating real LLMs into MCTS for complex task planning.

## License
This project is for educational and assignment purposes only.

