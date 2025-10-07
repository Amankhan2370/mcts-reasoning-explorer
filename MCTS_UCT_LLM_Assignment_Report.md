# MCTS-UCT with LLM Proxy Assignment Report

## Paper Source
- Title: Large Language Models as Commonsense Knowledge for Large-Scale Task Planning  
- Authors: Zirui Zhao, Wee Sun Lee, David Hsu  
- Conference: NeurIPS 2023  
- Link: https://arxiv.org/abs/2305.14078

## Approach
Replicated a key idea from the paper by implementing the MCTS-UCT algorithm combined with a mock LLM scoring function on a toy sequential decision-making task. The LLM proxy scores states heuristically to guide the MCTS rollout phase.

## Implementation Highlights
- **Environment:** A simple numeric state that increments by 1 or 2 actions with a goal to reach state 10.
- **MCTS Algorithm:** Utilizes Upper Confidence Bound (UCT) formula to balance exploration and exploitation.
- **LLM Proxy:** Assigns scores to simulated states based on closeness to the goal to bias rollout action selection probabilistically.
- **Selection of Best Action:** After fixed simulations per decision step, the most visited child node's action is selected.

## Results
The MCTS-UCT code successfully finds an effective path to the goal state through action sequences, demonstrating the benefit of LLM-guided planning. Example output shows states progressing to terminal state 10.

## Limitations and Future Work
- Current LLM proxy is a simplistic heuristic simulator; real LLM integration would enhance applicability.
- The toy environment is basic; more complex planning tasks would better validate the approach.
- Hallucination detection and knowledge-grounding aspects from the original paper are not implemented here.

## References
- NeurIPS 2023 paper: https://arxiv.org/abs/2305.14078
- UW Lecture Notes on MCTS
- Monte Carlo Tree Search algorithm explanations

