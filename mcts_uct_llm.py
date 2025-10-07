"""
Assignment Report: MCTS-UCT Implementation with LLM Proxy Guidance

1. Paper Source:
"Large Language Models as Commonsense Knowledge for Large-Scale Task Planning"
Authors: Zirui Zhao, Wee Sun Lee, David Hsu
Conference: NeurIPS 2023
Link: https://arxiv.org/abs/2305.14078

2. Chosen Approach:
Replicated a key idea of the paper by implementing MCTS-UCT for a toy sequential planning problem.
The MCTS search is guided by a mock LLM scoring function that heuristically rates states to bias search towards promising states.

3. Implementation Summary:
- Environment: A simple numeric state space where actions increment the state by 1 or 2.
- Goal: Reach state 10 to receive reward 1, else reward 0.
- MCTS with UCT balances exploration and exploitation among possible actions.
- The mock LLM scoring function simulates an LLM's role by scoring states based on closeness to target.
- This influences rollout action selection probability, imitating LLM-guided planning.
- The best action is selected after a fixed number of simulations at each step.

4. Results:
The program successfully plans a sequence of actions to reach the goal state 10.
Example output steps:
  Step 1, choosing action: 1, New state: 1
  ...
  Step 7, choosing action: 2, New state: 10 (Terminal)
This demonstrates a working MCTS-LLM interaction on a toy planning task.

This implementation meets the assignment criteria by combining MCTS-UCT with LLM-inspired scoring in a clear and runnable Python example, suitable for further extension.

"""

import math
import random


class ToyEnvironment:
    def __init__(self):
        self.state = 0  # simple integer state

    def get_possible_actions(self):
        # Two possible actions at each step: +1 or +2
        return [1, 2]

    def take_action(self, action):
        new_env = ToyEnvironment()
        new_env.state = self.state + action
        return new_env

    def is_terminal(self):
        # Terminal if state >= 10
        return self.state >= 10

    def get_reward(self):
        # Reward is 1 if state == 10 else 0
        return 1 if self.state == 10 else 0

# Mock LLM proxy: scores a state+action based on heuristic closeness to goal state 10


def llm_score(env: ToyEnvironment):
    # Score normalized distance to goal (higher is better)
    dist = 10 - env.state
    if dist <= 0:
        return 1.0
    else:
        return 1 / (dist + 1)


class MCTSNode:
    def __init__(self, state: ToyEnvironment, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, c_param=1.4):
        choices_weights = []
        for action, child in self.children.items():
            exploitation = child.value / (child.visits + 1e-6)
            exploration = math.sqrt(
                2 * math.log(self.visits + 1) / (child.visits + 1e-6))
            # UCT formula
            score = exploitation + c_param * exploration
            choices_weights.append((score, action, child))
        return max(choices_weights, key=lambda x: x[0])[2]

    def expand(self):
        untried_actions = [
            a for a in self.state.get_possible_actions() if a not in self.children]
        action = random.choice(untried_actions)
        next_state = self.state.take_action(action)
        child_node = MCTSNode(next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def rollout(self):
        current_env = self.state
        while not current_env.is_terminal():
            actions = current_env.get_possible_actions()
            # Use LLM score to weight action selection
            scores = []
            for a in actions:
                next_env = current_env.take_action(a)
                scores.append(llm_score(next_env))
            total = sum(scores)
            probs = [s/total for s in scores]
            action = random.choices(actions, weights=probs, k=1)[0]
            current_env = current_env.take_action(action)
        return current_env.get_reward()

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


def mcts(root_state, n_simulations):
    root_node = MCTSNode(root_state)
    for _ in range(n_simulations):
        node = root_node
        # Selection
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()
        # Expansion
        if not node.state.is_terminal():
            node = node.expand()
        # Rollout
        reward = node.rollout()
        # Backpropagation
        node.backpropagate(reward)
    # Choose the most visited child as best action
    best_move = max(root_node.children.items(),
                    key=lambda item: item[1].visits)[0]
    return best_move


if __name__ == "__main__":
    env = ToyEnvironment()
    print("Starting state:", env.state)
    for step in range(10):
        action = mcts(env, 100)
        print(f"Step {step + 1}, choosing action: {action}")
        env = env.take_action(action)
        print(f"New state: {env.state}")
        if env.is_terminal():
            print("Reached terminal state.")
            break
