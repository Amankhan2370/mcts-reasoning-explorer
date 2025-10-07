"""
Core MCTS-UCT Implementation
Based on Sutton & Barto Ch. 8 and UCT algorithm
Author: [Your Name]
Date: October 2025
"""

import math
import random
from typing import Any, List, Dict, Optional, Tuple
from collections import defaultdict


class MCTSNode:
    """
    Represents a node in the MCTS tree.
    
    Attributes:
        state: The game state this node represents
        parent: Parent node (None for root)
        action: Action taken from parent to reach this node
        children: Dictionary mapping actions to child nodes
        visit_count: Number of times this node has been visited (N(s))
        total_reward: Sum of all rewards received through this node
        untried_actions: List of actions not yet tried from this state
    """
    
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, 
                 action: Any = None, untried_actions: List[Any] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Any, 'MCTSNode'] = {}
        self.visit_count = 0  # N(s)
        self.total_reward = 0.0
        self.untried_actions = untried_actions if untried_actions else []
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions from this node have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return len(self.children) == 0 and len(self.untried_actions) == 0
    
    def best_child(self, c: float = math.sqrt(2)) -> 'MCTSNode':
        """
        Select best child using UCT formula.
        
        UCT(s,a) = Q̄(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        
        Args:
            c: Exploration constant (default √2)
        
        Returns:
            Child node with highest UCT value
        """
        choices_weights = []
        
        for child in self.children.values():
            if child.visit_count == 0:
                uct_value = float('inf')
            else:
                # Exploitation term
                exploitation = child.total_reward / child.visit_count
                
                # Exploration term
                exploration = c * math.sqrt(
                    math.log(self.visit_count) / child.visit_count
                )
                
                uct_value = exploitation + exploration
            
            choices_weights.append((child, uct_value))
        
        return max(choices_weights, key=lambda x: x[1])[0]
    
    def expand(self, action: Any, next_state: Any, 
               legal_actions: List[Any]) -> 'MCTSNode':
        """
        Expand tree by creating a new child node.
        
        Args:
            action: Action to take
            next_state: Resulting state
            legal_actions: Legal actions from the new state
        
        Returns:
            The newly created child node
        """
        child = MCTSNode(
            state=next_state,
            parent=self,
            action=action,
            untried_actions=legal_actions.copy()
        )
        self.untried_actions.remove(action)
        self.children[action] = child
        return child
    
    def update(self, reward: float):
        """
        Update node statistics (backpropagation step).
        
        Args:
            reward: Reward to backpropagate
        """
        self.visit_count += 1
        self.total_reward += reward
    
    def get_average_reward(self) -> float:
        """Get average reward Q̄(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count


class MCTS:
    """
    Monte Carlo Tree Search with UCT.
    
    This is a general implementation that works with any environment
    that provides the required interface.
    """
    
    def __init__(self, 
                 exploration_constant: float = math.sqrt(2),
                 max_rollout_depth: int = 100,
                 discount_factor: float = 1.0):
        """
        Initialize MCTS.
        
        Args:
            exploration_constant: UCT exploration parameter (c)
            max_rollout_depth: Maximum depth for rollout simulation
            discount_factor: Discount factor γ for future rewards
        """
        self.c = exploration_constant
        self.max_rollout_depth = max_rollout_depth
        self.gamma = discount_factor
    
    def search(self, 
               root_state: Any,
               environment,
               num_simulations: int = 1000,
               verbose: bool = False) -> Tuple[Any, MCTSNode]:
        """
        Run MCTS search from root state.
        
        Args:
            root_state: Initial state to search from
            environment: Environment object with required methods
            num_simulations: Number of MCTS iterations
            verbose: Print progress
        
        Returns:
            Tuple of (best_action, root_node)
        """
        # Create root node
        root = MCTSNode(
            state=root_state,
            untried_actions=environment.get_legal_actions(root_state)
        )
        
        # Run simulations
        for i in range(num_simulations):
            if verbose and (i + 1) % 100 == 0:
                print(f"Simulation {i + 1}/{num_simulations}")
            
            # MCTS four phases
            node = self._select(root, environment)
            reward = self._simulate(node.state, environment)
            self._backpropagate(node, reward)
        
        # Return best action
        best_action = self._best_action(root)
        return best_action, root
    
    def _select(self, node: MCTSNode, environment) -> MCTSNode:
        """
        Phase 1: SELECTION
        Navigate down the tree using UCT until reaching a node that's not fully expanded.
        Then expand it.
        """
        while not environment.is_terminal(node.state):
            if not node.is_fully_expanded():
                # EXPANSION: Add a new child
                return self._expand(node, environment)
            else:
                # Keep selecting using UCT
                node = node.best_child(self.c)
        
        return node
    
    def _expand(self, node: MCTSNode, environment) -> MCTSNode:
        """
        Phase 2: EXPANSION
        Add one new child to the tree.
        """
        # Pick a random untried action
        action = random.choice(node.untried_actions)
        
        # Apply action to get next state
        next_state = environment.get_next_state(node.state, action)
        
        # Get legal actions from new state
        legal_actions = environment.get_legal_actions(next_state)
        
        # Create and return new child node
        child = node.expand(action, next_state, legal_actions)
        return child
    
    def _simulate(self, state: Any, environment) -> float:
        """
        Phase 3: SIMULATION (Rollout)
        Play out randomly from the given state until terminal or depth limit.
        
        Returns:
            Cumulative discounted reward
        """
        current_state = state
        total_reward = 0.0
        discount = 1.0
        depth = 0
        
        # Rollout with random policy
        while not environment.is_terminal(current_state) and depth < self.max_rollout_depth:
            legal_actions = environment.get_legal_actions(current_state)
            if not legal_actions:
                break
            
            action = random.choice(legal_actions)
            next_state = environment.get_next_state(current_state, action)
            reward = environment.get_reward(current_state, action, next_state)
            
            # Accumulate discounted reward
            total_reward += discount * reward
            discount *= self.gamma
            
            current_state = next_state
            depth += 1
        
        # Add terminal reward if reached terminal state
        if environment.is_terminal(current_state):
            terminal_reward = environment.get_terminal_reward(current_state)
            total_reward += discount * terminal_reward
        
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Phase 4: BACKPROPAGATION
        Update statistics for all nodes on the path from node to root.
        """
        current = node
        
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def _best_action(self, root: MCTSNode) -> Any:
        """
        Select the best action from root after all simulations.
        
        Strategy: Choose action with most visits (most robust).
        """
        if not root.children:
            return None
        
        # Return action with highest visit count
        best_child = max(root.children.values(), 
                        key=lambda child: child.visit_count)
        return best_child.action


def mcts_decision(state, environment, num_simulations=1000, **kwargs):
    """
    Convenient function to get MCTS decision for a single state.
    
    Args:
        state: Current state
        environment: Environment object
        num_simulations: Number of MCTS iterations
        **kwargs: Additional arguments for MCTS
    
    Returns:
        Best action to take
    """
    mcts = MCTS(**kwargs)
    action, _ = mcts.search(state, environment, num_simulations)
    return action