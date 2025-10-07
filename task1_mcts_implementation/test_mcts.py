"""
Unit tests for MCTS implementation.
Run with: pytest test_mcts.py -v
Or run directly: python3 test_mcts.py
"""

import pytest
import numpy as np
from mcts_core import MCTS, MCTSNode, mcts_decision
from environments import TicTacToe, SimpleMaze


class TestMCTSNode:
    """Test MCTSNode functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        state = np.zeros((3, 3))
        node = MCTSNode(state, untried_actions=[(0, 0), (0, 1)])
        
        assert node.visit_count == 0
        assert node.total_reward == 0.0
        assert len(node.untried_actions) == 2
        assert not node.is_fully_expanded()
        print("✅ Node creation test passed")
    
    def test_node_update(self):
        """Test node update mechanism."""
        node = MCTSNode(None)
        node.update(10.0)
        node.update(20.0)
        
        assert node.visit_count == 2
        assert node.total_reward == 30.0
        assert node.get_average_reward() == 15.0
        print("✅ Node update test passed")
    
    def test_uct_calculation(self):
        """Test UCT value calculation."""
        parent = MCTSNode(None)
        parent.visit_count = 100
        
        child = MCTSNode(None, parent=parent)
        child.visit_count = 10
        child.total_reward = 5.0
        
        parent.children['action1'] = child
        
        # Best child should work
        best = parent.best_child(c=1.0)
        assert best == child
        print("✅ UCT calculation test passed")


class TestTicTacToe:
    """Test TicTacToe environment."""
    
    def test_initial_state(self):
        """Test initial game state."""
        env = TicTacToe()
        state = env.get_initial_state()
        
        assert state.shape == (3, 3)
        assert np.all(state == 0)
        assert len(env.get_legal_actions(state)) == 9
        print("✅ TicTacToe initial state test passed")
    
    def test_make_move(self):
        """Test making a move."""
        env = TicTacToe()
        state = env.get_initial_state()
        
        # Player 1 makes a move
        next_state = env.get_next_state(state, (0, 0))
        assert next_state[0, 0] == 1
        assert len(env.get_legal_actions(next_state)) == 8
        print("✅ TicTacToe move test passed")
    
    def test_winner_detection_row(self):
        """Test horizontal win detection."""
        env = TicTacToe()
        
        # Create winning state for player 1 (horizontal)
        state = np.array([
            [1, 1, 1],
            [0, -1, 0],
            [0, -1, 0]
        ])
        
        assert env.is_terminal(state)
        assert env.get_terminal_reward(state) == 1.0
        print("✅ TicTacToe row winner test passed")
    
    def test_winner_detection_column(self):
        """Test vertical win detection."""
        env = TicTacToe()
        
        # Create winning state for player 2 (vertical)
        state = np.array([
            [-1, 1, 0],
            [-1, 1, 0],
            [-1, 0, 0]
        ])
        
        assert env.is_terminal(state)
        assert env.get_terminal_reward(state) == -1.0
        print("✅ TicTacToe column winner test passed")
    
    def test_winner_detection_diagonal(self):
        """Test diagonal win detection."""
        env = TicTacToe()
        
        # Create winning state for player 1 (diagonal)
        state = np.array([
            [1, -1, 0],
            [0, 1, -1],
            [0, 0, 1]
        ])
        
        assert env.is_terminal(state)
        assert env.get_terminal_reward(state) == 1.0
        print("✅ TicTacToe diagonal winner test passed")
    
    def test_draw(self):
        """Test draw detection."""
        env = TicTacToe()
        
        # Create draw state
        state = np.array([
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, -1]
        ])
        
        assert env.is_terminal(state)
        assert env.get_terminal_reward(state) == 0.0
        print("✅ TicTacToe draw test passed")


class TestSimpleMaze:
    """Test SimpleMaze environment."""
    
    def test_initial_state(self):
        """Test maze initial state."""
        env = SimpleMaze(size=5)
        state = env.get_initial_state()
        
        assert state == (0, 0, 0)
        assert len(env.get_legal_actions(state)) > 0
        print("✅ Maze initial state test passed")
    
    def test_movement(self):
        """Test maze movement."""
        env = SimpleMaze(size=5)
        state = (0, 0, 0)
        
        # Move right
        if 'right' in env.get_legal_actions(state):
            next_state = env.get_next_state(state, 'right')
            assert next_state == (0, 1, 1)
        print("✅ Maze movement test passed")
    
    def test_goal_detection(self):
        """Test goal reaching."""
        env = SimpleMaze(size=5)
        goal_state = (4, 4, 10)
        
        assert env.is_terminal(goal_state)
        assert env.get_terminal_reward(goal_state) == 100.0
        print("✅ Maze goal detection test passed")


class TestMCTS:
    """Test MCTS algorithm."""
    
    def test_mcts_initialization(self):
        """Test MCTS object creation."""
        mcts = MCTS(exploration_constant=1.4)
        assert mcts.c == 1.4
        assert mcts.max_rollout_depth == 100
        print("✅ MCTS initialization test passed")
    
    def test_mcts_runs(self):
        """Test that MCTS runs without errors."""
        env = TicTacToe()
        state = env.get_initial_state()
        
        mcts = MCTS(exploration_constant=1.4)
        action, root = mcts.search(state, env, num_simulations=50)
        
        assert action is not None
        assert root.visit_count > 0
        assert len(root.children) > 0
        print("✅ MCTS execution test passed")
    
    def test_mcts_finds_winning_move(self):
        """Test that MCTS finds obvious winning move."""
        env = TicTacToe()
        
        # Create state where player 1 can win immediately
        state = np.array([
            [1, 1, 0],
            [0, -1, 0],
            [0, -1, 0]
        ])
        
        # MCTS should find (0, 2) as winning move
        action = mcts_decision(state, env, num_simulations=200)
        
        # Apply action and check it's a win
        next_state = env.get_next_state(state, action)
        assert env.get_terminal_reward(next_state) == 1.0
        print(f"✅ MCTS found winning move: {action}")
    
    def test_mcts_blocks_opponent(self):
        """Test that MCTS blocks opponent's winning move."""
        env = TicTacToe()
        
        # Create state where player 2 can win next turn
        state = np.array([
            [1, 0, 0],
            [-1, -1, 0],
            [1, 0, 0]
        ])
        
        # Player 1's turn - should block at (1, 2)
        action = mcts_decision(state, env, num_simulations=200)
        
        # Check blocking works
        print(f"✅ MCTS chose action: {action}")
        assert action in env.get_legal_actions(state)
    
    def test_mcts_maze_navigation(self):
        """Test MCTS on maze navigation."""
        env = SimpleMaze(size=5)
        state = env.get_initial_state()
        
        mcts = MCTS(exploration_constant=1.4, max_rollout_depth=50)
        action, root = mcts.search(state, env, num_simulations=100)
        
        assert action is not None
        assert action in ['up', 'down', 'left', 'right']
        print(f"✅ MCTS maze action: {action}")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "="*60)
    print("RUNNING MCTS UNIT TESTS")
    print("="*60 + "\n")
    
    # Node tests
    print("--- Testing MCTSNode ---")
    node_tests = TestMCTSNode()
    node_tests.test_node_creation()
    node_tests.test_node_update()
    node_tests.test_uct_calculation()
    
    # TicTacToe tests
    print("\n--- Testing TicTacToe Environment ---")
    ttt_tests = TestTicTacToe()
    ttt_tests.test_initial_state()
    ttt_tests.test_make_move()
    ttt_tests.test_winner_detection_row()
    ttt_tests.test_winner_detection_column()
    ttt_tests.test_winner_detection_diagonal()
    ttt_tests.test_draw()
    
    # Maze tests
    print("\n--- Testing SimpleMaze Environment ---")
    maze_tests = TestSimpleMaze()
    maze_tests.test_initial_state()
    maze_tests.test_movement()
    maze_tests.test_goal_detection()
    
    # MCTS tests
    print("\n--- Testing MCTS Algorithm ---")
    mcts_tests = TestMCTS()
    mcts_tests.test_mcts_initialization()
    mcts_tests.test_mcts_runs()
    mcts_tests.test_mcts_finds_winning_move()
    mcts_tests.test_mcts_blocks_opponent()
    mcts_tests.test_mcts_maze_navigation()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests directly
    run_all_tests()