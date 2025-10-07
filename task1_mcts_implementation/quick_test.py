"""Quick test to verify MCTS works"""

from mcts_core import MCTS
from environments import TicTacToe

print("🎮 Testing MCTS with TicTacToe...\n")

# Create environment
env = TicTacToe()
state = env.get_initial_state()

print("Initial board:")
env.render(state)

# Run MCTS
mcts = MCTS()
print("Running MCTS with 100 simulations...")
action, root = mcts.search(state, env, num_simulations=100)

print(f"\n✅ MCTS recommends action: {action}")
print(f"✅ Root visited {root.visit_count} times")
print(f"✅ Found {len(root.children)} child nodes")

print("\n🎉 MCTS implementation works!")