MCTS Reasoning Explorer
A comprehensive implementation of Monte Carlo Tree Search (MCTS) with Upper Confidence bounds applied to Trees (UCT), featuring applications in game playing, navigation, and integration with Large Language Models for advanced reasoning tasks.

📋 Table of Contents

Overview
Features
Project Structure
Installation
Quick Start
Usage Guide
Experimental Results
API Configuration
Documentation
Testing
Technical Details
References
License


🎯 Overview
This project implements Monte Carlo Tree Search (MCTS), a powerful decision-making algorithm that combines tree search with Monte Carlo simulation. The implementation includes:

Pure MCTS-UCT: Classic implementation for game playing and navigation
LLM-MCTS Integration: Novel application of MCTS to guide Large Language Model reasoning

MCTS has revolutionized AI game-playing (notably in AlphaGo) and this project demonstrates its versatility across multiple domains, from TicTacToe to mathematical reasoning with LLMs.

✨ Features
Core MCTS Implementation

UCT Algorithm: Upper Confidence bounds applied to Trees with configurable exploration constant
Four-Phase Process: Selection, Expansion, Simulation, Backpropagation
Domain-Independent: Works with any environment implementing required interface
Anytime Algorithm: Performance improves gracefully with computational budget

Environments

TicTacToe: 3×3 grid game for testing strategic play
Maze Navigation: 5×5 grid with obstacles for spatial reasoning
Extensible: Easy to add new environments

LLM Integration

Multi-Provider Support: OpenAI, Anthropic, HuggingFace, OpenRouter
Reasoning Search: MCTS explores different reasoning paths
Math Problem Solving: Demonstrates improved accuracy through search
Secure: Environment variable-based API key management

Testing & Analysis

18+ Unit Tests: Comprehensive test coverage
Performance Experiments: Systematic evaluation across configurations
Visualization: Matplotlib-based plots of experimental results
Detailed Reports: In-depth analysis and documentation


📁## 📁 Project Structure

### Root Files
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT License
- `.gitignore` - Git ignore rules

### Task 1: Core MCTS Implementation (`task1_mcts_implementation/`)
- `mcts_core.py` - MCTS algorithm (MCTSNode, MCTS classes)
- `environments.py` - Game environments (TicTacToe, SimpleMaze)
- `test_mcts.py` - Unit tests (18 tests, all passing)
- `experiments.py` - Performance evaluation experiments
- `visualize.py` - Tree and result visualization
- `quick_test.py` - Quick sanity check
- `results/` - Experimental outputs (PNG plots, JSON data)

### Task 2: LLM-MCTS Integration (`task2_llm_mcts/`)
- `llm_interface.py` - Multi-provider LLM API wrapper
- `llm_mcts.py` - MCTS adapted for text reasoning
- `tasks.py` - Math problem definitions
- `evaluators.py` - Answer correctness evaluation
- `run_experiments.py` - Main experiment runner
- `results/` - LLM-MCTS outputs (comparison plots, JSON data)

### Documentation (`docs/`)
- `task1_report.md` - Complete MCTS implementation report (20+ pages)
- `task2_report.md` - LLM-MCTS integration report

### Notebooks (`notebooks/` - Optional)
- `task1_demo.ipynb` - Interactive MCTS demonstration
- `task2_analysis.ipynb` - LLM-MCTS result analysis

🚀 Installation
Prerequisites

Python 3.8 or higher
pip package manager
(Optional) Virtual environment

Basic Setup
bash# Clone the repository
git clone https://github.com/Amankhan2370/mcts-reasoning-explorer.git
cd mcts-reasoning-explorer

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Verify Installation
bash# Run quick test
python3 task1_mcts_implementation/quick_test.py

# Expected output:
# ✅ MCTS recommends action: (1, 1)
# ✅ Root visited 100 times
# 🎉 MCTS implementation works!

🎮 Quick Start
Task 1: Run Pure MCTS
bash# Run comprehensive unit tests
python3 task1_mcts_implementation/test_mcts.py

# Output: 18/18 tests passing ✓

# Run performance experiments (takes 5-10 minutes)
python3 task1_mcts_implementation/experiments.py

# Results saved to: task1_mcts_implementation/results/
Task 2: Run LLM-MCTS (Requires API Key)
bash# Step 1: Set API key (choose your provider)
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
# OR
export HF_API_KEY="your-key-here"

# Step 2: Run LLM-MCTS experiments (takes 20-30 minutes)
python3 task2_llm_mcts/run_experiments.py

# Results saved to: task2_llm_mcts/results/

📖 Usage Guide
Using MCTS for Games
pythonfrom task1_mcts_implementation.mcts_core import MCTS
from task1_mcts_implementation.environments import TicTacToe

# Initialize environment and MCTS
env = TicTacToe()
mcts = MCTS(exploration_constant=1.414)

# Get initial state
state = env.get_initial_state()

# Get best action using MCTS
action, root = mcts.search(state, env, num_simulations=100)
print(f"MCTS recommends: {action}")

# Analyze the search tree
print(f"Explored {root.visit_count} nodes")
print(f"Found {len(root.children)} possible moves")
Using LLM-MCTS for Reasoning
pythonfrom task2_llm_mcts.llm_interface import LLMInterface
from task2_llm_mcts.llm_mcts import LLM_MCTS

# Initialize LLM (requires API key in environment)
llm = LLMInterface(provider="openai")

# Initialize LLM-MCTS
mcts = LLM_MCTS(llm, max_depth=4, num_simulations=8)

# Solve a problem
problem = "Sarah has 8 cookies. She eats 3. How many are left?"
result = mcts.search(problem, correct_answer="5")

# View results
print(f"Reasoning:\n{result['reasoning_chain']}")
print(f"Final answer: {result['final_answer']}")
print(f"Correct: {result['evaluation']['correct']}")
Creating Custom Environments
pythonclass CustomEnvironment:
    """Your custom environment must implement these methods."""
    
    def get_legal_actions(self, state):
        """Return list of valid actions from state."""
        pass
    
    def get_next_state(self, state, action):
        """Return resulting state after taking action."""
        pass
    
    def is_terminal(self, state):
        """Return True if state is terminal."""
        pass
    
    def get_reward(self, state, action, next_state):
        """Return reward for transition."""
        pass
    
    def get_terminal_reward(self, state):
        """Return final reward at terminal state."""
        pass

📊 Experimental Results
Task 1: Pure MCTS Performance
Experiment 1: Simulation Budget Impact

Tested: 10, 50, 100, 200, 500 simulations
Result: 100% win rate across all budgets vs 50-sim opponent
Conclusion: Even minimal computation (10 sims) achieves strong performance

Experiment 2: Exploration Constant Tuning

Tested: c ∈ {0.5, 1.0, √2, 2.0, 3.0}
Result: Standard √2 performs optimally
Conclusion: Theoretical recommendations validated empirically

Experiment 3: Maze Navigation

Success rate: 67-100% depending on simulation budget
Path efficiency: 15-18 steps (optimal: ~8 steps)
Conclusion: MCTS finds solutions but random rollouts limit optimality

Task 2: LLM-MCTS Results
Setup: 10 math word problems, baseline LLM vs LLM-MCTS
Key Findings:

Accuracy improvement through systematic search
Cost analysis: ~10x more API calls, improved correctness
Best use case: Multi-step reasoning problems

Detailed results: See docs/task2_report.md

🔑 API Configuration
Supported Providers
ProviderGet API KeyCostBest ForOpenAIplatform.openai.com~$0.10/10 problemsGeneral purposeAnthropicconsole.anthropic.com~$0.10/10 problemsReasoning tasksHuggingFacehuggingface.co/settings/tokensFree tier availableExperimentationOpenRouteropenrouter.ai/keysVaries by modelMultiple models
Setting Up API Keys
Environment Variables (Recommended)
bash# Temporary (current session only)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_API_KEY="hf_..."
export OPENROUTER_API_KEY="sk-or-..."

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
⚠️ Security Warning: Never commit API keys to version control. Always use environment variables.

📚 Documentation
Complete Reports

Task 1 Report: docs/task1_report.md

Algorithm design and implementation
Experimental methodology
Results and analysis
20+ pages of detailed documentation


Task 2 Report: docs/task2_report.md

LLM-MCTS integration approach
Comparison with baseline methods
Cost-benefit analysis
Future directions



Key Algorithms
UCT Formula:
UCT(s,a) = Q̄(s,a) + c√(ln N(s) / N(s,a))
MCTS Four Phases:

Selection: Navigate tree using UCT
Expansion: Add new child node
Simulation: Random rollout to terminal
Backpropagation: Update statistics upward


🧪 Testing
Run All Tests
bash# Task 1 unit tests
python3 task1_mcts_implementation/test_mcts.py

# Or with pytest (more detailed)
pytest task1_mcts_implementation/test_mcts.py -v

# Expected: 18/18 tests passing ✓
Test Coverage

MCTSNode: Creation, updates, UCT calculation
TicTacToe: State transitions, win detection, draw detection
SimpleMaze: Navigation, obstacle avoidance, goal detection
MCTS Algorithm: Search execution, action selection, strategic play


🔧 Technical Details
Dependencies
Core Requirements:
numpy>=1.24.0          # Numerical operations
matplotlib>=3.7.0      # Visualization
seaborn>=0.12.0       # Statistical plotting
tqdm>=4.65.0          # Progress bars
Optional (for LLM integration):
openai>=1.0.0         # OpenAI API
anthropic>=0.18.0     # Anthropic API
huggingface-hub>=0.20.0  # HuggingFace API
Performance Characteristics
Task 1 MCTS:

100 simulations: ~1 second (TicTacToe)
500 simulations: ~3-4 seconds (TicTacToe)
Memory: O(simulations × depth)

Task 2 LLM-MCTS:

8 simulations: ~20-30 seconds per problem
API latency: ~1-2 seconds per call
Cost: ~$0.01 per problem (varies by provider)

Code Quality

Lines of Code: ~1,800 total
Code Style: PEP 8 compliant
Documentation: Comprehensive docstrings
Type Hints: Used throughout for clarity


🎓 References
Academic Papers

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
Kocsis, L., & Szepesvári, C. (2006). Bandit based monte-carlo planning. ECML.
Browne, C., et al. (2012). A survey of monte carlo tree search methods. IEEE TCIAIG.
Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. Nature.
Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS.


📄 License
MIT License - Copyright (c) 2025 Aman Khan
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

👤 Author
Aman Khan

Student ID: 002050777
Institution: Northeastern University
Course: Self-Learning AI


🙏 Acknowledgments

Implementation based on Sutton & Barto's textbook
UCT algorithm from Kocsis & Szepesvári (2006)
LLM integration inspired by Tree of Thoughts research
AlphaGo methodology from DeepMind


⭐ If you find this project useful, please star the repository!
