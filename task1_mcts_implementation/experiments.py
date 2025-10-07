"""
Run experiments and generate results for Task 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from mcts_core import MCTS
from environments import TicTacToe, SimpleMaze

sns.set_style("whitegrid")


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs('task1_mcts_implementation/results', exist_ok=True)


def play_tictactoe_game(player1_sims, player2_sims, env):
    """
    Play one game of TicTacToe between two MCTS agents.
    
    Returns:
        winner: 1, -1, or 0 (draw)
        num_moves: Number of moves in the game
    """
    state = env.get_initial_state()
    move_count = 0
    
    mcts1 = MCTS(exploration_constant=np.sqrt(2))
    mcts2 = MCTS(exploration_constant=np.sqrt(2))
    
    while not env.is_terminal(state):
        # Alternate between players
        if move_count % 2 == 0:
            action, _ = mcts1.search(state, env, num_simulations=player1_sims, verbose=False)
        else:
            action, _ = mcts2.search(state, env, num_simulations=player2_sims, verbose=False)
        
        state = env.get_next_state(state, action)
        move_count += 1
    
    winner = env.get_terminal_reward(state)
    return winner, move_count


def experiment_1_simulation_budget():
    """
    Experiment 1: Test MCTS performance vs simulation budget
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Performance vs Simulation Budget")
    print("="*60)
    
    ensure_results_dir()
    
    env = TicTacToe()
    simulation_counts = [10, 50, 100, 200, 500]
    num_games = 30
    opponent_sims = 50
    
    results = {
        'simulations': [],
        'win_rate': [],
        'draw_rate': [],
        'loss_rate': [],
        'avg_moves': []
    }
    
    for num_sims in simulation_counts:
        print(f"\nTesting with {num_sims} simulations (vs opponent with {opponent_sims})...")
        
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        
        for game in tqdm(range(num_games), desc=f"Sims={num_sims}"):
            winner, moves = play_tictactoe_game(num_sims, opponent_sims, env)
            total_moves += moves
            
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        
        results['simulations'].append(num_sims)
        results['win_rate'].append(wins / num_games * 100)
        results['draw_rate'].append(draws / num_games * 100)
        results['loss_rate'].append(losses / num_games * 100)
        results['avg_moves'].append(total_moves / num_games)
        
        print(f"  Wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
        print(f"  Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
        print(f"  Losses: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['simulations'], results['win_rate'], 'o-', label='Win Rate', linewidth=2, markersize=8)
    plt.plot(results['simulations'], results['draw_rate'], 's-', label='Draw Rate', linewidth=2, markersize=8)
    plt.plot(results['simulations'], results['loss_rate'], '^-', label='Loss Rate', linewidth=2, markersize=8)
    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title(f'MCTS Performance vs Simulations\n(opponent: {opponent_sims} sims)', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['simulations'], results['avg_moves'], 'D-', color='purple', linewidth=2, markersize=8)
    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('Average Moves per Game', fontsize=12)
    plt.title('Game Length vs Thinking Time', fontsize=13)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_mcts_implementation/results/exp1_simulation_budget.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: exp1_simulation_budget.png")
    plt.close()
    
    with open('task1_mcts_implementation/results/exp1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def experiment_2_exploration_constant():
    """
    Experiment 2: Effect of exploration constant c
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Effect of Exploration Constant")
    print("="*60)
    
    ensure_results_dir()
    
    env = TicTacToe()
    c_values = [0.5, 1.0, np.sqrt(2), 2.0, 3.0]
    num_games = 25
    num_simulations = 100
    
    results = {
        'c_values': [],
        'win_rate': [],
        'draw_rate': []
    }
    
    baseline_mcts = MCTS(exploration_constant=np.sqrt(2))
    
    for c in c_values:
        print(f"\nTesting c = {c:.3f}...")
        
        test_mcts = MCTS(exploration_constant=c)
        wins = 0
        draws = 0
        
        for game in tqdm(range(num_games), desc=f"c={c:.2f}"):
            state = env.get_initial_state()
            move_count = 0
            
            while not env.is_terminal(state):
                if move_count % 2 == 0:
                    action, _ = test_mcts.search(state, env, num_simulations=num_simulations)
                else:
                    action, _ = baseline_mcts.search(state, env, num_simulations=num_simulations)
                
                state = env.get_next_state(state, action)
                move_count += 1
            
            winner = env.get_terminal_reward(state)
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
        
        win_rate = wins / num_games * 100
        draw_rate = draws / num_games * 100
        
        results['c_values'].append(c)
        results['win_rate'].append(win_rate)
        results['draw_rate'].append(draw_rate)
        
        print(f"  Win rate: {win_rate:.1f}%, Draw rate: {draw_rate:.1f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['c_values'], results['win_rate'], 'o-', linewidth=2, markersize=10, label='Win Rate')
    plt.plot(results['c_values'], results['draw_rate'], 's-', linewidth=2, markersize=10, label='Draw Rate')
    plt.axvline(x=np.sqrt(2), color='r', linestyle='--', label=f'sqrt(2) = {np.sqrt(2):.3f}', alpha=0.7)
    plt.xlabel('Exploration Constant (c)', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Effect of Exploration Constant on Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('task1_mcts_implementation/results/exp2_exploration_constant.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: exp2_exploration_constant.png")
    plt.close()
    
    with open('task1_mcts_implementation/results/exp2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def experiment_3_maze_solving():
    """
    Experiment 3: Solve maze with different MCTS configurations
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Maze Solving")
    print("="*60)
    
    ensure_results_dir()
    
    env = SimpleMaze(size=5)
    simulation_counts = [20, 50, 100, 200, 400]
    num_trials = 15
    
    results = {
        'simulations': [],
        'success_rate': [],
        'avg_steps': [],
        'avg_steps_successful': []
    }
    
    for num_sims in simulation_counts:
        print(f"\nTesting with {num_sims} simulations...")
        
        mcts = MCTS(exploration_constant=1.4, max_rollout_depth=50)
        successes = 0
        all_steps = []
        successful_steps = []
        
        for trial in tqdm(range(num_trials), desc=f"Sims={num_sims}"):
            state = env.get_initial_state()
            max_steps = 30
            
            for step in range(max_steps):
                if env.is_terminal(state):
                    break
                
                action, _ = mcts.search(state, env, num_simulations=num_sims, verbose=False)
                state = env.get_next_state(state, action)
            
            row, col, steps = state
            all_steps.append(steps)
            
            if (row, col) == env.goal:
                successes += 1
                successful_steps.append(steps)
        
        success_rate = successes / num_trials * 100
        avg_all = np.mean(all_steps)
        avg_successful = np.mean(successful_steps) if successful_steps else 0
        
        results['simulations'].append(num_sims)
        results['success_rate'].append(success_rate)
        results['avg_steps'].append(avg_all)
        results['avg_steps_successful'].append(avg_successful)
        
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Avg steps (all): {avg_all:.1f}")
        print(f"  Avg steps (successful): {avg_successful:.1f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['simulations'], results['success_rate'], 'o-', linewidth=2, markersize=10, color='green')
    ax1.set_xlabel('Number of Simulations', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Maze Success Rate vs Simulations', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    ax2.plot(results['simulations'], results['avg_steps_successful'], 's-', 
             linewidth=2, markersize=10, color='blue', label='Successful trials')
    ax2.axhline(y=8, color='r', linestyle='--', label='Optimal path ~8', alpha=0.7)
    ax2.set_xlabel('Number of Simulations', fontsize=12)
    ax2.set_ylabel('Average Steps to Goal', fontsize=12)
    ax2.set_title('Path Efficiency vs Simulations', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_mcts_implementation/results/exp3_maze_solving.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: exp3_maze_solving.png")
    plt.close()
    
    with open('task1_mcts_implementation/results/exp3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_all_experiments():
    """Run all experiments for Task 1."""
    print("\n" + "#"*70)
    print("# RUNNING ALL TASK 1 EXPERIMENTS")
    print("#"*70)
    
    results1 = experiment_1_simulation_budget()
    results2 = experiment_2_exploration_constant()
    results3 = experiment_3_maze_solving()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("\nResults saved in: task1_mcts_implementation/results/")
    print("   - exp1_simulation_budget.png")
    print("   - exp2_exploration_constant.png")
    print("   - exp3_maze_solving.png")
    print("   - exp1_results.json")
    print("   - exp2_results.json")
    print("   - exp3_results.json")
    print("\nYou can now write your Task 1 report!")
    
    return {
        'simulation_budget': results1,
        'exploration_constant': results2,
        'maze_solving': results3
    }


if __name__ == "__main__":
    results = run_all_experiments()
    # Results dictionary contains all experiment results
