"""
Run experiments comparing LLM-MCTS vs baseline.
"""

import json
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from llm_interface import LLMInterface
from llm_mcts import LLM_MCTS, baseline_llm_solve
from tasks import get_math_problems, get_problems_by_difficulty

sns.set_style("whitegrid")


def run_comparison_experiment(num_problems: int = 10, 
                               difficulty: str = "easy") -> Dict:
    """
    Compare LLM-MCTS vs baseline LLM.
    
    Args:
        num_problems: Number of problems to test
        difficulty: "easy", "medium", or "hard"
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Comparison Experiment ({difficulty} problems)")
    print(f"{'='*60}\n")
    
    # Initialize
    llm = LLMInterface(provider="anthropic")
    mcts = LLM_MCTS(llm, max_depth=4, num_simulations=8)
    
    # Get problems
    problems = get_problems_by_difficulty(difficulty)[:num_problems]
    
    results = {
        'baseline': [],
        'mcts': [],
        'problems': []
    }
    
    for i, problem in enumerate(tqdm(problems, desc="Testing problems")):
        print(f"\n--- Problem {i+1}/{len(problems)} ---")
        print(f"Q: {problem.question}")
        print(f"A: {problem.answer}")
        
        # Baseline
        print("\nBaseline LLM...")
        baseline_result = baseline_llm_solve(llm, problem.question, problem.answer)
        results['baseline'].append(baseline_result)
        print(f"  Correct: {baseline_result['evaluation']['correct']}")
        
        # MCTS
        print("\nLLM-MCTS...")
        mcts_result = mcts.search(problem.question, problem.answer)
        results['mcts'].append(mcts_result)
        print(f"  Correct: {mcts_result['evaluation']['correct']}")
        
        results['problems'].append({
            'question': problem.question,
            'answer': problem.answer,
            'category': problem.category
        })
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    # Compute statistics
    baseline_correct = sum(r['evaluation']['correct'] for r in results['baseline'])
    mcts_correct = sum(r['evaluation']['correct'] for r in results['mcts'])
    
    stats = {
        'num_problems': len(problems),
        'baseline_accuracy': baseline_correct / len(problems) * 100,
        'mcts_accuracy': mcts_correct / len(problems) * 100,
        'improvement': (mcts_correct - baseline_correct) / len(problems) * 100
    }
    
    results['statistics'] = stats
    results['llm_stats'] = llm.get_stats()
    
    return results


def visualize_results(results: Dict, save_path: str = "task2_llm_mcts/results/comparison.png"):
    """Create visualization of results."""
    import os
    os.makedirs("task2_llm_mcts/results", exist_ok=True)
    
    stats = results['statistics']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy comparison
    methods = ['Baseline LLM', 'LLM-MCTS']
    accuracies = [stats['baseline_accuracy'], stats['mcts_accuracy']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Problem-by-problem comparison
    problem_nums = list(range(1, len(results['baseline']) + 1))
    baseline_scores = [1 if r['evaluation']['correct'] else 0 for r in results['baseline']]
    mcts_scores = [1 if r['evaluation']['correct'] else 0 for r in results['mcts']]
    
    ax2.plot(problem_nums, baseline_scores, 'o-', label='Baseline', color='#FF6B6B', linewidth=2, markersize=8)
    ax2.plot(problem_nums, mcts_scores, 's-', label='MCTS', color='#4ECDC4', linewidth=2, markersize=8)
    ax2.set_xlabel('Problem Number', fontsize=12)
    ax2.set_ylabel('Correct (1) / Incorrect (0)', fontsize=12)
    ax2.set_title('Problem-by-Problem Results', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.1, 1.1])
    
    # 3. Reward distribution
    baseline_rewards = [r['evaluation']['total_reward'] for r in results['baseline']]
    mcts_rewards = [r['evaluation']['total_reward'] for r in results['mcts']]
    
    ax3.hist([baseline_rewards, mcts_rewards], label=['Baseline', 'MCTS'], 
             color=colors, alpha=0.7, bins=10, edgecolor='black')
    ax3.set_xlabel('Reward', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics
    ax4.axis('off')
    summary_text = f"""
    EXPERIMENT SUMMARY
    
    Problems Tested: {stats['num_problems']}
    
    Baseline LLM Accuracy: {stats['baseline_accuracy']:.1f}%
    LLM-MCTS Accuracy: {stats['mcts_accuracy']:.1f}%
    
    Improvement: {stats['improvement']:.1f} percentage points
    
    API Calls: {results['llm_stats']['calls']}
    Est. Cost: ${results['llm_stats']['estimated_cost']:.2f}
    
    Conclusion: {"MCTS improves performance" if stats['improvement'] > 0 else "No clear improvement"}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def save_results(results: Dict, filename: str = "task2_llm_mcts/results/results.json"):
    """Save results to JSON."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")


def run_full_experiment():
    """Run complete experimental pipeline."""
    print("\n" + "="*70)
    print("TASK 2: LLM-MCTS EXPERIMENTS")
    print("="*70)
    
    # Run on easy problems
    results = run_comparison_experiment(num_problems=10, difficulty="easy")
    
    # Save and visualize
    save_results(results)
    visualize_results(results)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nBaseline Accuracy: {results['statistics']['baseline_accuracy']:.1f}%")
    print(f"MCTS Accuracy: {results['statistics']['mcts_accuracy']:.1f}%")
    print(f"Improvement: {results['statistics']['improvement']:.1f} points")
    print(f"\nAPI Calls: {results['llm_stats']['calls']}")
    print(f"Estimated Cost: ${results['llm_stats']['estimated_cost']:.2f}")
    print(f"\nResults saved in: task2_llm_mcts/results/")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_full_experiment()