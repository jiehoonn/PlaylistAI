"""
Demo script showing evaluation capabilities.

This demonstrates:
1. Quick single algorithm evaluation
2. Algorithm comparison 
3. Parameter tuning evaluation
4. Results interpretation

Usage:
    python scripts/demo_eval.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_command(cmd, description):
    """Run a command and show its output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"💻 Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
    return result.returncode

def main():
    """Run evaluation demos."""
    print("🎵 PLAYLIST AI - EVALUATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows how to evaluate recommendation quality")
    print()
    
    # Activate virtual environment prefix
    venv_cmd = "source .venv/bin/activate &&"
    
    # Demo 1: Quick single algorithm test
    run_command(
        f"{venv_cmd} python scripts/eval_recs.py --algorithm hybrid --k 5 --num-seeds 10",
        "Quick test: Hybrid algorithm with 10 seeds"
    )
    
    # Demo 2: Compare MMR parameters 
    print(f"\n{'='*60}")
    print("🔬 PARAMETER TUNING EXAMPLE")
    print("Testing different MMR λ values...")
    print(f"{'='*60}")
    
    for lambda_val in [0.1, 0.3, 0.5]:
        run_command(
            f"{venv_cmd} python scripts/eval_recs.py --algorithm playlist --lambda-mmr {lambda_val} --k 10 --num-seeds 15",
            f"MMR with λ={lambda_val} (λ=0.1: diverse, λ=0.5: similar)"
        )
    
    # Demo 3: Full comparison (smaller for demo)
    run_command(
        f"{venv_cmd} python scripts/eval_recs.py --compare-all --k 10 --num-seeds 20 --export-results",
        "Full algorithm comparison with results export"
    )
    
    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETE!")
    print(f"{'='*60}")
    print("📊 Check artifacts/eval/ for detailed results")
    print("🎯 Key insights:")
    print("   • Hybrid algorithms balance similarity + metadata")
    print("   • MMR λ controls similarity/diversity tradeoff")
    print("   • Tempo coherence is crucial for playlist flow")
    print("   • Artist diversity prevents repetitive playlists")
    print()
    print("💡 Use these metrics to optimize your recommendation system!")

if __name__ == "__main__":
    main()
