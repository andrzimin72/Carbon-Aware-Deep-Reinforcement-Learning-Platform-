"""
Interpretability package for the Carbon-Aware Deep Reinforcement Learning Platform.

Provides tools for understanding what the recurrent PPO agent has learned
about grid dynamics, including:

1. **Hidden-state clustering** (t-SNE) — reveals whether the LSTM separates
   clean vs. dirty grid regimes in its internal representation.

2. **Temporal attribution** (occlusion + gradient saliency) — identifies
   which historical hours most influence scheduling decisions.

3. **Multi-horizon linear probing** — tests whether the hidden state
   encodes predictive information about future carbon intensity.

4. **Policy analysis** — examines action distributions by grid regime
   and confidence-reward calibration.

Public API:
    from src.interpretability import (
        LSTMInterpreter,
        InterpreterConfig,
        InterpreterReport,
        TrajectoryCollector,
        Trajectories,
    )

    config = InterpreterConfig(output_dir="outputs/interpretability")
    interpreter = LSTMInterpreter(policy, env, config)
    report = interpreter.generate_full_report(num_episodes=30)
    print(report.summary())

Output:
    - Publication-quality PNG/PDF figures
    - JSON report with all numerical results
    - Graceful degradation if individual analyses fail

Dependencies:
    - torch, numpy, matplotlib, seaborn, scikit-learn
    - Optional: umap-learn (for UMAP embeddings)
"""

from __future__ import annotations

from src.interpretability.interpreter import (
    InterpreterConfig,
    InterpreterReport,
    LSTMInterpreter,
    Trajectories,
    TrajectoryCollector,
)

__all__ = [
    "LSTMInterpreter",
    "InterpreterConfig",
    "InterpreterReport",
    "TrajectoryCollector",
    "Trajectories",
]

__version__ = "1.0.0"
