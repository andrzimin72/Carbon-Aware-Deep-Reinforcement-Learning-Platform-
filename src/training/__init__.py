"""
Training and evaluation package for the Carbon-Aware Deep Reinforcement Learning Platform.

Public API:
    from src.training import (
        train_agent,
        PPOAgent,
        run_statistical_ablation,
        run_three_way_ablation,
        ForecastAwareBaseline,
        RealForecastBaseline,
    )
"""

from src.training.replay_trainer import train_agent, PPOAgent
from src.training.statistical_ablation import run_statistical_ablation
from src.training.three_way_ablation import run_three_way_ablation
from src.training.forecast_baseline import ForecastAwareBaseline
from src.training.real_forecast_baseline import RealForecastBaseline

__all__ = [
    "train_agent",
    "PPOAgent",
    "run_statistical_ablation",
    "run_three_way_ablation",
    "ForecastAwareBaseline",
    "RealForecastBaseline",
]
