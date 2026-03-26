"""Evaluation metrics and business KPIs for supply chain optimization."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..utils import (
    calculate_fill_rate,
    calculate_inventory_turns,
    calculate_service_level,
    format_currency
)


class SupplyChainEvaluator:
    """Evaluator for supply chain optimization results."""
    
    def __init__(self, config: DictConfig):
        """Initialize evaluator with configuration.
        
        Args:
            config: Configuration object with evaluation parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_transportation(
        self,
        result: Dict,
        baseline_result: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Evaluate transportation optimization results.
        
        Args:
            result: Transportation optimization result dictionary.
            baseline_result: Baseline result for comparison.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics["total_cost"] = result.get("total_cost", 0)
        metrics["service_level"] = result.get("service_level", 0)
        metrics["solve_time"] = result.get("solve_time", 0)
        
        # Calculate additional metrics
        shipping_plan = result.get("shipping_plan", np.array([]))
        if shipping_plan.size > 0:
            metrics["total_units_shipped"] = np.sum(shipping_plan)
            metrics["average_cost_per_unit"] = (
                metrics["total_cost"] / metrics["total_units_shipped"] 
                if metrics["total_units_shipped"] > 0 else 0
            )
            
            # Calculate utilization metrics
            supply_utilization = result.get("supply_utilization", np.array([]))
            if supply_utilization.size > 0:
                metrics["avg_supply_utilization"] = np.mean(supply_utilization)
                metrics["min_supply_utilization"] = np.min(supply_utilization)
                metrics["max_supply_utilization"] = np.max(supply_utilization)
            
            demand_satisfaction = result.get("demand_satisfaction", np.array([]))
            if demand_satisfaction.size > 0:
                metrics["avg_demand_satisfaction"] = np.mean(demand_satisfaction)
                metrics["min_demand_satisfaction"] = np.min(demand_satisfaction)
        
        # Compare with baseline if provided
        if baseline_result:
            metrics["cost_reduction"] = self._calculate_percentage_change(
                baseline_result.get("total_cost", 0),
                metrics["total_cost"]
            )
            metrics["service_level_improvement"] = self._calculate_percentage_change(
                baseline_result.get("service_level", 0),
                metrics["service_level"]
            )
        
        return metrics
    
    def evaluate_inventory(
        self,
        result: Dict,
        baseline_result: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Evaluate inventory optimization results.
        
        Args:
            result: Inventory optimization result dictionary.
            baseline_result: Baseline result for comparison.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        metrics = {}
        
        # Basic inventory metrics
        metrics["total_inventory_cost"] = result.get("total_cost", 0)
        metrics["service_level"] = result.get("service_level", 0)
        metrics["inventory_turns"] = result.get("inventory_turns", 0)
        metrics["fill_rate"] = result.get("fill_rate", 0)
        
        # Calculate additional metrics
        inventory_levels = result.get("inventory_levels", pd.DataFrame())
        if not inventory_levels.empty:
            metrics["avg_inventory_level"] = inventory_levels["current_stock"].mean()
            metrics["total_inventory_value"] = (
                inventory_levels["current_stock"] * 
                inventory_levels.get("unit_cost", 1)
            ).sum()
            
            # Stockout analysis
            stockouts = result.get("stockouts", pd.DataFrame())
            if not stockouts.empty:
                metrics["stockout_frequency"] = len(stockouts) / len(inventory_levels)
                metrics["avg_stockout_duration"] = stockouts.get("duration_days", 0).mean()
        
        # Forecasting accuracy if available
        forecast_metrics = result.get("forecast_metrics", {})
        if forecast_metrics:
            metrics["forecast_mape"] = forecast_metrics.get("mape", 0)
            metrics["forecast_smape"] = forecast_metrics.get("smape", 0)
            metrics["forecast_bias"] = forecast_metrics.get("bias", 0)
        
        # Compare with baseline if provided
        if baseline_result:
            metrics["cost_reduction"] = self._calculate_percentage_change(
                baseline_result.get("total_inventory_cost", 0),
                metrics["total_inventory_cost"]
            )
            metrics["inventory_reduction"] = self._calculate_percentage_change(
                baseline_result.get("avg_inventory_level", 0),
                metrics["avg_inventory_level"]
            )
            metrics["service_level_improvement"] = self._calculate_percentage_change(
                baseline_result.get("service_level", 0),
                metrics["service_level"]
            )
        
        return metrics
    
    def evaluate_workforce(
        self,
        result: Dict,
        baseline_result: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Evaluate workforce optimization results.
        
        Args:
            result: Workforce optimization result dictionary.
            baseline_result: Baseline result for comparison.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        metrics = {}
        
        # Basic workforce metrics
        metrics["total_labor_cost"] = result.get("total_cost", 0)
        metrics["utilization_rate"] = result.get("utilization_rate", 0)
        metrics["skill_matching_score"] = result.get("skill_matching_score", 0)
        
        # Calculate additional metrics
        assignments = result.get("assignments", pd.DataFrame())
        if not assignments.empty:
            metrics["total_assignments"] = len(assignments)
            metrics["avg_hours_per_employee"] = assignments.get("hours", 0).mean()
            metrics["overtime_hours"] = assignments[
                assignments.get("hours", 0) > 40
            ].get("hours", 0).sum() - 40 * len(assignments[assignments.get("hours", 0) > 40])
            
            # Skill utilization
            skill_utilization = result.get("skill_utilization", {})
            if skill_utilization:
                metrics["avg_skill_utilization"] = np.mean(list(skill_utilization.values()))
                metrics["underutilized_skills"] = sum(1 for v in skill_utilization.values() if v < 0.5)
        
        # Fairness metrics
        fairness_metrics = result.get("fairness_metrics", {})
        if fairness_metrics:
            metrics["gender_parity_score"] = fairness_metrics.get("gender_parity", 0)
            metrics["age_diversity_score"] = fairness_metrics.get("age_diversity", 0)
        
        # Compare with baseline if provided
        if baseline_result:
            metrics["cost_reduction"] = self._calculate_percentage_change(
                baseline_result.get("total_labor_cost", 0),
                metrics["total_labor_cost"]
            )
            metrics["utilization_improvement"] = self._calculate_percentage_change(
                baseline_result.get("utilization_rate", 0),
                metrics["utilization_rate"]
            )
        
        return metrics
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict],
        metric_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Create a leaderboard comparing different optimization approaches.
        
        Args:
            results: Dictionary mapping approach names to result dictionaries.
            metric_weights: Optional weights for different metrics.
            
        Returns:
            DataFrame with leaderboard results.
        """
        if not results:
            return pd.DataFrame()
        
        # Default weights if not provided
        if metric_weights is None:
            metric_weights = {
                "total_cost": -0.3,  # Negative weight for minimization
                "service_level": 0.3,
                "solve_time": -0.1,
                "utilization_rate": 0.2,
                "inventory_turns": 0.1
            }
        
        leaderboard_data = []
        
        for approach_name, result in results.items():
            # Extract metrics based on result type
            if "total_cost" in result and "service_level" in result:
                # Transportation or general optimization result
                metrics = self.evaluate_transportation(result)
            elif "total_inventory_cost" in result:
                # Inventory optimization result
                metrics = self.evaluate_inventory(result)
            elif "total_labor_cost" in result:
                # Workforce optimization result
                metrics = self.evaluate_workforce(result)
            else:
                # Generic result
                metrics = result
            
            # Calculate composite score
            composite_score = 0
            for metric_name, weight in metric_weights.items():
                if metric_name in metrics:
                    # Normalize metric to 0-1 scale for scoring
                    normalized_value = self._normalize_metric(metric_name, metrics[metric_name])
                    composite_score += weight * normalized_value
            
            leaderboard_data.append({
                "approach": approach_name,
                "composite_score": composite_score,
                **metrics
            })
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by composite score (descending)
        leaderboard_df = leaderboard_df.sort_values("composite_score", ascending=False)
        leaderboard_df["rank"] = range(1, len(leaderboard_df) + 1)
        
        return leaderboard_df
    
    def _calculate_percentage_change(self, baseline: float, current: float) -> float:
        """Calculate percentage change from baseline.
        
        Args:
            baseline: Baseline value.
            current: Current value.
            
        Returns:
            Percentage change.
        """
        if baseline == 0:
            return 0.0 if current == 0 else float('inf')
        
        return ((current - baseline) / baseline) * 100
    
    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric value to 0-1 scale for scoring.
        
        Args:
            metric_name: Name of the metric.
            value: Metric value.
            
        Returns:
            Normalized value between 0 and 1.
        """
        # Define normalization ranges for different metrics
        normalization_ranges = {
            "total_cost": (0, 1000000),  # Assume max cost of 1M
            "service_level": (0, 100),
            "solve_time": (0, 300),  # Assume max solve time of 5 minutes
            "utilization_rate": (0, 1),
            "inventory_turns": (0, 12),  # Assume max 12 turns per year
            "fill_rate": (0, 100),
            "inventory_turns": (0, 12)
        }
        
        if metric_name not in normalization_ranges:
            return 0.5  # Default neutral value
        
        min_val, max_val = normalization_ranges[metric_name]
        
        # Clamp value to range
        clamped_value = max(min_val, min(max_val, value))
        
        # Normalize to 0-1
        return (clamped_value - min_val) / (max_val - min_val)
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Dict],
        output_path: str
    ) -> None:
        """Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary mapping approach names to results.
            output_path: Path to save the report.
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Create leaderboard
        leaderboard = self.create_leaderboard(results)
        leaderboard.to_csv(f"{output_path}/leaderboard.csv", index=False)
        
        # Generate detailed metrics for each approach
        for approach_name, result in results.items():
            if "total_cost" in result:
                metrics = self.evaluate_transportation(result)
            elif "total_inventory_cost" in result:
                metrics = self.evaluate_inventory(result)
            elif "total_labor_cost" in result:
                metrics = self.evaluate_workforce(result)
            else:
                metrics = result
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"{output_path}/{approach_name}_metrics.csv", index=False)
        
        self.logger.info(f"Evaluation report saved to {output_path}")
