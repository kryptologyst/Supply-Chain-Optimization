"""Visualization utilities for supply chain optimization results."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from ..utils import format_currency


class SupplyChainVisualizer:
    """Visualizer for supply chain optimization results."""
    
    def __init__(self, config: DictConfig):
        """Initialize visualizer with configuration.
        
        Args:
            config: Configuration object with visualization parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use(self.config.visualization.plot_style)
        sns.set_palette("husl")
    
    def plot_transportation_solution(
        self,
        result: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot transportation optimization solution.
        
        Args:
            result: Transportation optimization result dictionary.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.visualization.figure_size)
        fig.suptitle("Transportation Optimization Results", fontsize=16, fontweight='bold')
        
        shipping_plan = result.get("shipping_plan", pd.DataFrame())
        costs = result.get("cost_breakdown", np.array([]))
        
        # Plot 1: Shipping plan heatmap
        if not shipping_plan.empty:
            sns.heatmap(
                shipping_plan.iloc[:-1, :-1],  # Exclude totals row/column
                annot=True,
                fmt='.0f',
                cmap='Blues',
                ax=axes[0, 0],
                cbar_kws={'label': 'Units Shipped'}
            )
            axes[0, 0].set_title("Shipping Plan (Units)")
            axes[0, 0].set_xlabel("Stores")
            axes[0, 0].set_ylabel("Warehouses")
        
        # Plot 2: Cost breakdown heatmap
        if costs.size > 0:
            sns.heatmap(
                costs,
                annot=True,
                fmt='.2f',
                cmap='Reds',
                ax=axes[0, 1],
                cbar_kws={'label': 'Cost ($)'}
            )
            axes[0, 1].set_title("Cost Breakdown")
            axes[0, 1].set_xlabel("Stores")
            axes[0, 1].set_ylabel("Warehouses")
        
        # Plot 3: Supply utilization
        supply_utilization = result.get("supply_utilization", np.array([]))
        if supply_utilization.size > 0:
            warehouse_names = [f"W{i+1}" for i in range(len(supply_utilization))]
            bars = axes[1, 0].bar(warehouse_names, supply_utilization * 100)
            axes[1, 0].set_title("Supply Utilization")
            axes[1, 0].set_ylabel("Utilization (%)")
            axes[1, 0].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, util in zip(bars, supply_utilization * 100):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                              f'{util:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Demand satisfaction
        demand_satisfaction = result.get("demand_satisfaction", np.array([]))
        if demand_satisfaction.size > 0:
            store_names = [f"S{i+1}" for i in range(len(demand_satisfaction))]
            bars = axes[1, 1].bar(store_names, demand_satisfaction * 100)
            axes[1, 1].set_title("Demand Satisfaction")
            axes[1, 1].set_ylabel("Satisfaction (%)")
            axes[1, 1].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, sat in zip(bars, demand_satisfaction * 100):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                              f'{sat:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            self.logger.info(f"Transportation solution plot saved to {save_path}")
        
        return fig
    
    def plot_leaderboard(
        self,
        leaderboard: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot optimization leaderboard.
        
        Args:
            leaderboard: Leaderboard DataFrame.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config.visualization.figure_size)
        fig.suptitle("Optimization Leaderboard", fontsize=16, fontweight='bold')
        
        # Plot 1: Composite scores
        axes[0].barh(
            leaderboard["approach"],
            leaderboard["composite_score"]
        )
        axes[0].set_title("Composite Scores")
        axes[0].set_xlabel("Score")
        
        # Plot 2: Key metrics comparison
        if "total_cost" in leaderboard.columns:
            metrics_to_plot = ["total_cost", "service_level", "solve_time"]
            available_metrics = [m for m in metrics_to_plot if m in leaderboard.columns]
            
            if available_metrics:
                x = np.arange(len(leaderboard))
                width = 0.25
                
                for i, metric in enumerate(available_metrics):
                    # Normalize metrics for comparison
                    values = leaderboard[metric].values
                    if metric == "total_cost" or metric == "solve_time":
                        # For minimization metrics, invert for visualization
                        values = 1 / (values + 1e-6)
                    
                    axes[1].bar(
                        x + i * width,
                        values,
                        width,
                        label=metric.replace("_", " ").title()
                    )
                
                axes[1].set_title("Key Metrics Comparison")
                axes[1].set_xlabel("Approach")
                axes[1].set_ylabel("Normalized Value")
                axes[1].set_xticks(x + width)
                axes[1].set_xticklabels(leaderboard["approach"], rotation=45)
                axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            self.logger.info(f"Leaderboard plot saved to {save_path}")
        
        return fig
    
    def plot_what_if_analysis(
        self,
        scenarios: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot what-if analysis results.
        
        Args:
            scenarios: Dictionary mapping scenario names to results.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.visualization.figure_size)
        fig.suptitle("What-If Analysis", fontsize=16, fontweight='bold')
        
        scenario_names = list(scenarios.keys())
        
        # Extract metrics
        costs = [scenarios[name].get("total_cost", 0) for name in scenario_names]
        service_levels = [scenarios[name].get("service_level", 0) for name in scenario_names]
        solve_times = [scenarios[name].get("solve_time", 0) for name in scenario_names]
        
        # Plot 1: Cost comparison
        axes[0, 0].bar(scenario_names, costs)
        axes[0, 0].set_title("Total Cost by Scenario")
        axes[0, 0].set_ylabel("Cost ($)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Service level comparison
        axes[0, 1].bar(scenario_names, service_levels)
        axes[0, 1].set_title("Service Level by Scenario")
        axes[0, 1].set_ylabel("Service Level (%)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Solve time comparison
        axes[1, 0].bar(scenario_names, solve_times)
        axes[1, 0].set_title("Solve Time by Scenario")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Cost vs Service Level scatter
        axes[1, 1].scatter(costs, service_levels, s=100, alpha=0.7)
        for i, name in enumerate(scenario_names):
            axes[1, 1].annotate(name, (costs[i], service_levels[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_title("Cost vs Service Level")
        axes[1, 1].set_xlabel("Total Cost ($)")
        axes[1, 1].set_ylabel("Service Level (%)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            self.logger.info(f"What-if analysis plot saved to {save_path}")
        
        return fig
    
    def create_dashboard(
        self,
        results: Dict[str, Dict],
        output_path: str
    ) -> None:
        """Create comprehensive dashboard with all visualizations.
        
        Args:
            results: Dictionary mapping approach names to results.
            output_path: Directory path to save dashboard plots.
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Generate plots for each result type
        for approach_name, result in results.items():
            if "shipping_plan" in result:
                # Transportation result
                self.plot_transportation_solution(
                    result,
                    f"{output_path}/{approach_name}_transportation.png"
                )
        
        self.logger.info(f"Dashboard plots saved to {output_path}")
