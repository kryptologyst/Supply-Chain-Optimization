#!/usr/bin/env python3
"""Main script for supply chain optimization demonstration."""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from omegaconf import DictConfig, OmegaConf

from src.data import SyntheticDataGenerator
from src.evaluation import SupplyChainEvaluator
from src.models import TransportationOptimizer
from src.utils import load_config, setup_logging, set_seed
from src.visualization import SupplyChainVisualizer


def main():
    """Main function to demonstrate supply chain optimization."""
    # Set up paths
    project_root = Path(__file__).parent
    config_path = project_root / "configs" / "transportation.yaml"
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Set up logging
    logger = setup_logging(config)
    logger.info("Starting Supply Chain Optimization Demo")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directories
    output_dir = project_root / "assets"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    data_generator = SyntheticDataGenerator(config)
    optimizer = TransportationOptimizer(config)
    evaluator = SupplyChainEvaluator(config)
    visualizer = SupplyChainVisualizer(config)
    
    # Generate synthetic data
    logger.info("Generating synthetic transportation data...")
    data = data_generator.generate_transportation_data()
    
    # Solve transportation problem with different methods
    logger.info("Solving transportation problem...")
    results = {}
    
    methods = ["highs", "pulp", "ortools"]
    for method in methods:
        try:
            logger.info(f"Solving with {method} method...")
            result = optimizer.optimize(
                supply=data["supply"],
                demand=data["demand"],
                costs=data["costs"],
                method=method
            )
            
            if result.success:
                # Analyze solution
                analysis = optimizer.analyze_solution(
                    result, data["supply"], data["demand"], data["costs"]
                )
                results[method] = analysis
                logger.info(f"{method} method completed successfully")
            else:
                logger.warning(f"{method} method failed: {result.message}")
        
        except Exception as e:
            logger.error(f"Error with {method} method: {e}")
    
    if not results:
        logger.error("No successful optimizations. Exiting.")
        return
    
    # Evaluate results
    logger.info("Evaluating optimization results...")
    evaluation_results = {}
    for method, result in results.items():
        evaluation_results[method] = evaluator.evaluate_transportation(result)
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(results)
    logger.info("Leaderboard created:")
    print(leaderboard[["rank", "approach", "total_cost", "service_level", "solve_time"]].to_string())
    
    # Perform what-if analysis
    logger.info("Performing what-if analysis...")
    base_result = list(results.values())[0]  # Use first successful result as baseline
    
    scenarios = [
        {"name": "Increased_Demand", "demand_multiplier": 1.2},
        {"name": "Reduced_Supply", "supply_multiplier": 0.8},
        {"name": "Higher_Costs", "cost_multiplier": 1.5},
        {"name": "Lower_Costs", "cost_multiplier": 0.7}
    ]
    
    what_if_results = optimizer.what_if_analysis(
        data["supply"], data["demand"], data["costs"], scenarios
    )
    
    # Convert what-if results to analysis format
    what_if_analysis = {}
    for scenario_name, result in what_if_results.items():
        if result.success:
            analysis = optimizer.analyze_solution(
                result, data["supply"], data["demand"], data["costs"]
            )
            what_if_analysis[scenario_name] = analysis
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Plot individual results
    for method, result in results.items():
        visualizer.plot_transportation_solution(
            result, str(output_dir / f"{method}_solution.png")
        )
    
    # Plot leaderboard
    visualizer.plot_leaderboard(
        leaderboard, str(output_dir / "leaderboard.png")
    )
    
    # Plot what-if analysis
    if what_if_analysis:
        visualizer.plot_what_if_analysis(
            what_if_analysis, str(output_dir / "what_if_analysis.png")
        )
    
    # Create comprehensive dashboard
    visualizer.create_dashboard(results, str(output_dir / "dashboard"))
    
    # Save results
    logger.info("Saving results...")
    
    # Save leaderboard
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    
    # Save detailed results
    for method, result in results.items():
        result["shipping_plan"].to_csv(output_dir / f"{method}_shipping_plan.csv")
        
        # Save evaluation metrics
        evaluation_df = pd.DataFrame([evaluation_results[method]])
        evaluation_df.to_csv(output_dir / f"{method}_evaluation.csv", index=False)
    
    # Generate summary report
    generate_summary_report(results, evaluation_results, leaderboard, output_dir, logger)
    
    logger.info("Supply Chain Optimization Demo completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


def generate_summary_report(
    results: dict,
    evaluation_results: dict,
    leaderboard: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Generate a summary report of the optimization results."""
    report_path = output_dir / "summary_report.txt"
    
    with open(report_path, "w") as f:
        f.write("SUPPLY CHAIN OPTIMIZATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXPERIMENTAL RESEARCH PROJECT - NOT FOR AUTOMATED DECISIONS\n")
        f.write("All recommendations require human review before implementation.\n\n")
        
        f.write("OPTIMIZATION METHODS TESTED:\n")
        for method in results.keys():
            f.write(f"- {method.upper()}\n")
        f.write("\n")
        
        f.write("LEADERBOARD RESULTS:\n")
        f.write(leaderboard.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED METRICS:\n")
        for method, metrics in evaluation_results.items():
            f.write(f"\n{method.upper()} METHOD:\n")
            f.write("-" * 20 + "\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.2f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
        
        f.write("\n\nKEY INSIGHTS:\n")
        best_method = leaderboard.iloc[0]["approach"]
        best_cost = leaderboard.iloc[0]["total_cost"]
        best_service = leaderboard.iloc[0]["service_level"]
        
        f.write(f"- Best performing method: {best_method}\n")
        f.write(f"- Lowest total cost: ${best_cost:.2f}\n")
        f.write(f"- Best service level: {best_service:.1f}%\n")
        
        f.write("\n\nLIMITATIONS AND DISCLAIMERS:\n")
        f.write("- This is experimental software for research purposes only\n")
        f.write("- All optimization results require professional review\n")
        f.write("- Real-world constraints may not be fully captured\n")
        f.write("- Historical data quality affects solution accuracy\n")
        f.write("- No warranty or guarantee of optimality in production\n")
    
    logger.info(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    main()
