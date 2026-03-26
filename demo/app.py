"""Streamlit demo application for supply chain optimization."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import SyntheticDataGenerator
from src.evaluation import SupplyChainEvaluator
from src.models import TransportationOptimizer
from src.utils import load_config, set_seed
from src.visualization import SupplyChainVisualizer


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Supply Chain Optimization",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add disclaimer
    st.error("""
    **EXPERIMENTAL RESEARCH PROJECT - NOT FOR AUTOMATED DECISIONS**
    
    This application is for educational and research purposes only. 
    All optimization results require human review before implementation 
    in production systems.
    """)


def load_configuration():
    """Load configuration and initialize components."""
    config_path = Path(__file__).parent.parent / "configs" / "transportation.yaml"
    config = OmegaConf.load(config_path)
    
    # Set random seed
    set_seed(42)
    
    return config


def initialize_components(config: DictConfig):
    """Initialize all components."""
    return {
        "data_generator": SyntheticDataGenerator(config),
        "optimizer": TransportationOptimizer(config),
        "evaluator": SupplyChainEvaluator(config),
        "visualizer": SupplyChainVisualizer(config)
    }


def main():
    """Main Streamlit application."""
    setup_page()
    
    # Load configuration
    config = load_configuration()
    components = initialize_components(config)
    
    # Title and description
    st.title("📦 Supply Chain Optimization Demo")
    st.markdown("""
    This interactive demo showcases transportation optimization techniques using linear programming.
    Experiment with different parameters and see how they affect the optimal solution.
    """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_warehouses = st.sidebar.slider("Number of Warehouses", 2, 10, 5)
    n_stores = st.sidebar.slider("Number of Stores", 3, 20, 15)
    cost_variability = st.sidebar.slider("Cost Variability", 0.1, 0.5, 0.15)
    
    # Optimization parameters
    st.sidebar.subheader("Optimization Parameters")
    method = st.sidebar.selectbox(
        "Optimization Method",
        ["highs", "pulp", "ortools"],
        index=0
    )
    timeout = st.sidebar.slider("Timeout (seconds)", 30, 600, 300)
    
    # Generate data button
    if st.sidebar.button("Generate New Data", type="primary"):
        st.session_state.data_generated = False
    
    # Generate or load data
    if "data_generated" not in st.session_state or not st.session_state.data_generated:
        with st.spinner("Generating synthetic data..."):
            # Update config with user parameters
            config.data.synthetic.n_warehouses = n_warehouses
            config.data.synthetic.n_stores = n_stores
            config.data.synthetic.cost_variability = cost_variability
            config.optimization.timeout = timeout
            
            # Generate data
            data = components["data_generator"].generate_transportation_data()
            st.session_state.data = data
            st.session_state.data_generated = True
    
    data = st.session_state.data
    
    # Display data overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Warehouses", n_warehouses)
        st.metric("Total Supply", f"{np.sum(data['supply']):,}")
    
    with col2:
        st.metric("Stores", n_stores)
        st.metric("Total Demand", f"{np.sum(data['demand']):,}")
    
    with col3:
        st.metric("Supply/Demand Ratio", f"{np.sum(data['supply']) / np.sum(data['demand']):.2f}")
        st.metric("Avg Cost per Unit", f"${np.mean(data['costs']):.2f}")
    
    # Run optimization
    if st.button("Run Optimization", type="primary"):
        with st.spinner(f"Solving with {method} method..."):
            try:
                result = components["optimizer"].optimize(
                    supply=data["supply"],
                    demand=data["demand"],
                    costs=data["costs"],
                    method=method
                )
                
                if result.success:
                    # Analyze solution
                    analysis = components["optimizer"].analyze_solution(
                        result, data["supply"], data["demand"], data["costs"]
                    )
                    st.session_state.analysis = analysis
                    st.session_state.result = result
                    st.success("Optimization completed successfully!")
                else:
                    st.error(f"Optimization failed: {result.message}")
                    
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
    
    # Display results if available
    if "analysis" in st.session_state:
        analysis = st.session_state.analysis
        result = st.session_state.result
        
        st.header("📊 Optimization Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost",
                f"${analysis['total_cost']:,.2f}",
                delta=f"{analysis['solve_time']:.2f}s solve time"
            )
        
        with col2:
            st.metric(
                "Service Level",
                f"{analysis['service_level']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Avg Supply Utilization",
                f"{np.mean(analysis['supply_utilization']) * 100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Avg Demand Satisfaction",
                f"{np.mean(analysis['demand_satisfaction']) * 100:.1f}%"
            )
        
        # Shipping plan
        st.subheader("🚚 Shipping Plan")
        shipping_plan = analysis["shipping_plan"]
        st.dataframe(shipping_plan, use_container_width=True)
        
        # Cost breakdown
        st.subheader("💰 Cost Breakdown")
        cost_breakdown = analysis["cost_breakdown"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cost by Route**")
            cost_by_route_df = pd.DataFrame({
                "Store": [f"Store {i+1}" for i in range(len(analysis["total_cost_by_route"]))],
                "Total Cost": analysis["total_cost_by_route"]
            })
            st.dataframe(cost_by_route_df, use_container_width=True)
        
        with col2:
            st.write("**Cost by Warehouse**")
            cost_by_warehouse_df = pd.DataFrame({
                "Warehouse": [f"Warehouse {i+1}" for i in range(len(analysis["total_cost_by_warehouse"]))],
                "Total Cost": analysis["total_cost_by_warehouse"]
            })
            st.dataframe(cost_by_warehouse_df, use_container_width=True)
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        # Create and display plots
        fig = components["visualizer"].plot_transportation_solution(analysis)
        st.pyplot(fig)
        
        # What-if analysis
        st.subheader("🔍 What-If Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            demand_multiplier = st.slider("Demand Multiplier", 0.5, 2.0, 1.0, 0.1)
            supply_multiplier = st.slider("Supply Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            cost_multiplier = st.slider("Cost Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("Run What-If Analysis"):
            with st.spinner("Running what-if analysis..."):
                scenarios = [
                    {
                        "name": "Current",
                        "demand_multiplier": 1.0,
                        "supply_multiplier": 1.0,
                        "cost_multiplier": 1.0
                    },
                    {
                        "name": "Modified",
                        "demand_multiplier": demand_multiplier,
                        "supply_multiplier": supply_multiplier,
                        "cost_multiplier": cost_multiplier
                    }
                ]
                
                what_if_results = components["optimizer"].what_if_analysis(
                    data["supply"], data["demand"], data["costs"], scenarios
                )
                
                # Display comparison
                comparison_data = []
                for scenario_name, result in what_if_results.items():
                    if result.success:
                        analysis_whatif = components["optimizer"].analyze_solution(
                            result, data["supply"], data["demand"], data["costs"]
                        )
                        comparison_data.append({
                            "Scenario": scenario_name,
                            "Total Cost": f"${analysis_whatif['total_cost']:,.2f}",
                            "Service Level": f"{analysis_whatif['service_level']:.1f}%",
                            "Solve Time": f"{analysis_whatif['solve_time']:.2f}s"
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Plot comparison
                    fig_whatif = components["visualizer"].plot_what_if_analysis(
                        {name: analysis for name, analysis in zip(
                            [d["Scenario"] for d in comparison_data],
                            [analysis_whatif for analysis_whatif in [components["optimizer"].analyze_solution(
                                what_if_results[d["Scenario"]], data["supply"], data["demand"], data["costs"]
                            ) for d in comparison_data]]
                        )}
                    )
                    st.pyplot(fig_whatif)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This is experimental software for research and educational purposes only. 
    All optimization results require professional review before implementation in production systems.
    """)
    
    # Add some sample data display
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("📋 Raw Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Supply Data**")
            supply_df = pd.DataFrame({
                "Warehouse": [f"Warehouse {i+1}" for i in range(len(data["supply"]))],
                "Capacity": data["supply"]
            })
            st.dataframe(supply_df, use_container_width=True)
        
        with col2:
            st.write("**Demand Data**")
            demand_df = pd.DataFrame({
                "Store": [f"Store {i+1}" for i in range(len(data["demand"]))],
                "Demand": data["demand"]
            })
            st.dataframe(demand_df, use_container_width=True)
        
        st.write("**Cost Matrix**")
        cost_df = pd.DataFrame(
            data["costs"],
            index=[f"Warehouse {i+1}" for i in range(data["costs"].shape[0])],
            columns=[f"Store {i+1}" for i in range(data["costs"].shape[1])]
        )
        st.dataframe(cost_df, use_container_width=True)


if __name__ == "__main__":
    main()
