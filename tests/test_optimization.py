"""Tests for supply chain optimization modules."""

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from src.data import SyntheticDataGenerator
from src.models import TransportationOptimizer
from src.utils import set_seed


@pytest.fixture
def config():
    """Create test configuration."""
    config_dict = {
        "optimization": {
            "method": "highs",
            "timeout": 60,
            "tolerance": 1e-6,
            "max_iterations": 1000
        },
        "data": {
            "synthetic": {
                "n_warehouses": 3,
                "n_stores": 4,
                "cost_variability": 0.1
            }
        },
        "costs": {
            "transportation_cost_base": 1.0
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "test.log"
        },
        "visualization": {
            "plot_style": "default",
            "figure_size": [8, 6],
            "dpi": 100
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_data():
    """Create sample transportation data."""
    set_seed(42)
    return {
        "supply": np.array([100, 150, 120]),
        "demand": np.array([80, 90, 70, 60]),
        "costs": np.array([
            [2, 4, 5, 3],
            [3, 1, 7, 2],
            [4, 2, 6, 5]
        ])
    }


def test_data_generator(config):
    """Test synthetic data generation."""
    generator = SyntheticDataGenerator(config)
    
    # Test transportation data generation
    data = generator.generate_transportation_data()
    
    assert "supply" in data
    assert "demand" in data
    assert "costs" in data
    assert len(data["supply"]) == config.data.synthetic.n_warehouses
    assert len(data["demand"]) == config.data.synthetic.n_stores
    assert data["costs"].shape == (config.data.synthetic.n_warehouses, config.data.synthetic.n_stores)
    
    # Test inventory data generation
    inventory_data = generator.generate_inventory_data()
    
    assert "products" in inventory_data
    assert "locations" in inventory_data
    assert "demand_history" in inventory_data
    assert "inventory_levels" in inventory_data


def test_transportation_optimizer(config, sample_data):
    """Test transportation optimization."""
    optimizer = TransportationOptimizer(config)
    
    # Test optimization
    result = optimizer.optimize(
        supply=sample_data["supply"],
        demand=sample_data["demand"],
        costs=sample_data["costs"]
    )
    
    assert result.success
    assert result.total_cost > 0
    assert result.shipping_plan.shape == sample_data["costs"].shape
    assert np.allclose(np.sum(result.shipping_plan, axis=1), sample_data["supply"], atol=1e-6)
    assert np.allclose(np.sum(result.shipping_plan, axis=0), sample_data["demand"], atol=1e-6)


def test_solution_analysis(config, sample_data):
    """Test solution analysis."""
    optimizer = TransportationOptimizer(config)
    
    result = optimizer.optimize(
        supply=sample_data["supply"],
        demand=sample_data["demand"],
        costs=sample_data["costs"]
    )
    
    analysis = optimizer.analyze_solution(
        result, sample_data["supply"], sample_data["demand"], sample_data["costs"]
    )
    
    assert "total_cost" in analysis
    assert "service_level" in analysis
    assert "supply_utilization" in analysis
    assert "demand_satisfaction" in analysis
    assert "shipping_plan" in analysis
    
    assert analysis["total_cost"] == result.total_cost
    assert analysis["service_level"] == 100.0  # Should be 100% for feasible solution


def test_what_if_analysis(config, sample_data):
    """Test what-if analysis."""
    optimizer = TransportationOptimizer(config)
    
    scenarios = [
        {"name": "baseline", "demand_multiplier": 1.0},
        {"name": "high_demand", "demand_multiplier": 1.5}
    ]
    
    results = optimizer.what_if_analysis(
        sample_data["supply"],
        sample_data["demand"],
        sample_data["costs"],
        scenarios
    )
    
    assert len(results) == 2
    assert "baseline" in results
    assert "high_demand" in results
    
    # High demand scenario should have higher cost
    assert results["high_demand"].total_cost >= results["baseline"].total_cost


def test_utility_functions():
    """Test utility functions."""
    from src.utils import (
        calculate_service_level,
        calculate_inventory_turns,
        calculate_fill_rate,
        format_currency
    )
    
    # Test service level calculation
    service_level = calculate_service_level(
        np.array([100, 200]),
        np.array([100, 200]),
        np.array([0, 0])
    )
    assert service_level == 100.0
    
    # Test inventory turns calculation
    turns = calculate_inventory_turns(1200, 100)
    assert turns == 12.0
    
    # Test fill rate calculation
    fill_rate = calculate_fill_rate(80, 100)
    assert fill_rate == 80.0
    
    # Test currency formatting
    formatted = format_currency(1234.56)
    assert formatted == "$1,234.56"


def test_data_validation():
    """Test data validation functions."""
    from src.utils import validate_dataframe_schema
    import pandas as pd
    
    # Test valid DataFrame
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    assert validate_dataframe_schema(df, ["col1", "col2"])
    
    # Test missing required column
    with pytest.raises(ValueError):
        validate_dataframe_schema(df, ["col1", "col3"])


if __name__ == "__main__":
    pytest.main([__file__])
