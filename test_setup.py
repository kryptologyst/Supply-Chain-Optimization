#!/usr/bin/env python3
"""Simple test script to verify the supply chain optimization setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data import SyntheticDataGenerator
        from src.models import TransportationOptimizer
        from src.evaluation import SupplyChainEvaluator
        from src.visualization import SupplyChainVisualizer
        from src.utils import set_seed, load_config
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from src.data import SyntheticDataGenerator
        from src.models import TransportationOptimizer
        from src.utils import set_seed, load_config
        from omegaconf import OmegaConf
        
        # Create minimal config
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
        config = OmegaConf.create(config_dict)
        
        # Test data generation
        set_seed(42)
        generator = SyntheticDataGenerator(config)
        data = generator.generate_transportation_data()
        
        print(f"✓ Generated data: {data['n_warehouses']} warehouses, {data['n_stores']} stores")
        
        # Test optimization
        optimizer = TransportationOptimizer(config)
        
        # Ensure feasible problem by adjusting supply if needed
        total_demand = np.sum(data["demand"])
        total_supply = np.sum(data["supply"])
        if total_supply < total_demand:
            # Scale up supply to meet demand
            scale_factor = total_demand / total_supply
            data["supply"] = (data["supply"] * scale_factor).astype(int)
        
        result = optimizer.optimize(
            supply=data["supply"],
            demand=data["demand"],
            costs=data["costs"]
        )
        
        if result.success:
            print(f"✓ Optimization successful: ${result.total_cost:.2f}")
        else:
            print(f"✗ Optimization failed: {result.message}")
            return False
        
        # Test analysis
        analysis = optimizer.analyze_solution(
            result, data["supply"], data["demand"], data["costs"]
        )
        print(f"✓ Analysis completed: {analysis['service_level']:.1f}% service level")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Supply Chain Optimization - Basic Tests")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please check dependencies.")
        return False
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    if not functionality_ok:
        print("\n❌ Functionality tests failed.")
        return False
    
    print("\n✅ All tests passed! The supply chain optimization system is ready.")
    print("\nNext steps:")
    print("1. Run the demo: python scripts/demo.py")
    print("2. Start the Streamlit app: streamlit run demo/app.py")
    print("3. Run tests: pytest tests/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
