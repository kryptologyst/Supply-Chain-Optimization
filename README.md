# Supply Chain Optimization

**EXPERIMENTAL RESEARCH PROJECT - NOT FOR AUTOMATED DECISIONS WITHOUT HUMAN REVIEW**

This project demonstrates supply chain optimization techniques for educational and research purposes. All models and recommendations should be reviewed by qualified professionals before implementation in production systems.

## Overview

This project implements comprehensive supply chain optimization solutions including:

- **Transportation Optimization**: Classic transportation problem solving with linear programming
- **Inventory Management**: Multi-echelon inventory optimization with service level constraints
- **Resource Allocation**: Staff scheduling and workforce planning optimization
- **Demand Forecasting**: Time series forecasting for supply chain planning
- **Process Optimization**: Supply chain process mining and bottleneck analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Supply-Chain-Optimization.git
cd Supply-Chain-Optimization

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.transportation import TransportationOptimizer
from src.data.synthetic import generate_transportation_data

# Generate synthetic data
data = generate_transportation_data()

# Create optimizer
optimizer = TransportationOptimizer()

# Solve transportation problem
result = optimizer.optimize(
    supply=data['supply'],
    demand=data['demand'],
    costs=data['costs']
)

print(f"Optimal cost: ${result.total_cost:.2f}")
print("Shipping plan:")
print(result.shipping_plan)
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
supply-chain-optimization/
├── src/                    # Source code
│   ├── data/              # Data processing and synthetic generation
│   ├── features/           # Feature engineering
│   ├── models/            # Optimization models
│   ├── forecasting/       # Demand forecasting
│   ├── optimization/      # Core optimization algorithms
│   ├── evaluation/        # Metrics and evaluation
│   ├── visualization/     # Plotting and visualization
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data storage
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated artifacts and reports
├── demo/                  # Streamlit demo application
└── docs/                  # Documentation
```

## Dataset Schema

### Transportation Data
- `warehouses.csv`: Warehouse locations, capacities, and costs
- `stores.csv`: Store locations and demand requirements
- `transportation_costs.csv`: Shipping costs between warehouses and stores

### Inventory Data
- `products.csv`: Product catalog with attributes and costs
- `inventory_levels.csv`: Current inventory levels by location
- `demand_history.csv`: Historical demand patterns
- `supplier_leadtimes.csv`: Supplier lead times and reliability

### Workforce Data
- `employees.csv`: Employee skills, availability, and costs
- `shifts.csv`: Shift requirements and constraints
- `tasks.csv`: Task definitions and skill requirements

## Models and Algorithms

### Transportation Optimization
- **Linear Programming**: Classic transportation problem
- **Multi-commodity Flow**: Complex network optimization
- **Vehicle Routing**: VRP with time windows and capacity constraints

### Inventory Management
- **Newsvendor Model**: Single-period inventory optimization
- **Multi-echelon**: Complex supply chain inventory optimization
- **Service Level Optimization**: Balancing cost vs service levels

### Resource Allocation
- **Assignment Problem**: Hungarian algorithm for optimal assignments
- **Shift Scheduling**: Mixed-integer programming for workforce planning
- **Skill Matching**: Optimization with skill constraints

### Forecasting
- **Time Series**: ARIMA, ETS, and seasonal decomposition
- **Machine Learning**: XGBoost and LightGBM for demand forecasting
- **Hierarchical**: Multi-level demand reconciliation

## Evaluation Metrics

### Business KPIs
- **Cost Optimization**: Total cost reduction, cost per unit
- **Service Levels**: Fill rate, on-time delivery percentage
- **Efficiency**: Inventory turnover, utilization rates
- **Flexibility**: Response time to demand changes

### Technical Metrics
- **Optimization**: Objective value, constraint violations, solve time
- **Forecasting**: MAPE, SMAPE, MASE, bias
- **Allocation**: Assignment efficiency, skill utilization

## Configuration

The project uses YAML configuration files for easy parameter tuning:

```yaml
# configs/transportation.yaml
optimization:
  method: "highs"  # or "pulp", "ortools"
  timeout: 300
  tolerance: 1e-6

constraints:
  max_capacity_utilization: 0.95
  min_service_level: 0.90

costs:
  holding_cost_rate: 0.20
  stockout_cost_multiplier: 10.0
```

## Privacy and Compliance

- **Data Anonymization**: All customer and employee data is anonymized
- **PII Minimization**: Only necessary business metrics are retained
- **Audit Trail**: Complete decision logging for compliance
- **Fairness**: Bias detection and mitigation in workforce optimization

## Limitations and Disclaimers

### Important Disclaimers

1. **EXPERIMENTAL USE ONLY**: This software is for research and educational purposes
2. **NO AUTOMATED DECISIONS**: All recommendations require human review and approval
3. **AS-IS BASIS**: Software provided without warranties or guarantees
4. **PROFESSIONAL REVIEW**: Consult qualified supply chain professionals before implementation

### Known Limitations

- Models assume static demand patterns (seasonality handled separately)
- Network complexity limited to reasonable computational bounds
- Real-world constraints may not be fully captured
- Historical data quality affects forecast accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run linting and tests: `black . && ruff check . && pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in research, please cite:

```bibtex
@software{supply_chain_optimization,
  title={Supply Chain Optimization: Transportation, Inventory, and Multi-echelon Optimization},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Supply-Chain-Optimization}
}
```
# Supply-Chain-Optimization
