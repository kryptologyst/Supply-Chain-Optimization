"""Data processing and synthetic data generation for supply chain optimization."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..utils import set_seed


class SyntheticDataGenerator:
    """Generate synthetic supply chain data for testing and demonstration."""
    
    def __init__(self, config: DictConfig):
        """Initialize data generator with configuration.
        
        Args:
            config: Configuration object with data generation parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        set_seed(42)  # Ensure reproducible data
    
    def generate_transportation_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic transportation problem data.
        
        Returns:
            Dictionary containing supply, demand, and cost matrices.
        """
        n_warehouses = self.config.data.synthetic.n_warehouses
        n_stores = self.config.data.synthetic.n_stores
        
        # Generate supply capacities (warehouses)
        supply = np.random.uniform(50, 200, n_warehouses).astype(int)
        
        # Generate demand requirements (stores)
        demand = np.random.uniform(30, 150, n_stores).astype(int)
        
        # Ensure total supply >= total demand
        total_demand = np.sum(demand)
        total_supply = np.sum(supply)
        
        if total_supply < total_demand:
            # Scale up supply to meet demand
            scale_factor = total_demand / total_supply
            supply = (supply * scale_factor).astype(int)
        
        # Generate cost matrix (transportation costs per unit)
        base_cost = self.config.costs.transportation_cost_base
        cost_variability = self.config.data.synthetic.cost_variability
        
        costs = np.random.uniform(
            base_cost * (1 - cost_variability),
            base_cost * (1 + cost_variability),
            (n_warehouses, n_stores)
        )
        
        self.logger.info(f"Generated transportation data: {n_warehouses} warehouses, {n_stores} stores")
        
        return {
            "supply": supply,
            "demand": demand,
            "costs": costs,
            "n_warehouses": n_warehouses,
            "n_stores": n_stores
        }
    
    def generate_inventory_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic inventory management data.
        
        Returns:
            Dictionary containing inventory-related DataFrames.
        """
        n_products = self.config.data.synthetic.n_products
        n_locations = self.config.data.synthetic.n_locations
        time_horizon = self.config.data.synthetic.time_horizon_days
        
        # Generate product catalog
        products = pd.DataFrame({
            "product_id": [f"P{i:03d}" for i in range(n_products)],
            "name": [f"Product {i+1}" for i in range(n_products)],
            "category": np.random.choice(["Electronics", "Clothing", "Home", "Sports"], n_products),
            "unit_cost": np.random.uniform(10, 500, n_products),
            "weight": np.random.uniform(0.1, 10, n_products),
            "volume": np.random.uniform(0.01, 1, n_products)
        })
        
        # Generate location data
        locations = pd.DataFrame({
            "location_id": [f"L{i:03d}" for i in range(n_locations)],
            "name": [f"Location {i+1}" for i in range(n_locations)],
            "type": np.random.choice(["Warehouse", "Store", "DC"], n_locations),
            "latitude": np.random.uniform(25, 50, n_locations),
            "longitude": np.random.uniform(-125, -65, n_locations),
            "capacity": np.random.uniform(1000, 10000, n_locations)
        })
        
        # Generate demand history
        demand_history = []
        for product_id in products["product_id"]:
            for location_id in locations["location_id"]:
                # Generate time series with seasonality and trend
                base_demand = np.random.uniform(1, 20)
                trend = np.random.uniform(-0.1, 0.1)
                seasonality = np.random.uniform(0.1, 0.5)
                
                dates = pd.date_range(
                    start="2023-01-01",
                    periods=time_horizon,
                    freq="D"
                )
                
                # Create seasonal pattern
                seasonal_pattern = seasonality * np.sin(2 * np.pi * np.arange(time_horizon) / 365)
                
                # Create trend
                trend_pattern = trend * np.arange(time_horizon)
                
                # Add noise
                noise = np.random.normal(0, 0.1, time_horizon)
                
                demand_values = base_demand * (1 + seasonal_pattern + trend_pattern + noise)
                demand_values = np.maximum(demand_values, 0)  # Ensure non-negative
                
                for i, date in enumerate(dates):
                    demand_history.append({
                        "date": date,
                        "product_id": product_id,
                        "location_id": location_id,
                        "demand": max(0, int(demand_values[i]))
                    })
        
        demand_df = pd.DataFrame(demand_history)
        
        # Generate current inventory levels
        inventory_levels = []
        for product_id in products["product_id"]:
            for location_id in locations["location_id"]:
                avg_demand = demand_df[
                    (demand_df["product_id"] == product_id) & 
                    (demand_df["location_id"] == location_id)
                ]["demand"].mean()
                
                # Set initial inventory to cover 30-60 days of demand
                initial_inventory = int(avg_demand * np.random.uniform(30, 60))
                
                inventory_levels.append({
                    "product_id": product_id,
                    "location_id": location_id,
                    "current_stock": initial_inventory,
                    "reorder_point": int(avg_demand * 14),  # 14 days of demand
                    "order_quantity": int(avg_demand * 30)  # 30 days of demand
                })
        
        inventory_df = pd.DataFrame(inventory_levels)
        
        self.logger.info(f"Generated inventory data: {n_products} products, {n_locations} locations")
        
        return {
            "products": products,
            "locations": locations,
            "demand_history": demand_df,
            "inventory_levels": inventory_df
        }
    
    def generate_workforce_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic workforce planning data.
        
        Returns:
            Dictionary containing workforce-related DataFrames.
        """
        n_employees = 50
        n_shifts = 7  # Days of the week
        n_skills = 10
        
        # Generate employee data
        skills = [f"Skill_{i+1}" for i in range(n_skills)]
        
        employees = []
        for i in range(n_employees):
            employee_skills = np.random.choice(skills, size=np.random.randint(2, 6), replace=False)
            employees.append({
                "employee_id": f"E{i:03d}",
                "name": f"Employee {i+1}",
                "skills": ",".join(employee_skills),
                "hourly_rate": np.random.uniform(15, 50),
                "max_hours_per_week": np.random.choice([20, 30, 40]),
                "availability": ",".join(np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], 
                                                         size=np.random.randint(3, 8), replace=False))
            })
        
        employees_df = pd.DataFrame(employees)
        
        # Generate shift requirements
        shifts = []
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            for shift_type in ["Morning", "Afternoon", "Evening"]:
                shifts.append({
                    "day": day,
                    "shift_type": shift_type,
                    "required_employees": np.random.randint(2, 8),
                    "required_skills": ",".join(np.random.choice(skills, size=np.random.randint(1, 4), replace=False)),
                    "duration_hours": np.random.choice([4, 6, 8])
                })
        
        shifts_df = pd.DataFrame(shifts)
        
        # Generate task requirements
        tasks = []
        for i in range(20):
            tasks.append({
                "task_id": f"T{i:03d}",
                "name": f"Task {i+1}",
                "required_skills": ",".join(np.random.choice(skills, size=np.random.randint(1, 3), replace=False)),
                "estimated_hours": np.random.uniform(1, 8),
                "priority": np.random.choice(["Low", "Medium", "High"]),
                "deadline_days": np.random.randint(1, 30)
            })
        
        tasks_df = pd.DataFrame(tasks)
        
        self.logger.info(f"Generated workforce data: {n_employees} employees, {len(shifts)} shifts")
        
        return {
            "employees": employees_df,
            "shifts": shifts_df,
            "tasks": tasks_df
        }


def load_transportation_data(file_path: str) -> Dict[str, np.ndarray]:
    """Load transportation data from CSV files.
    
    Args:
        file_path: Path to directory containing transportation data files.
        
    Returns:
        Dictionary containing loaded data arrays.
    """
    warehouses_df = pd.read_csv(f"{file_path}/warehouses.csv")
    stores_df = pd.read_csv(f"{file_path}/stores.csv")
    costs_df = pd.read_csv(f"{file_path}/transportation_costs.csv")
    
    # Extract supply and demand
    supply = warehouses_df["capacity"].values
    demand = stores_df["demand"].values
    
    # Create cost matrix
    n_warehouses = len(warehouses_df)
    n_stores = len(stores_df)
    costs = np.zeros((n_warehouses, n_stores))
    
    for _, row in costs_df.iterrows():
        warehouse_idx = warehouses_df[warehouses_df["warehouse_id"] == row["warehouse_id"]].index[0]
        store_idx = stores_df[stores_df["store_id"] == row["store_id"]].index[0]
        costs[warehouse_idx, store_idx] = row["cost_per_unit"]
    
    return {
        "supply": supply,
        "demand": demand,
        "costs": costs,
        "warehouses": warehouses_df,
        "stores": stores_df
    }


def save_results(results: Dict, output_path: str) -> None:
    """Save optimization results to files.
    
    Args:
        results: Dictionary containing results to save.
        output_path: Directory path to save results.
    """
    import os
    os.makedirs(output_path, exist_ok=True)
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(f"{output_path}/{key}.csv", index=False)
        elif isinstance(value, np.ndarray):
            np.save(f"{output_path}/{key}.npy", value)
        else:
            # Save as JSON for other data types
            import json
            with open(f"{output_path}/{key}.json", "w") as f:
                json.dump(value, f, indent=2)
