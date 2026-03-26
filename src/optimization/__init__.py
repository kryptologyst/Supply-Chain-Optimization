"""Core optimization algorithms and solvers."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class InventoryOptimizer:
    """Inventory optimization using various methods."""
    
    def __init__(self, config: DictConfig):
        """Initialize optimizer with configuration.
        
        Args:
            config: Configuration object with optimization parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_newsvendor(
        self,
        demand_data: pd.DataFrame,
        cost_data: pd.DataFrame,
        service_level: float = 0.95
    ) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
        """Optimize inventory using Newsvendor model.
        
        Args:
            demand_data: DataFrame with demand data.
            cost_data: DataFrame with cost data.
            service_level: Target service level.
            
        Returns:
            Dictionary with optimization results.
        """
        results = {}
        
        # Calculate optimal order quantities
        optimal_quantities = []
        
        for _, row in demand_data.iterrows():
            product_id = row.get("product_id", "unknown")
            demand_mean = row.get("demand_mean", 0)
            demand_std = row.get("demand_std", 0)
            
            # Newsvendor formula: Q = μ + z * σ
            z_score = self._get_z_score(service_level)
            optimal_qty = demand_mean + z_score * demand_std
            
            optimal_quantities.append({
                "product_id": product_id,
                "optimal_quantity": max(0, optimal_qty),
                "service_level": service_level
            })
        
        results["optimal_quantities"] = pd.DataFrame(optimal_quantities)
        
        # Calculate total cost
        total_cost = 0
        for _, row in results["optimal_quantities"].iterrows():
            product_id = row["product_id"]
            qty = row["optimal_quantity"]
            
            # Find cost for this product
            product_cost = cost_data[cost_data["product_id"] == product_id]["unit_cost"].iloc[0]
            total_cost += qty * product_cost
        
        results["total_cost"] = total_cost
        results["service_level"] = service_level
        
        return results
    
    def optimize_multi_echelon(
        self,
        network_data: pd.DataFrame,
        demand_data: pd.DataFrame,
        cost_data: pd.DataFrame
    ) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
        """Optimize multi-echelon inventory system.
        
        Args:
            network_data: DataFrame with network structure.
            demand_data: DataFrame with demand data.
            cost_data: DataFrame with cost data.
            
        Returns:
            Dictionary with optimization results.
        """
        # Simplified multi-echelon optimization
        # In practice, this would use more sophisticated algorithms
        
        results = {}
        
        # Calculate optimal inventory levels for each echelon
        inventory_levels = []
        
        for _, row in network_data.iterrows():
            location_id = row["location_id"]
            echelon_level = row["echelon_level"]
            
            # Find demand for this location
            location_demand = demand_data[demand_data["location_id"] == location_id]
            
            if not location_demand.empty:
                avg_demand = location_demand["demand"].mean()
                lead_time = row.get("lead_time", 7)
                
                # Calculate safety stock based on echelon level
                safety_stock_multiplier = 1.0 + (echelon_level * 0.1)
                safety_stock = avg_demand * lead_time * safety_stock_multiplier
                
                inventory_levels.append({
                    "location_id": location_id,
                    "echelon_level": echelon_level,
                    "optimal_inventory": safety_stock,
                    "safety_stock": safety_stock * 0.3
                })
        
        results["inventory_levels"] = pd.DataFrame(inventory_levels)
        
        # Calculate total cost
        total_cost = 0
        for _, row in results["inventory_levels"].iterrows():
            location_id = row["location_id"]
            inventory = row["optimal_inventory"]
            
            # Find cost for this location
            location_cost = cost_data[cost_data["location_id"] == location_id]["holding_cost_rate"].iloc[0]
            total_cost += inventory * location_cost
        
        results["total_cost"] = total_cost
        
        return results
    
    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for given service level."""
        # Simplified z-score calculation
        # In practice, you'd use scipy.stats.norm.ppf
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33
        }
        return z_scores.get(service_level, 1.65)


class WorkforceOptimizer:
    """Workforce optimization using assignment and scheduling algorithms."""
    
    def __init__(self, config: DictConfig):
        """Initialize optimizer with configuration.
        
        Args:
            config: Configuration object with optimization parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_assignment(
        self,
        employees: pd.DataFrame,
        tasks: pd.DataFrame,
        cost_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
        """Optimize employee-task assignments using Hungarian algorithm.
        
        Args:
            employees: DataFrame with employee data.
            tasks: DataFrame with task data.
            cost_matrix: Optional cost matrix for assignments.
            
        Returns:
            Dictionary with optimization results.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            self.logger.warning("SciPy not available. Using greedy assignment.")
            return self._greedy_assignment(employees, tasks)
        
        n_employees = len(employees)
        n_tasks = len(tasks)
        
        # Create cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = self._create_cost_matrix(employees, tasks)
        
        # Ensure square matrix for Hungarian algorithm
        if n_employees != n_tasks:
            cost_matrix = self._pad_cost_matrix(cost_matrix, max(n_employees, n_tasks))
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignment results
        assignments = []
        total_cost = 0
        
        for i, j in zip(row_indices, col_indices):
            if i < n_employees and j < n_tasks:
                employee_id = employees.iloc[i]["employee_id"]
                task_id = tasks.iloc[j]["task_id"]
                cost = cost_matrix[i, j]
                
                assignments.append({
                    "employee_id": employee_id,
                    "task_id": task_id,
                    "cost": cost,
                    "assignment_score": 1.0 - (cost / np.max(cost_matrix))
                })
                
                total_cost += cost
        
        results = {
            "assignments": pd.DataFrame(assignments),
            "total_cost": total_cost,
            "utilization_rate": len(assignments) / max(n_employees, n_tasks),
            "method": "hungarian"
        }
        
        return results
    
    def optimize_shift_scheduling(
        self,
        employees: pd.DataFrame,
        shifts: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
        """Optimize shift scheduling using constraint programming.
        
        Args:
            employees: DataFrame with employee data.
            shifts: DataFrame with shift data.
            constraints: Optional scheduling constraints.
            
        Returns:
            Dictionary with optimization results.
        """
        # Simplified shift scheduling
        # In practice, this would use more sophisticated constraint programming
        
        results = {}
        assignments = []
        
        for _, shift in shifts.iterrows():
            shift_id = shift["shift_id"]
            required_employees = shift["required_employees"]
            required_skills = shift.get("required_skills", "").split(",")
            
            # Find eligible employees
            eligible_employees = employees[
                employees["availability"].str.contains(shift["day"], na=False)
            ]
            
            # Filter by skills if specified
            if required_skills and required_skills[0]:
                skill_mask = eligible_employees["skills"].str.contains(
                    "|".join(required_skills), na=False
                )
                eligible_employees = eligible_employees[skill_mask]
            
            # Select employees (simplified - just take first N)
            selected_employees = eligible_employees.head(required_employees)
            
            for _, employee in selected_employees.iterrows():
                assignments.append({
                    "employee_id": employee["employee_id"],
                    "shift_id": shift_id,
                    "day": shift["day"],
                    "hours": shift["duration_hours"],
                    "cost": employee["hourly_rate"] * shift["duration_hours"]
                })
        
        results["assignments"] = pd.DataFrame(assignments)
        results["total_cost"] = sum(assignment["cost"] for assignment in assignments)
        results["utilization_rate"] = len(assignments) / len(employees)
        
        return results
    
    def _create_cost_matrix(self, employees: pd.DataFrame, tasks: pd.DataFrame) -> np.ndarray:
        """Create cost matrix for employee-task assignments."""
        n_employees = len(employees)
        n_tasks = len(tasks)
        
        cost_matrix = np.zeros((n_employees, n_tasks))
        
        for i, employee in employees.iterrows():
            for j, task in tasks.iterrows():
                # Calculate assignment cost based on skills match and hourly rate
                employee_skills = set(employee["skills"].split(","))
                required_skills = set(task.get("required_skills", "").split(","))
                
                skill_match = len(employee_skills.intersection(required_skills)) / max(len(required_skills), 1)
                base_cost = employee["hourly_rate"] * task.get("estimated_hours", 1)
                
                # Lower cost for better skill matches
                cost_matrix[i, j] = base_cost * (1 - skill_match * 0.3)
        
        return cost_matrix
    
    def _pad_cost_matrix(self, cost_matrix: np.ndarray, size: int) -> np.ndarray:
        """Pad cost matrix to make it square."""
        padded = np.full((size, size), np.max(cost_matrix) * 2)
        padded[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix
        return padded
    
    def _greedy_assignment(self, employees: pd.DataFrame, tasks: pd.DataFrame) -> Dict:
        """Greedy assignment algorithm as fallback."""
        assignments = []
        assigned_tasks = set()
        
        for _, employee in employees.iterrows():
            employee_skills = set(employee["skills"].split(","))
            best_task = None
            best_score = -1
            
            for _, task in tasks.iterrows():
                if task["task_id"] in assigned_tasks:
                    continue
                
                required_skills = set(task.get("required_skills", "").split(","))
                skill_match = len(employee_skills.intersection(required_skills)) / max(len(required_skills), 1)
                
                if skill_match > best_score:
                    best_score = skill_match
                    best_task = task
            
            if best_task is not None:
                assignments.append({
                    "employee_id": employee["employee_id"],
                    "task_id": best_task["task_id"],
                    "cost": employee["hourly_rate"] * best_task.get("estimated_hours", 1),
                    "assignment_score": best_score
                })
                assigned_tasks.add(best_task["task_id"])
        
        return {
            "assignments": pd.DataFrame(assignments),
            "total_cost": sum(assignment["cost"] for assignment in assignments),
            "utilization_rate": len(assignments) / len(employees),
            "method": "greedy"
        }
