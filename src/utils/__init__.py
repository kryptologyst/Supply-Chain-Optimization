"""Utility functions for supply chain optimization."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Loaded configuration object.
    """
    return OmegaConf.load(config_path)


def setup_logging(config: DictConfig) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        config: Configuration object containing logging settings.
        
    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_dataframe_schema(
    df: pd.DataFrame, 
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> bool:
    """Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        optional_columns: List of optional column names.
        
    Returns:
        True if validation passes, False otherwise.
    """
    missing_required = set(required_columns) - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    if optional_columns:
        extra_columns = set(df.columns) - set(required_columns) - set(optional_columns)
        if extra_columns:
            logging.warning(f"Unexpected columns found: {extra_columns}")
    
    return True


def calculate_distance_matrix(
    locations: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> np.ndarray:
    """Calculate distance matrix between locations using Haversine formula.
    
    Args:
        locations: DataFrame with location coordinates.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        
    Returns:
        Distance matrix in kilometers.
    """
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points."""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    n_locations = len(locations)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = haversine_distance(
                    locations.iloc[i][lat_col],
                    locations.iloc[i][lon_col],
                    locations.iloc[j][lat_col],
                    locations.iloc[j][lon_col]
                )
    
    return distance_matrix


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency string.
    
    Args:
        amount: Amount to format.
        currency: Currency code.
        
    Returns:
        Formatted currency string.
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_service_level(
    demand: np.ndarray,
    supply: np.ndarray,
    stockouts: np.ndarray
) -> float:
    """Calculate service level as percentage of demand satisfied.
    
    Args:
        demand: Total demand array.
        supply: Total supply array.
        stockouts: Number of stockouts array.
        
    Returns:
        Service level as percentage (0-100).
    """
    total_demand = np.sum(demand)
    total_stockouts = np.sum(stockouts)
    
    if total_demand == 0:
        return 100.0
    
    service_level = ((total_demand - total_stockouts) / total_demand) * 100
    return max(0.0, min(100.0, service_level))


def calculate_inventory_turns(
    cost_of_goods_sold: float,
    average_inventory_value: float
) -> float:
    """Calculate inventory turnover ratio.
    
    Args:
        cost_of_goods_sold: Annual cost of goods sold.
        average_inventory_value: Average inventory value.
        
    Returns:
        Inventory turnover ratio.
    """
    if average_inventory_value == 0:
        return float('inf')
    
    return cost_of_goods_sold / average_inventory_value


def calculate_fill_rate(
    orders_fulfilled: int,
    total_orders: int
) -> float:
    """Calculate fill rate percentage.
    
    Args:
        orders_fulfilled: Number of orders completely fulfilled.
        total_orders: Total number of orders.
        
    Returns:
        Fill rate as percentage (0-100).
    """
    if total_orders == 0:
        return 100.0
    
    return (orders_fulfilled / total_orders) * 100


def anonymize_data(df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
    """Anonymize sensitive data by hashing ID columns.
    
    Args:
        df: DataFrame to anonymize.
        id_columns: List of column names to anonymize.
        
    Returns:
        DataFrame with anonymized ID columns.
    """
    df_anon = df.copy()
    
    for col in id_columns:
        if col in df_anon.columns:
            df_anon[col] = df_anon[col].astype(str).apply(
                lambda x: f"anon_{hash(x) % 1000000:06d}"
            )
    
    return df_anon


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for DataFrame.
    
    Args:
        df: DataFrame to summarize.
        
    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "categorical_summary": {}
    }
    
    # Add categorical column summaries
    for col in df.select_dtypes(include=['object', 'category']).columns:
        summary["categorical_summary"][col] = {
            "unique_values": df[col].nunique(),
            "most_common": df[col].value_counts().head().to_dict()
        }
    
    return summary
