"""Feature engineering utilities for supply chain optimization."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class FeatureEngineer:
    """Feature engineering for supply chain data."""
    
    def __init__(self, config: DictConfig):
        """Initialize feature engineer with configuration.
        
        Args:
            config: Configuration object with feature engineering parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Create time-based features from date column.
        
        Args:
            df: DataFrame with date column.
            date_col: Name of the date column.
            
        Returns:
            DataFrame with additional time features.
        """
        df_features = df.copy()
        
        if date_col in df_features.columns:
            df_features[date_col] = pd.to_datetime(df_features[date_col])
            
            # Basic time features
            df_features["year"] = df_features[date_col].dt.year
            df_features["month"] = df_features[date_col].dt.month
            df_features["day"] = df_features[date_col].dt.day
            df_features["dayofweek"] = df_features[date_col].dt.dayofweek
            df_features["dayofyear"] = df_features[date_col].dt.dayofyear
            df_features["week"] = df_features[date_col].dt.isocalendar().week
            df_features["quarter"] = df_features[date_col].dt.quarter
            
            # Cyclical features
            df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
            df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
            df_features["dayofweek_sin"] = np.sin(2 * np.pi * df_features["dayofweek"] / 7)
            df_features["dayofweek_cos"] = np.cos(2 * np.pi * df_features["dayofweek"] / 7)
            
            # Holiday indicators (simplified)
            df_features["is_weekend"] = df_features["dayofweek"].isin([5, 6]).astype(int)
            df_features["is_month_start"] = df_features["day"].le(7).astype(int)
            df_features["is_month_end"] = df_features["day"].ge(25).astype(int)
        
        return df_features
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_cols: Optional[List[str]] = None,
        lags: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """Create lag features for time series data.
        
        Args:
            df: DataFrame with time series data.
            value_col: Name of the value column to create lags for.
            group_cols: Columns to group by when creating lags.
            lags: List of lag periods.
            
        Returns:
            DataFrame with additional lag features.
        """
        df_features = df.copy()
        
        if group_cols:
            for lag in lags:
                df_features[f"{value_col}_lag_{lag}"] = (
                    df_features.groupby(group_cols)[value_col].shift(lag)
                )
        else:
            for lag in lags:
                df_features[f"{value_col}_lag_{lag}"] = df_features[value_col].shift(lag)
        
        return df_features
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_cols: Optional[List[str]] = None,
        windows: List[int] = [7, 30, 90]
    ) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            df: DataFrame with time series data.
            value_col: Name of the value column.
            group_cols: Columns to group by.
            windows: List of window sizes.
            
        Returns:
            DataFrame with additional rolling features.
        """
        df_features = df.copy()
        
        if group_cols:
            grouped = df_features.groupby(group_cols)[value_col]
        else:
            grouped = df_features[value_col]
        
        for window in windows:
            df_features[f"{value_col}_rolling_mean_{window}"] = grouped.rolling(window).mean()
            df_features[f"{value_col}_rolling_std_{window}"] = grouped.rolling(window).std()
            df_features[f"{value_col}_rolling_min_{window}"] = grouped.rolling(window).min()
            df_features[f"{value_col}_rolling_max_{window}"] = grouped.rolling(window).max()
        
        return df_features
    
    def create_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demand-specific features.
        
        Args:
            df: DataFrame with demand data.
            
        Returns:
            DataFrame with additional demand features.
        """
        df_features = df.copy()
        
        if "demand" in df_features.columns:
            # Demand volatility
            df_features["demand_volatility"] = df_features["demand"].rolling(30).std()
            
            # Demand trend
            df_features["demand_trend"] = df_features["demand"].rolling(30).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Demand seasonality (simplified)
            if "month" in df_features.columns:
                monthly_avg = df_features.groupby("month")["demand"].transform("mean")
                df_features["demand_seasonality"] = df_features["demand"] / monthly_avg
        
        return df_features
    
    def create_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cost-specific features.
        
        Args:
            df: DataFrame with cost data.
            
        Returns:
            DataFrame with additional cost features.
        """
        df_features = df.copy()
        
        if "cost" in df_features.columns:
            # Cost efficiency metrics
            df_features["cost_per_unit"] = df_features["cost"] / df_features.get("quantity", 1)
            
            # Cost volatility
            df_features["cost_volatility"] = df_features["cost"].rolling(30).std()
            
            # Cost trend
            df_features["cost_trend"] = df_features["cost"].rolling(30).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        return df_features
    
    def create_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create inventory-specific features.
        
        Args:
            df: DataFrame with inventory data.
            
        Returns:
            DataFrame with additional inventory features.
        """
        df_features = df.copy()
        
        if "current_stock" in df_features.columns:
            # Inventory turnover
            if "demand" in df_features.columns:
                df_features["inventory_turnover"] = df_features["demand"] / df_features["current_stock"]
            
            # Stock-out risk
            if "reorder_point" in df_features.columns:
                df_features["stockout_risk"] = (
                    df_features["current_stock"] < df_features["reorder_point"]
                ).astype(int)
            
            # Inventory efficiency
            if "unit_cost" in df_features.columns:
                df_features["inventory_value"] = df_features["current_stock"] * df_features["unit_cost"]
        
        return df_features
    
    def create_distance_features(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude"
    ) -> pd.DataFrame:
        """Create distance-based features.
        
        Args:
            df: DataFrame with location data.
            lat_col: Name of latitude column.
            lon_col: Name of longitude column.
            
        Returns:
            DataFrame with additional distance features.
        """
        df_features = df.copy()
        
        if lat_col in df_features.columns and lon_col in df_features.columns:
            # Distance from center (simplified)
            center_lat = df_features[lat_col].mean()
            center_lon = df_features[lon_col].mean()
            
            df_features["distance_from_center"] = np.sqrt(
                (df_features[lat_col] - center_lat) ** 2 + 
                (df_features[lon_col] - center_lon) ** 2
            )
            
            # Regional features
            df_features["is_north"] = (df_features[lat_col] > center_lat).astype(int)
            df_features["is_east"] = (df_features[lon_col] > center_lon).astype(int)
        
        return df_features
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Create interaction features between pairs of features.
        
        Args:
            df: DataFrame with features.
            feature_pairs: List of (feature1, feature2) tuples.
            
        Returns:
            DataFrame with additional interaction features.
        """
        df_features = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_features.columns and feat2 in df_features.columns:
                # Multiplication interaction
                df_features[f"{feat1}_x_{feat2}"] = df_features[feat1] * df_features[feat2]
                
                # Ratio interaction (avoid division by zero)
                df_features[f"{feat1}_div_{feat2}"] = np.where(
                    df_features[feat2] != 0,
                    df_features[feat1] / df_features[feat2],
                    0
                )
        
        return df_features
    
    def engineer_all_features(
        self,
        df: pd.DataFrame,
        feature_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame.
            feature_config: Optional configuration for feature engineering.
            
        Returns:
            DataFrame with all engineered features.
        """
        if feature_config is None:
            feature_config = {
                "time_features": True,
                "lag_features": {"lags": [1, 7, 30]},
                "rolling_features": {"windows": [7, 30, 90]},
                "demand_features": True,
                "cost_features": True,
                "inventory_features": True,
                "distance_features": True
            }
        
        df_features = df.copy()
        
        # Apply feature engineering steps
        if feature_config.get("time_features", False):
            df_features = self.create_time_features(df_features)
        
        if feature_config.get("lag_features", False):
            lag_config = feature_config["lag_features"]
            df_features = self.create_lag_features(
                df_features, 
                "demand", 
                lags=lag_config.get("lags", [1, 7, 30])
            )
        
        if feature_config.get("rolling_features", False):
            rolling_config = feature_config["rolling_features"]
            df_features = self.create_rolling_features(
                df_features,
                "demand",
                windows=rolling_config.get("windows", [7, 30, 90])
            )
        
        if feature_config.get("demand_features", False):
            df_features = self.create_demand_features(df_features)
        
        if feature_config.get("cost_features", False):
            df_features = self.create_cost_features(df_features)
        
        if feature_config.get("inventory_features", False):
            df_features = self.create_inventory_features(df_features)
        
        if feature_config.get("distance_features", False):
            df_features = self.create_distance_features(df_features)
        
        # Remove any infinite or NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)
        
        self.logger.info(f"Feature engineering completed. Shape: {df.shape} -> {df_features.shape}")
        
        return df_features
