"""Demand forecasting models and algorithms."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class DemandForecaster:
    """Demand forecasting using various methods."""
    
    def __init__(self, config: DictConfig):
        """Initialize forecaster with configuration.
        
        Args:
            config: Configuration object with forecasting parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def forecast_arima(
        self,
        data: pd.DataFrame,
        target_col: str = "demand",
        group_cols: Optional[List[str]] = None,
        horizon: int = 30
    ) -> pd.DataFrame:
        """Forecast using ARIMA model.
        
        Args:
            data: DataFrame with time series data.
            target_col: Name of target column.
            group_cols: Columns to group by for separate forecasts.
            horizon: Forecast horizon in periods.
            
        Returns:
            DataFrame with forecasts.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            self.logger.warning("statsmodels not available. Using simple forecasting.")
            return self._simple_forecast(data, target_col, group_cols, horizon)
        
        forecasts = []
        
        if group_cols:
            for group, group_data in data.groupby(group_cols):
                forecast = self._forecast_single_series(group_data, target_col, horizon, ARIMA)
                forecast[group_cols] = group if isinstance(group, tuple) else [group]
                forecasts.append(forecast)
        else:
            forecast = self._forecast_single_series(data, target_col, horizon, ARIMA)
            forecasts.append(forecast)
        
        return pd.concat(forecasts, ignore_index=True)
    
    def forecast_xgboost(
        self,
        data: pd.DataFrame,
        target_col: str = "demand",
        feature_cols: Optional[List[str]] = None,
        group_cols: Optional[List[str]] = None,
        horizon: int = 30
    ) -> pd.DataFrame:
        """Forecast using XGBoost model.
        
        Args:
            data: DataFrame with time series data.
            target_col: Name of target column.
            feature_cols: List of feature columns to use.
            group_cols: Columns to group by for separate models.
            horizon: Forecast horizon in periods.
            
        Returns:
            DataFrame with forecasts.
        """
        try:
            import xgboost as xgb
        except ImportError:
            self.logger.warning("XGBoost not available. Using simple forecasting.")
            return self._simple_forecast(data, target_col, group_cols, horizon)
        
        forecasts = []
        
        if group_cols:
            for group, group_data in data.groupby(group_cols):
                forecast = self._forecast_xgboost_single(
                    group_data, target_col, feature_cols, horizon, xgb
                )
                forecast[group_cols] = group if isinstance(group, tuple) else [group]
                forecasts.append(forecast)
        else:
            forecast = self._forecast_xgboost_single(data, target_col, feature_cols, horizon, xgb)
            forecasts.append(forecast)
        
        return pd.concat(forecasts, ignore_index=True)
    
    def _forecast_single_series(
        self,
        data: pd.DataFrame,
        target_col: str,
        horizon: int,
        model_class
    ) -> pd.DataFrame:
        """Forecast a single time series."""
        try:
            # Simple ARIMA(1,1,1) model
            model = model_class(data[target_col], order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            
            # Create forecast DataFrame
            last_date = data.index[-1] if hasattr(data.index, 'dtype') else len(data) - 1
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            return pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast,
                'method': 'arima'
            })
        
        except Exception as e:
            self.logger.warning(f"ARIMA forecasting failed: {e}. Using simple forecast.")
            return self._simple_forecast_single(data, target_col, horizon)
    
    def _forecast_xgboost_single(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]],
        horizon: int,
        xgb
    ) -> pd.DataFrame:
        """Forecast using XGBoost for a single series."""
        try:
            if feature_cols is None:
                feature_cols = [col for col in data.columns if col != target_col]
            
            # Prepare features
            X = data[feature_cols].fillna(0)
            y = data[target_col]
            
            # Train model
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Generate forecast (simplified - using last known features)
            last_features = X.iloc[-1:].values
            forecast = []
            
            for _ in range(horizon):
                pred = model.predict(last_features)[0]
                forecast.append(pred)
                # Update features for next prediction (simplified)
                last_features[0, 0] = pred  # Assuming first feature is lagged target
            
            # Create forecast DataFrame
            last_date = data.index[-1] if hasattr(data.index, 'dtype') else len(data) - 1
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            return pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast,
                'method': 'xgboost'
            })
        
        except Exception as e:
            self.logger.warning(f"XGBoost forecasting failed: {e}. Using simple forecast.")
            return self._simple_forecast_single(data, target_col, horizon)
    
    def _simple_forecast(
        self,
        data: pd.DataFrame,
        target_col: str,
        group_cols: Optional[List[str]],
        horizon: int
    ) -> pd.DataFrame:
        """Simple forecasting using moving average."""
        forecasts = []
        
        if group_cols:
            for group, group_data in data.groupby(group_cols):
                forecast = self._simple_forecast_single(group_data, target_col, horizon)
                forecast[group_cols] = group if isinstance(group, tuple) else [group]
                forecasts.append(forecast)
        else:
            forecast = self._simple_forecast_single(data, target_col, horizon)
            forecasts.append(forecast)
        
        return pd.concat(forecasts, ignore_index=True)
    
    def _simple_forecast_single(
        self,
        data: pd.DataFrame,
        target_col: str,
        horizon: int
    ) -> pd.DataFrame:
        """Simple forecasting for a single series."""
        # Use 30-day moving average as forecast
        avg_demand = data[target_col].tail(30).mean()
        
        # Create forecast DataFrame
        last_date = data.index[-1] if hasattr(data.index, 'dtype') else len(data) - 1
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': [avg_demand] * horizon,
            'method': 'simple_ma'
        })
    
    def evaluate_forecast(
        self,
        actual: pd.Series,
        forecast: pd.Series
    ) -> Dict[str, float]:
        """Evaluate forecast accuracy.
        
        Args:
            actual: Actual values.
            forecast: Forecasted values.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]
        
        if len(actual_clean) == 0:
            return {"mape": float('inf'), "smape": float('inf'), "mase": float('inf'), "bias": 0}
        
        # Calculate metrics
        mape = np.mean(np.abs((actual_clean - forecast_clean) / actual_clean)) * 100
        smape = np.mean(2 * np.abs(actual_clean - forecast_clean) / (np.abs(actual_clean) + np.abs(forecast_clean))) * 100
        
        # MASE (simplified - using naive forecast as baseline)
        naive_forecast = actual_clean.shift(1).dropna()
        naive_mae = np.mean(np.abs(actual_clean[1:] - naive_forecast))
        forecast_mae = np.mean(np.abs(actual_clean - forecast_clean))
        mase = forecast_mae / naive_mae if naive_mae > 0 else float('inf')
        
        bias = np.mean(forecast_clean - actual_clean)
        
        return {
            "mape": mape,
            "smape": smape,
            "mase": mase,
            "bias": bias
        }
    
    def hierarchical_forecast(
        self,
        data: pd.DataFrame,
        hierarchy_cols: List[str],
        target_col: str = "demand",
        method: str = "bottom_up"
    ) -> pd.DataFrame:
        """Perform hierarchical forecasting.
        
        Args:
            data: DataFrame with hierarchical time series data.
            hierarchy_cols: Columns defining the hierarchy.
            target_col: Name of target column.
            method: Reconciliation method ('bottom_up', 'top_down').
            
        Returns:
            DataFrame with reconciled forecasts.
        """
        # This is a simplified implementation
        # In practice, you'd use specialized hierarchical forecasting libraries
        
        if method == "bottom_up":
            # Forecast at the lowest level and aggregate up
            lowest_level = hierarchy_cols[-1]
            forecasts = []
            
            for group, group_data in data.groupby(hierarchy_cols):
                forecast = self._simple_forecast_single(group_data, target_col, 30)
                for i, col in enumerate(hierarchy_cols):
                    forecast[col] = group[i] if isinstance(group, tuple) else group
                forecasts.append(forecast)
            
            return pd.concat(forecasts, ignore_index=True)
        
        else:  # top_down
            # Forecast at the top level and disaggregate down
            top_level = hierarchy_cols[0]
            top_forecast = self._simple_forecast_single(data, target_col, 30)
            top_forecast[top_level] = data[top_level].iloc[0]
            
            return top_forecast
