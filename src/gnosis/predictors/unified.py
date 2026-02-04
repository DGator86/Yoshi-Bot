"""Unified Prediction Pipeline - connects all components.

This module integrates:
1. All steering field modules (funding, liquidations, gamma, macro, temporal, orderbook)
2. Physics features from PriceParticle
3. Quantum features from QuantumPriceEngine
4. Regime-aware model selection
5. Ensemble prediction with multiple backends

The goal is to maximize prediction accuracy by:
- Using ALL available features
- Selecting models appropriate for current regime
- Ensembling multiple approaches
- Properly calibrated confidence intervals
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


@dataclass
class UnifiedConfig:
    """Configuration for unified predictor."""

    # Model selection
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "gradient_boost": 0.4,
        "random_forest": 0.3,
        "ridge": 0.2,
        "physics": 0.1,
    })

    # Regime-specific adjustments
    regime_aware: bool = True
    regime_model_overrides: Dict[str, str] = field(default_factory=lambda: {
        "trending": "gradient_boost",
        "ranging": "ridge",
        "volatile": "random_forest",
    })

    # Feature settings
    use_physics_features: bool = True
    use_steering_features: bool = True
    use_quantum_features: bool = True

    # Quantile settings
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.50, 0.95])

    # Model hyperparameters
    gb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
    })
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_leaf": 5,
    })
    ridge_alpha: float = 1.0

    # Calibration
    sigma_scale: float = 1.0
    min_confidence_width: float = 0.005  # Minimum 0.5% interval


# Core features that should always exist
CORE_FEATURES = [
    "returns", "realized_vol", "ofi", "range_pct",
    "flow_momentum", "regime_stability", "barrier_proximity",
]

# Physics features from PriceParticle
PHYSICS_FEATURES = [
    "velocity", "acceleration", "jerk", "momentum_alignment",
    "mass", "force_net", "force_impulse",
    "kinetic_energy", "potential_energy", "energy_injection",
    "field_gradient", "field_strength",
    "mean_reversion_strength", "damping_ratio",
    "vwap_zscore", "volume_momentum",
    "momentum_state", "tension_state",
    "breakout_potential", "reversion_potential",
    "particle_physics_score",
]

# Steering field features (from new modules)
STEERING_FEATURES = [
    # Funding
    "funding_weighted", "funding_force", "funding_ema", "funding_divergence",
    # Liquidation
    "liquidation_force", "liquidation_asymmetry", "cascade_probability",
    # Gamma
    "gamma_force", "max_pain_force", "dealer_position",
    # Macro
    "macro_drift", "spx_correlation", "dxy_correlation",
    # Temporal
    "vol_multiplier", "session_vol_mult", "hour_vol_mult",
    # Orderbook
    "imbalance_force", "gravity_force", "depth_pressure",
    "large_order_bias", "flow_acceleration",
]

# Quantum features
QUANTUM_FEATURES = [
    "regime_code", "momentum_weighted", "spring_potential",
    "vol_ratio", "jump_intensity", "regime_stability",
]

# Regime features
REGIME_FEATURES = [
    "K_pmax", "P_pmax", "C_pmax", "O_pmax", "F_pmax", "G_pmax", "S_pmax",
    "regime_entropy",
]


class UnifiedPredictor:
    """Unified predictor integrating all components.

    This is the main prediction class that:
    1. Collects features from all sources
    2. Trains regime-aware ensemble models
    3. Produces calibrated quantile predictions
    """

    def __init__(self, config: Optional[UnifiedConfig] = None):
        """Initialize unified predictor.

        Args:
            config: Configuration object
        """
        self.config = config or UnifiedConfig()
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.quantile_models: Dict[float, Dict[str, Any]] = {}
        self._feature_cols: List[str] = []
        self._fitted = False

    def _get_all_feature_names(self) -> List[str]:
        """Get list of all potential feature names."""
        features = CORE_FEATURES.copy()

        if self.config.use_physics_features:
            features.extend(PHYSICS_FEATURES)

        if self.config.use_steering_features:
            features.extend(STEERING_FEATURES)

        if self.config.use_quantum_features:
            features.extend(QUANTUM_FEATURES)

        features.extend(REGIME_FEATURES)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for f in features:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Feature matrix
        """
        all_features = self._get_all_feature_names()

        # Filter to available columns
        available = [f for f in all_features if f in df.columns]

        if not self._fitted:
            self._feature_cols = available
        else:
            # Use same features as training
            available = [f for f in self._feature_cols if f in df.columns]
            # Pad missing features with 0
            missing = set(self._feature_cols) - set(available)
            if missing:
                for m in missing:
                    df = df.copy()
                    df[m] = 0.0
                available = self._feature_cols

        X = df[available].fillna(0).values
        return X

    def _train_quantile_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quantile: float,
        model_type: str,
    ) -> Any:
        """Train a single quantile model.

        Args:
            X: Feature matrix
            y: Target values
            quantile: Quantile level
            model_type: Type of model to train

        Returns:
            Trained model
        """
        if model_type == "gradient_boost":
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=self.config.gb_params.get('n_estimators', 100),
                max_depth=self.config.gb_params.get('max_depth', 4),
                learning_rate=self.config.gb_params.get('learning_rate', 0.1),
                subsample=self.config.gb_params.get('subsample', 0.8),
                random_state=42,
            )
            model.fit(X, y)

        elif model_type == "random_forest":
            # RF doesn't support quantile loss directly - use weighted samples
            weights = np.where(y > 0, quantile, 1 - quantile)
            model = RandomForestRegressor(
                n_estimators=self.config.rf_params.get('n_estimators', 100),
                max_depth=self.config.rf_params.get('max_depth', 6),
                min_samples_leaf=self.config.rf_params.get('min_samples_leaf', 5),
                random_state=42,
            )
            model.fit(X, y, sample_weight=weights)

        elif model_type == "ridge":
            # Ridge with weighted loss approximation
            weights = np.where(y > 0, quantile, 1 - quantile)
            model = Ridge(alpha=self.config.ridge_alpha)
            model.fit(X, y, sample_weight=weights)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "future_return",
    ) -> "UnifiedPredictor":
        """Fit unified predictor on training data.

        Args:
            train_df: Training DataFrame with features and target
            target_col: Name of target column

        Returns:
            Self
        """
        # Extract features
        X = self._extract_features(train_df)
        y = train_df[target_col].fillna(0).values

        # Scale features
        X = self.scaler.fit_transform(X)

        # Train models for each quantile
        model_types = ["gradient_boost", "random_forest", "ridge"]

        for q in self.config.quantiles:
            self.quantile_models[q] = {}
            for model_type in model_types:
                try:
                    model = self._train_quantile_model(X, y, q, model_type)
                    self.quantile_models[q][model_type] = model
                except Exception as e:
                    warnings.warn(f"Failed to train {model_type} for q={q}: {e}")

        self._fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate predictions.

        Args:
            df: DataFrame with features
            regime: Optional regime override ('trending', 'ranging', 'volatile')

        Returns:
            DataFrame with predictions
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract and scale features
        X = self._extract_features(df)
        X = self.scaler.transform(X)

        # Detect regime if not provided
        if regime is None and self.config.regime_aware:
            regime = self._detect_regime(df)

        # Get ensemble weights
        if self.config.regime_aware and regime in self.config.regime_model_overrides:
            # Boost preferred model for this regime
            preferred = self.config.regime_model_overrides[regime]
            weights = self.config.ensemble_weights.copy()
            weights[preferred] = weights.get(preferred, 0.3) * 1.5
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        else:
            weights = self.config.ensemble_weights

        # Build result DataFrame
        result = df[["symbol", "bar_idx", "timestamp_end", "close"]].copy()

        # Predict each quantile
        for q in self.config.quantiles:
            preds = []
            total_weight = 0

            for model_type, weight in weights.items():
                if model_type == "physics":
                    # Physics prediction from quantum features
                    if "particle_physics_score" in df.columns:
                        pred = df["particle_physics_score"].fillna(0).values * 0.01
                    else:
                        pred = np.zeros(len(df))
                    preds.append((pred, weight))
                    total_weight += weight
                elif model_type in self.quantile_models.get(q, {}):
                    model = self.quantile_models[q][model_type]
                    pred = model.predict(X)
                    preds.append((pred, weight))
                    total_weight += weight

            # Weighted ensemble
            if preds:
                ensemble_pred = np.zeros(len(df))
                for pred, weight in preds:
                    ensemble_pred += pred * (weight / total_weight)
                result[f"q{int(q*100):02d}"] = ensemble_pred
            else:
                result[f"q{int(q*100):02d}"] = 0.0

        # Point estimate is median
        result["x_hat"] = result["q50"]

        # Compute uncertainty
        result["sigma_hat"] = (result["q95"] - result["q05"]) / 3.29

        # Apply sigma scale and minimum width
        if self.config.sigma_scale != 1.0:
            center = result["q50"]
            half_width = (result["q95"] - result["q05"]) / 2 * self.config.sigma_scale
            half_width = half_width.clip(lower=self.config.min_confidence_width / 2)
            result["q05"] = center - half_width
            result["q95"] = center + half_width
            result["sigma_hat"] = (result["q95"] - result["q05"]) / 3.29

        return result

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime from data.

        Args:
            df: DataFrame with price data

        Returns:
            Regime string ('trending', 'ranging', 'volatile')
        """
        if len(df) < 20:
            return "ranging"

        recent = df.tail(60)

        if "returns" in recent.columns:
            returns = recent["returns"].dropna()
        else:
            returns = recent["close"].pct_change().dropna()

        if len(returns) < 10:
            return "ranging"

        # Trend strength
        up_moves = (returns > 0).sum()
        down_moves = (returns < 0).sum()
        total = up_moves + down_moves
        trend_strength = abs(up_moves - down_moves) / total if total > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(60)

        # Classify
        if trend_strength > 0.6 and volatility < 0.025:
            return "trending"
        elif volatility > 0.04:
            return "volatile"
        else:
            return "ranging"

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained models.

        Returns:
            DataFrame with feature importances
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        importances = {}

        # Get importances from gradient boosting (median quantile)
        if 0.50 in self.quantile_models and "gradient_boost" in self.quantile_models[0.50]:
            gb = self.quantile_models[0.50]["gradient_boost"]
            importances["gradient_boost"] = gb.feature_importances_

        # Get importances from random forest
        if 0.50 in self.quantile_models and "random_forest" in self.quantile_models[0.50]:
            rf = self.quantile_models[0.50]["random_forest"]
            importances["random_forest"] = rf.feature_importances_

        if not importances:
            return pd.DataFrame()

        df = pd.DataFrame(importances, index=self._feature_cols)
        df["mean_importance"] = df.mean(axis=1)
        df = df.sort_values("mean_importance", ascending=False)

        return df


class SteeringFieldFeatureEngine:
    """Compute all steering field features for a DataFrame.

    This class coordinates feature extraction from all steering field modules
    and combines them into a unified feature set for prediction.
    """

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        """Initialize feature engine.

        Args:
            configs: Dict of configs for each module
        """
        self.configs = configs or {}
        self._modules_initialized = False

    def _init_modules(self):
        """Lazily initialize steering field modules."""
        if self._modules_initialized:
            return

        try:
            from ..particle.funding import FundingRateAggregator, FundingConfig
            from ..particle.liquidations import LiquidationHeatmap, LiquidationConfig
            from ..particle.gamma import GammaFieldCalculator, GammaConfig
            from ..particle.macro import CrossAssetCoupling, MacroCouplingConfig
            from ..particle.temporal import TimeOfDayEffects, TemporalConfig
            from ..particle.orderbook import MultiLevelOrderBookAnalyzer, OrderBookConfig
            from ..particle.physics import PriceParticle

            self.funding = FundingRateAggregator(
                self.configs.get('funding', FundingConfig())
            )
            self.liquidations = LiquidationHeatmap(
                self.configs.get('liquidation', LiquidationConfig())
            )
            self.gamma = GammaFieldCalculator(
                self.configs.get('gamma', GammaConfig())
            )
            self.macro = CrossAssetCoupling(
                self.configs.get('macro', MacroCouplingConfig())
            )
            self.temporal = TimeOfDayEffects(
                self.configs.get('temporal', TemporalConfig())
            )
            self.orderbook = MultiLevelOrderBookAnalyzer(
                self.configs.get('orderbook', OrderBookConfig())
            )
            self.physics = PriceParticle(self.configs.get('physics', {}))

            self._modules_initialized = True
        except ImportError as e:
            warnings.warn(f"Could not initialize steering modules: {e}")
            self._modules_initialized = False

    def compute_features(
        self,
        df: pd.DataFrame,
        funding_rates: Optional[Dict[str, float]] = None,
        orderbook: Optional[Dict[str, List]] = None,
        current_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Compute all steering field features.

        Args:
            df: DataFrame with OHLCV data
            funding_rates: Funding rates by exchange
            orderbook: Order book data
            current_time: Current datetime

        Returns:
            DataFrame with additional features
        """
        self._init_modules()
        result = df.copy()

        # Compute physics features first (base features)
        if self._modules_initialized:
            try:
                result = self.physics.compute_features(result)
            except Exception as e:
                warnings.warn(f"Physics feature computation failed: {e}")

        # Add steering field features as scalars/series
        if self._modules_initialized and len(result) > 0:
            current_price = float(result["close"].iloc[-1])
            current_time = current_time or datetime.now()

            # Detect regime
            regime_str = self._detect_regime_simple(result)

            # Recent volatility
            returns = result["returns"] if "returns" in result.columns else result["close"].pct_change()
            recent_vol = returns.tail(60).std() if len(returns) >= 60 else 0.02

            # Funding features
            if funding_rates:
                try:
                    funding_result = self.funding.aggregate(funding_rates, regime_str)
                    result["funding_force"] = funding_result.get('funding_force', 0)
                    result["funding_weighted"] = funding_result.get('weighted_funding', 0)
                except Exception:
                    pass

            # Temporal features (applies to all rows based on current time pattern)
            try:
                temporal_result = self.temporal.calculate_combined_multiplier(
                    current_time, recent_vol
                )
                result["vol_multiplier"] = temporal_result.get('combined_multiplier', 1.0)
                result["session_vol_mult"] = temporal_result.get('session_multiplier', 1.0)
            except Exception:
                pass

            # Order book features (if available)
            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                try:
                    snapshot = self.orderbook.process_snapshot(
                        orderbook['bids'], orderbook['asks']
                    )
                    ob_result = self.orderbook.calculate_force(snapshot, regime_str)
                    result["imbalance_force"] = ob_result.get('imbalance_force', 0)
                    result["gravity_force"] = ob_result.get('gravity_force', 0)
                except Exception:
                    pass

        return result

    def _detect_regime_simple(self, df: pd.DataFrame) -> str:
        """Simple regime detection."""
        if len(df) < 20:
            return "ranging"

        returns = df["returns"] if "returns" in df.columns else df["close"].pct_change()
        returns = returns.dropna().tail(60)

        if len(returns) < 10:
            return "ranging"

        vol = returns.std()
        trend = abs(returns.mean()) / (vol + 1e-9)

        if trend > 0.3 and vol < 0.02:
            return "trending"
        elif vol > 0.03:
            return "volatile"
        return "ranging"


def create_prediction_pipeline(
    config: Optional[Dict[str, Any]] = None
) -> Tuple[SteeringFieldFeatureEngine, UnifiedPredictor]:
    """Create a complete prediction pipeline.

    Args:
        config: Configuration dict

    Returns:
        Tuple of (feature_engine, predictor)
    """
    config = config or {}

    # Create feature engine
    feature_engine = SteeringFieldFeatureEngine(
        configs=config.get('steering_fields', {})
    )

    # Create predictor with defaults optimized for accuracy
    predictor_config = UnifiedConfig(
        use_ensemble=config.get('use_ensemble', True),
        regime_aware=config.get('regime_aware', True),
        use_physics_features=True,
        use_steering_features=True,
        use_quantum_features=True,
    )
    predictor = UnifiedPredictor(predictor_config)

    return feature_engine, predictor
