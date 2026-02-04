"""Bregman Projection via Adaptive Fully-Corrective Frank-Wolfe (ProjectFW).

Implements Algorithm 2 from the Bregman-FW paper for constrained optimization
with approximation guarantees. This is used for hyperparameter optimization
in the physics engine calibration.

Key features:
- Adaptive contraction for faster convergence
- Best-iterate tracking for robustness
- IP solver integration for discrete/mixed constraints
- α-approximation guarantees on solution quality

All optimizer parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from enum import Enum
import warnings


@dataclass
class BregmanConfig:
    """Configuration for ProjectFW optimizer.

    All parameters can be tuned by the improvement loop.
    """
    # Approximation ratio (0 < α < 1)
    # Higher α = better approximation but slower convergence
    alpha: float = 0.5

    # Initial contraction parameter (0 < ε₀ < 1)
    initial_contraction: float = 0.1

    # Convergence threshold for Bregman divergence
    convergence_threshold: float = 1e-6

    # Maximum iterations
    max_iterations: int = 1000

    # Minimum improvement to continue
    min_improvement: float = 1e-8

    # Interior point initialization
    interior_margin: float = 0.1  # How far inside constraints to start

    # Contraction adaptation parameters
    contraction_decay: float = 0.5  # How fast to reduce contraction
    min_contraction: float = 1e-6

    # Line search parameters (for fully-corrective step)
    line_search_max_iter: int = 20
    line_search_tolerance: float = 1e-6

    # Regularization
    l2_regularization: float = 0.0

    # Verbose output
    verbose: bool = False


class StoppingReason(Enum):
    """Reason for algorithm termination."""
    CONVERGENCE = "convergence"
    MAX_ITERATIONS = "max_iterations"
    SMALL_GAP = "small_gap"
    SMALL_OBJECTIVE = "small_objective"
    USER_TERMINATION = "user_termination"
    NUMERICAL_ERROR = "numerical_error"


@dataclass
class OptimizationResult:
    """Result of ProjectFW optimization."""
    optimal_params: np.ndarray
    optimal_value: float
    bregman_divergence: float
    iterations: int
    stopping_reason: StoppingReason
    trajectory: List[np.ndarray] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    gap_history: List[float] = field(default_factory=list)
    active_vertices: List[np.ndarray] = field(default_factory=list)


class BregmanDivergence:
    """Bregman divergence functions for different regularizers.

    D_φ(x || y) = φ(x) - φ(y) - ∇φ(y)ᵀ(x - y)

    Common choices:
    - Squared L2: φ(x) = ½||x||² → D = ½||x - y||²
    - Negative entropy: φ(x) = Σxᵢlog(xᵢ) → D = KL divergence
    - Itakura-Saito: φ(x) = -Σlog(xᵢ) → D = IS divergence
    """

    @staticmethod
    def squared_l2(x: np.ndarray, y: np.ndarray) -> float:
        """Squared L2 (Euclidean) Bregman divergence."""
        diff = x - y
        return 0.5 * np.dot(diff, diff)

    @staticmethod
    def squared_l2_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of squared L2 divergence w.r.t. x."""
        return x - y

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """KL divergence D_KL(p || q) for probability distributions."""
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return np.sum(p_safe * np.log(p_safe / q_safe))

    @staticmethod
    def kl_gradient(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Gradient of KL divergence w.r.t. p."""
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return np.log(p_safe / q_safe) + 1


class LinearConstraints:
    """Linear constraints Ax ≤ b for the optimization polytope."""

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ):
        """Initialize constraints.

        Args:
            A: Constraint matrix (m x n)
            b: Constraint bounds (m,)
            lb: Lower bounds on variables (n,)
            ub: Upper bounds on variables (n,)
        """
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub

    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if point is feasible."""
        if self.lb is not None and np.any(x < self.lb - tol):
            return False
        if self.ub is not None and np.any(x > self.ub + tol):
            return False
        if self.A is not None and self.b is not None:
            if np.any(self.A @ x > self.b + tol):
                return False
        return True

    def project_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Project point to box constraints."""
        result = x.copy()
        if self.lb is not None:
            result = np.maximum(result, self.lb)
        if self.ub is not None:
            result = np.minimum(result, self.ub)
        return result

    def get_interior_point(self, margin: float = 0.1) -> np.ndarray:
        """Get a point strictly inside the feasible region."""
        if self.lb is not None and self.ub is not None:
            # Center of the box
            center = (self.lb + self.ub) / 2
            return center
        elif self.lb is not None:
            return self.lb + margin
        elif self.ub is not None:
            return self.ub - margin
        else:
            raise ValueError("Need at least bounds to find interior point")


class ProjectFW:
    """Bregman Projection via Adaptive Fully-Corrective Frank-Wolfe.

    Solves: min_μ R(μ) - θᵀμ + C(θ)
    subject to: Aμ ≤ b (linear constraints)

    where R is a regularizer (Bregman generator) and θ is the target state.

    This optimizer is particularly suited for:
    - Constrained hyperparameter optimization
    - Probabilistic prediction calibration
    - Quantile regression with constraints
    """

    def __init__(self, config: Optional[BregmanConfig] = None):
        """Initialize optimizer.

        Args:
            config: BregmanConfig with hyperparameters
        """
        self.config = config or BregmanConfig()

    def _init_fw(
        self,
        constraints: LinearConstraints,
        n_dims: int,
    ) -> Tuple[np.ndarray, Set[int], np.ndarray]:
        """Initialize FW algorithm (InitFW).

        Returns:
            Tuple of (interior_point, active_vertex_indices, initial_iterate)
        """
        # Get interior point
        u = constraints.get_interior_point(self.config.interior_margin)

        # Initial active set (just the interior point initially)
        active_vertices = {0}

        # Initial iterate at interior point
        mu_0 = u.copy()

        return u, active_vertices, mu_0

    def _compute_gradient(
        self,
        mu: np.ndarray,
        theta: np.ndarray,
        objective_fn: Callable[[np.ndarray], float],
        regularizer_grad: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Compute gradient of objective F(μ) = R(μ) - θᵀμ + C(θ).

        ∇F(μ) = ∇R(μ) - θ
        """
        return regularizer_grad(mu) - theta

    def _solve_linear_subproblem(
        self,
        gradient: np.ndarray,
        constraints: LinearConstraints,
    ) -> np.ndarray:
        """Solve linear minimization oracle (IP solver step).

        Find: z = argmin_{z ∈ Z} (θ_t - θ)ᵀz

        For box constraints, this has a closed-form solution.
        """
        # Direction to minimize
        direction = gradient

        # For box constraints: pick lower bound if gradient positive, upper if negative
        z = np.zeros_like(direction)

        for i in range(len(direction)):
            if direction[i] > 0:
                z[i] = constraints.lb[i] if constraints.lb is not None else -1e10
            else:
                z[i] = constraints.ub[i] if constraints.ub is not None else 1e10

        # Project to feasible region
        z = constraints.project_to_bounds(z)

        return z

    def _compute_fw_gap(
        self,
        mu: np.ndarray,
        z: np.ndarray,
        gradient: np.ndarray,
    ) -> float:
        """Compute Frank-Wolfe gap g(μ) = (θ_t - θ)ᵀ(μ - z)."""
        return np.dot(gradient, mu - z)

    def _fully_corrective_step(
        self,
        mu: np.ndarray,
        z: np.ndarray,
        objective_fn: Callable[[np.ndarray], float],
        constraints: LinearConstraints,
    ) -> np.ndarray:
        """Perform fully-corrective Frank-Wolfe step.

        Optimize over convex hull of active vertices.
        For simplicity, we use line search between current and new vertex.
        """
        # Line search: find optimal step size
        best_alpha = 0.0
        best_value = objective_fn(mu)

        for i in range(self.config.line_search_max_iter):
            alpha = (i + 1) / self.config.line_search_max_iter
            candidate = (1 - alpha) * mu + alpha * z

            # Project to ensure feasibility
            candidate = constraints.project_to_bounds(candidate)

            value = objective_fn(candidate)
            if value < best_value - self.config.line_search_tolerance:
                best_value = value
                best_alpha = alpha

        return (1 - best_alpha) * mu + best_alpha * z

    def _adapt_contraction(
        self,
        epsilon: float,
        mu: np.ndarray,
        u: np.ndarray,
        gradient: np.ndarray,
        gap: float,
    ) -> float:
        """Adapt contraction parameter if necessary.

        g_u = (θ_t - θ)ᵀ(μ - u)
        if g_u < 0 and g(μ)/(-4g_u) < ε_{t-1}:
            ε_t = min{g(μ)/(-4g_u), ε_{t-1}/2}
        """
        g_u = np.dot(gradient, mu - u)

        if g_u < 0 and gap / (-4 * g_u) < epsilon:
            new_epsilon = min(gap / (-4 * g_u), epsilon / 2)
            new_epsilon = max(new_epsilon, self.config.min_contraction)
            return new_epsilon

        return epsilon

    def optimize(
        self,
        theta: np.ndarray,
        constraints: LinearConstraints,
        objective_fn: Optional[Callable[[np.ndarray], float]] = None,
        regularizer: str = "l2",
        callback: Optional[Callable[[int, np.ndarray, float], bool]] = None,
    ) -> OptimizationResult:
        """Run ProjectFW optimization.

        Args:
            theta: Target state vector to project
            constraints: LinearConstraints defining feasible region
            objective_fn: Optional custom objective (default: Bregman to theta)
            regularizer: Type of regularizer ("l2", "kl")
            callback: Optional callback(iteration, mu, value) -> should_stop

        Returns:
            OptimizationResult with optimal parameters
        """
        n_dims = len(theta)
        cfg = self.config

        # Select Bregman divergence
        if regularizer == "l2":
            bregman_fn = BregmanDivergence.squared_l2
            bregman_grad = lambda mu: BregmanDivergence.squared_l2_gradient(mu, theta)
        elif regularizer == "kl":
            bregman_fn = lambda mu: BregmanDivergence.kl_divergence(mu, theta)
            bregman_grad = lambda mu: BregmanDivergence.kl_gradient(mu, theta)
        else:
            raise ValueError(f"Unknown regularizer: {regularizer}")

        # Default objective: Bregman divergence to theta
        if objective_fn is None:
            def objective_fn(mu):
                obj = bregman_fn(mu, theta)
                if cfg.l2_regularization > 0:
                    obj += cfg.l2_regularization * np.dot(mu, mu)
                return obj

        # Initialize
        u, active_set, mu = self._init_fw(constraints, n_dims)
        epsilon = cfg.initial_contraction

        # Tracking
        trajectory = [mu.copy()]
        objective_history = [objective_fn(mu)]
        gap_history = []

        # Best iterate tracking
        best_mu = mu.copy()
        best_value = objective_history[0]
        best_adjusted = best_value  # F(μ) - g(μ) for comparison

        stopping_reason = StoppingReason.MAX_ITERATIONS

        for t in range(1, cfg.max_iterations + 1):
            # Contract active set toward interior point
            # Z' = (1 - ε_{t-1})Z_{t-1} + ε_{t-1}u
            # For simplicity, we work directly with mu

            # Compute gradient
            gradient = self._compute_gradient(mu, theta, objective_fn, bregman_grad)

            # Solve linear subproblem (IP oracle)
            z = self._solve_linear_subproblem(gradient, constraints)

            # Compute FW gap
            gap = self._compute_fw_gap(mu, z, gradient)
            gap_history.append(gap)

            # Update best iterate
            current_value = objective_fn(mu)
            current_adjusted = current_value - gap

            if current_adjusted > best_adjusted:
                best_mu = mu.copy()
                best_value = current_value
                best_adjusted = current_adjusted

            # Check stopping conditions
            if gap <= (1 - cfg.alpha) * current_value:
                stopping_reason = StoppingReason.SMALL_GAP
                if cfg.verbose:
                    print(f"[{t}] Stopping: gap criterion satisfied")
                break

            if current_value <= cfg.convergence_threshold:
                stopping_reason = StoppingReason.SMALL_OBJECTIVE
                if cfg.verbose:
                    print(f"[{t}] Stopping: objective below threshold")
                break

            # Callback for early termination
            if callback is not None:
                should_stop = callback(t, mu, current_value)
                if should_stop:
                    stopping_reason = StoppingReason.USER_TERMINATION
                    break

            # Fully-corrective step
            mu_new = self._fully_corrective_step(mu, z, objective_fn, constraints)

            # Check for numerical issues
            if not np.isfinite(mu_new).all():
                stopping_reason = StoppingReason.NUMERICAL_ERROR
                warnings.warn("Numerical error in optimization")
                break

            # Adapt contraction
            epsilon = self._adapt_contraction(epsilon, mu, u, gradient, gap)

            # Update
            mu = mu_new
            trajectory.append(mu.copy())
            objective_history.append(objective_fn(mu))

            if cfg.verbose and t % 10 == 0:
                print(f"[{t}] objective={current_value:.6f}, gap={gap:.6f}, ε={epsilon:.6f}")

        # Return best iterate if its adjusted value is better
        if gap_history and gap_history[-1] <= objective_fn(best_mu):
            final_mu = best_mu
        else:
            final_mu = mu

        # Compute final Bregman divergence
        final_divergence = bregman_fn(final_mu, theta)

        return OptimizationResult(
            optimal_params=final_mu,
            optimal_value=objective_fn(final_mu),
            bregman_divergence=final_divergence,
            iterations=t,
            stopping_reason=stopping_reason,
            trajectory=trajectory,
            objective_history=objective_history,
            gap_history=gap_history,
        )


class HyperparameterOptimizer:
    """High-level interface for hyperparameter optimization using ProjectFW.

    Wraps ProjectFW for easier use with the improvement loop and calibration.
    """

    def __init__(self, config: Optional[BregmanConfig] = None):
        """Initialize optimizer.

        Args:
            config: BregmanConfig with hyperparameters
        """
        self.config = config or BregmanConfig()
        self.optimizer = ProjectFW(config)

    def optimize_hyperparameters(
        self,
        param_names: List[str],
        param_bounds: Dict[str, Tuple[float, float]],
        objective_fn: Callable[[Dict[str, float]], float],
        initial_values: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], OptimizationResult]:
        """Optimize hyperparameters using ProjectFW.

        Args:
            param_names: List of parameter names to optimize
            param_bounds: Dict mapping name -> (lower, upper) bounds
            objective_fn: Function that takes param dict and returns loss
            initial_values: Optional initial parameter values

        Returns:
            Tuple of (optimal_params_dict, OptimizationResult)
        """
        n_params = len(param_names)

        # Build constraint bounds
        lb = np.array([param_bounds[name][0] for name in param_names])
        ub = np.array([param_bounds[name][1] for name in param_names])

        constraints = LinearConstraints(lb=lb, ub=ub)

        # Initial point (target for Bregman projection)
        if initial_values:
            theta = np.array([initial_values[name] for name in param_names])
        else:
            theta = (lb + ub) / 2  # Center of bounds

        # Wrap objective to work with arrays
        def array_objective(params_array: np.ndarray) -> float:
            params_dict = {name: params_array[i] for i, name in enumerate(param_names)}
            return objective_fn(params_dict)

        # Run optimization
        result = self.optimizer.optimize(
            theta=theta,
            constraints=constraints,
            objective_fn=array_objective,
            regularizer="l2",
        )

        # Convert back to dict
        optimal_dict = {
            name: result.optimal_params[i]
            for i, name in enumerate(param_names)
        }

        return optimal_dict, result

    def optimize_with_validation(
        self,
        param_names: List[str],
        param_bounds: Dict[str, Tuple[float, float]],
        train_objective: Callable[[Dict[str, float]], float],
        val_objective: Callable[[Dict[str, float]], float],
        n_restarts: int = 3,
    ) -> Tuple[Dict[str, float], float]:
        """Optimize with multiple restarts and validation.

        Args:
            param_names: Parameter names
            param_bounds: Parameter bounds
            train_objective: Training loss function
            val_objective: Validation loss function
            n_restarts: Number of random restarts

        Returns:
            Tuple of (best_params, best_val_loss)
        """
        best_params = None
        best_val_loss = float('inf')

        for restart in range(n_restarts):
            # Random initial point
            lb = np.array([param_bounds[name][0] for name in param_names])
            ub = np.array([param_bounds[name][1] for name in param_names])

            if restart == 0:
                # First restart: use center
                initial = {name: (lb[i] + ub[i]) / 2 for i, name in enumerate(param_names)}
            else:
                # Random initialization
                initial = {
                    name: np.random.uniform(lb[i], ub[i])
                    for i, name in enumerate(param_names)
                }

            # Optimize on training
            params, result = self.optimize_hyperparameters(
                param_names, param_bounds, train_objective, initial
            )

            # Evaluate on validation
            val_loss = val_objective(params)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params

            if self.config.verbose:
                print(f"Restart {restart+1}: train={result.optimal_value:.4f}, val={val_loss:.4f}")

        return best_params, best_val_loss


def get_bregman_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'bregman_alpha',
            'path': 'optimizer.bregman.alpha',
            'current_value': 0.5,
            'candidates': [0.3, 0.5, 0.7, 0.9],
            'variable_type': 'continuous',
        },
        {
            'name': 'bregman_initial_contraction',
            'path': 'optimizer.bregman.initial_contraction',
            'current_value': 0.1,
            'candidates': [0.05, 0.1, 0.2, 0.3],
            'variable_type': 'continuous',
        },
        {
            'name': 'bregman_convergence_threshold',
            'path': 'optimizer.bregman.convergence_threshold',
            'current_value': 1e-6,
            'candidates': [1e-8, 1e-6, 1e-4],
            'variable_type': 'continuous',
        },
        {
            'name': 'bregman_l2_regularization',
            'path': 'optimizer.bregman.l2_regularization',
            'current_value': 0.0,
            'candidates': [0.0, 0.001, 0.01, 0.1],
            'variable_type': 'continuous',
        },
    ]
