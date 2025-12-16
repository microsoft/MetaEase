"""
Gradient Ascent with Gaussian Process Surrogate.

This module implements the gradient-based optimization component of MetaEase.
Since heuristics are arbitrary code (no analytic form), we use a Gaussian Process
(GP) surrogate to estimate the heuristic function and compute gradients.

Key components:
1. fit_gaussian_process: Fits a GP to heuristic evaluations
2. estimate_gradient_with_gp: Computes analytical GP gradient (closed-form)
3. gradient_ascent: Performs gradient ascent with path-aware constraints

The gradient of the gap is: ∇Gap = ∇Benchmark - ∇Heuristic
- Benchmark gradient: From Lagrangian duality (computed in opt_utils.py)
- Heuristic gradient: From GP surrogate (computed here)

Path-aware updates ensure gradient steps stay within the same code path to avoid
instability near non-differentiable boundaries (sorting, conditionals).
"""

import os
import json
import pickle
import time
import random
import argparse
import numpy as np
from typing import Optional, Callable, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

DO_SCALING = True  # Whether to scale inputs/outputs for GP (recommended)
DEBUG = False
RED = "\033[91m"
RESET = "\033[0m"
USE_DIRECT_DERIVATIVE = True  # Use analytical GP gradient (faster, more accurate)

# Function to flatten nested lists
def flatten(individual):
    """Flatten a nested list structure."""
    return [item for sublist in individual for item in sublist]


def flatten_dict(data):
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = value[0] if value else None
    return data


# Define the ConstantGP class globally
class ConstantGP:
    """
    Fallback GP model for constant functions.
    Used when all heuristic evaluations return the same value (degenerate case).
    This avoids numerical issues in GP fitting.
    """
    def __init__(self, constant_value):
        self.constant_value = constant_value

    def predict(self, X, return_std=False):
        """Predict constant value with zero uncertainty."""
        return np.full((X.shape[0],), self.constant_value), np.zeros((X.shape[0],))

    def fit(self, X, y):
        """No fitting necessary since it's a constant model."""
        pass


def fit_gaussian_process(population, fitness_scores, keys_for_heuristic):
    """
    Fit a Gaussian Process surrogate to heuristic evaluations.
    The GP models the heuristic function f(x) using observed evaluations.
    This allows us to compute gradients without re-evaluating the heuristic
    at every gradient step.

    Args:
        population: List of input samples (dictionaries)
        fitness_scores: List of heuristic values for each sample
        keys_for_heuristic: Which input variables to use for GP

    Returns:
        gp: Fitted GaussianProcessRegressor (or ConstantGP if degenerate)
        scaler_x: Input scaler (for inverse transform)
        scaler_y: Output scaler (for inverse transform)

    Kernel: RBF (Radial Basis Function) with constant scaling and white noise
    - RBF captures smoothness of heuristic function
    - WhiteKernel models observation noise
    """
    # Convert population to a 2D array (samples x features)
    X = np.array([[individual[key] for key in keys_for_heuristic] for individual in population])
    if X.ndim == 3:
        # Handle nested structure
        X = np.array([[individual[key][0] for key in keys_for_heuristic] for individual in population])

    # Validate input data
    if len(fitness_scores) == 0:
        print("Warning: Empty fitness scores array")
        return ConstantGP(0), None, None

    y = np.array(fitness_scores)

    # Check for invalid values (NaN, Inf)
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("Warning: Invalid values in fitness scores")
        valid_indices = ~(np.isnan(y) | np.isinf(y))
        X = X[valid_indices]
        y = y[valid_indices]
        if len(y) == 0:
            return ConstantGP(0), None, None

    # Ensure y is 2D for sklearn
    y = y.reshape(-1, 1)

    # Standardize inputs and outputs for numerical stability
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    if DO_SCALING:
        try:
            X = scaler_x.fit_transform(X)
            y = scaler_y.fit_transform(y)
        except ValueError as e:
            print(f"Warning: Scaling failed - {str(e)}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            print(f"X sample: {X[:5]}")
            print(f"y sample: {y[:5]}")
            return ConstantGP(np.mean(y)), None, None

    # Remove duplicate samples
    X, indices = np.unique(X, axis=0, return_index=True)
    y = y[indices]

    # Check if all fitness scores are the same (degenerate case)
    if len(set(y.flatten())) == 1:
        constant_value = y.flatten()[0]
        return ConstantGP(constant_value), scaler_x, scaler_y

    # Define kernel: Constant * RBF + WhiteKernel
    # - Constant kernel: scales the overall variance
    # - RBF kernel: captures smoothness (squared exponential)
    # - WhiteKernel: models observation noise
    kernel = C(1.0, (1e-2, 1e8)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e6)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))

    # Fit the Gaussian Process model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=25,  # Multiple restarts for better hyperparameter optimization
        alpha=1e-3,  # Small regularization for numerical stability
        random_state=42,  # For reproducibility
    )

    gp.fit(X, y)
    return gp, scaler_x, scaler_y


def predict_gaussian_process(gp, scaler_x, scaler_y, individual, keys_for_heuristic):
    # Prepare new data, making sure to use the same keys
    # if new_population is of type dict, convert it to a list
    if isinstance(individual, dict):
        X_new = np.array([[individual[key] for key in keys_for_heuristic]])
    else:
        X_new = np.array([individual])
    if DO_SCALING:
        X_new = scaler_x.transform(X_new)

    predictions = gp.predict(X_new.reshape(1, -1), return_std=True)
    predictions = scaler_y.inverse_transform(predictions)
    return predictions


def estimate_gradient_with_samples(
    heuristic_function,
    individual_array,
    population,
    fitness_scores,
    keys_for_heuristic,
    step_size=1,
):
    point = individual_array

    # Initialize an array to store the gradient
    gradient = np.zeros_like(individual_array)

    # Iterate over each dimension of the input point
    for i in range(len(individual_array)):
        # Create two perturbed points: one with a positive step and one with a negative step
        point_plus = individual_array.copy()
        point_minus = individual_array.copy()
        point_plus[i] += step_size
        point_minus[i] -= step_size

        # Calculate the heuristic function value at the perturbed points
        fitness_plus = heuristic_function(
            point_plus, population, fitness_scores, keys_for_heuristic
        )
        fitness_minus = heuristic_function(
            point_minus, population, fitness_scores, keys_for_heuristic
        )

        # Estimate the gradient using finite differences
        gradient[i] = (fitness_plus - fitness_minus) / (2 * step_size)

    return gradient


# estimate the heuristic function at a individual_array
def heuristic_function(
    individual_array, population, fitness_scores, keys_for_heuristic
):
    # Convert population to a numpy array for easy distance computation
    population_array = np.array(
        [[individual[key] for key in keys_for_heuristic] for individual in population]
    )
    # Calculate the Euclidean distances between the input point and all population points
    distances = np.linalg.norm(population_array - individual_array, axis=1)
    # Find the index of the closest population point
    closest_index = np.argmin(distances)
    # Return the a linear combination of the fitness score and the distance
    return fitness_scores[closest_index] + distances[closest_index]


def estimate_gradient_with_gp(gp, scaler_x, scaler_y, individual_array):
    """
    Compute the analytical gradient of the GP prediction using closed-form formula.

    This is much more efficient and accurate than finite differences. For a GP with
    RBF kernel, the gradient of the mean prediction is:

        ∇μ(x*) = Σ_i α_i * ∇k(x*, x_i)

    where:
        - α = (K(X,X) + σ²I)^(-1) @ y  (pre-computed during fitting)
        - ∇k(x*, x_i) is the gradient of the RBF kernel w.r.t. x*
        - K is the kernel matrix

    Args:
        gp: Fitted GaussianProcessRegressor
        scaler_x: Input scaler (for transforming x*)
        scaler_y: Output scaler (for scaling gradient back to original units)
        individual_array: Point at which to compute gradient

    Returns:
        gradient: ∇μ(x*) in original input space
    """
    if isinstance(gp, ConstantGP):
        return np.zeros_like(individual_array)

    # Ensure individual_array is 2D for sklearn
    if individual_array.ndim == 1:
        x_star = individual_array.reshape(1, -1)
    else:
        x_star = individual_array

    # Apply scaling if needed
    if DO_SCALING and scaler_x is not None:
        x_star_scaled = scaler_x.transform(x_star)
    else:
        x_star_scaled = x_star

    # Get training data
    X_train = gp.X_train_
    y_train = gp.y_train_

    # Get kernel parameters
    kernel = gp.kernel_
    if hasattr(kernel, 'k1') and hasattr(kernel, 'k2'):
        # Product kernel: C * RBF + WhiteKernel
        constant_kernel = kernel.k1.k1  # C kernel
        rbf_kernel = kernel.k1.k2       # RBF kernel
        white_kernel = kernel.k2        # WhiteKernel
    else:
        # Assume it's a simple RBF kernel
        rbf_kernel = kernel
        constant_kernel = None
        white_kernel = None

    # Get RBF parameters
    length_scale = rbf_kernel.length_scale
    if np.isscalar(length_scale):
        length_scale = np.full(x_star_scaled.shape[1], length_scale)

    # Get constant kernel parameter
    if constant_kernel is not None:
        constant_value = constant_kernel.constant_value
    else:
        constant_value = 1.0

    # Get white noise parameter
    if white_kernel is not None:
        noise_level = white_kernel.noise_level
    else:
        noise_level = 0.0

    # Compute kernel matrix K(X, X) + σ²I
    K_XX = rbf_kernel(X_train, X_train)
    if constant_kernel is not None:
        K_XX *= constant_value
    K_XX += noise_level * np.eye(X_train.shape[0])

    # Compute K(x*, X) - kernel between test point and training points
    K_xstar_X = rbf_kernel(x_star_scaled, X_train)
    if constant_kernel is not None:
        K_xstar_X *= constant_value

    # Solve for alpha = (K(X,X) + σ²I)^(-1) @ y
    try:
        alpha = np.linalg.solve(K_XX, y_train.flatten())
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if matrix is singular
        alpha = np.linalg.pinv(K_XX) @ y_train.flatten()

    # Compute gradient of kernel w.r.t. x*
    gradient = np.zeros_like(x_star_scaled)

    for i in range(x_star_scaled.shape[1]):
        # Gradient of RBF kernel w.r.t. i-th dimension
        # ∇k(x*, x) = -k(x*, x) * (x*_i - x_i) / length_scale_i^2
        diff = (x_star_scaled[0, i] - X_train[:, i]) / (length_scale[i] ** 2)
        kernel_grad_i = -K_xstar_X[0, :] * diff

        # Apply constant kernel scaling
        if constant_kernel is not None:
            kernel_grad_i *= constant_value

        # Compute gradient component
        gradient[0, i] = np.dot(kernel_grad_i, alpha)

    # Apply inverse scaling to gradient
    if DO_SCALING and scaler_x is not None:
        # For StandardScaler, the gradient needs to be scaled by 1/scale
        gradient = gradient / scaler_x.scale_

    # Apply inverse scaling to output
    if DO_SCALING and scaler_y is not None:
        gradient = gradient * scaler_y.scale_[0]

    return gradient.flatten()


def estimate_gradient_gp_finite_difference(gp, scaler_x, scaler_y, individual_array, step_size=1e-4):
    """
    Finite difference gradient estimation.
    Use estimate_gradient_analytical for better performance and accuracy.
    """
    gradient = np.zeros_like(individual_array)

    for i in range(len(individual_array)):
        individual_array[i] += step_size
        y_plus, _ = predict_gaussian_process(gp, scaler_x, scaler_y, individual_array, None)
        individual_array[i] -= 2 * step_size
        y_minus, _ = predict_gaussian_process(gp, scaler_x, scaler_y, individual_array, None)
        individual_array[i] += step_size
        gradient[i] = (y_plus.item() - y_minus.item()) / (
            2 * step_size
        )  # Extract scaler value

    return gradient


def gradient_ascent(
    gp, scaler_x, scaler_y, best_sample, population, gradient_ascent_rate, thresholds
):
    updated_population = []
    best_sample_array = np.array([best_sample[key] for key in best_sample])

    for individual in population:
        individual_array = np.array([individual[key] for key in individual])

        gradient = estimate_gradient(gp, scaler_x, scaler_y, individual_array, individual.keys())
        # print("Gradient ", gradient, 'X' * 10)

        updated_individual = {}
        for i, key in enumerate(individual.keys()):
            updated_individual[key] = (
                individual_array[i] + gradient_ascent_rate * gradient[i]
            )
            # ensure the values are within the specified bounds
            updated_individual[key] = min(
                max(updated_individual[key], min(thresholds[key])), max(thresholds[key])
            )

            # Check if the value is of ndarray type and convert to scaler if necessary
            if isinstance(updated_individual[key], np.ndarray):
                updated_individual[key] = updated_individual[
                    key
                ].item()  # Convert ndarray to scaler

        updated_population.append(updated_individual)

    # Add the best sample to the updated population
    # if best_sample values are lists, flatten them
    updated_population.append(flatten_dict(best_sample))
    return updated_population


def update_with_gradient(
    individual, gradient_dict, keys_for_heuristic, thresholds, assigned_fixed_keys=None
):
    """
    Update individual using gradient ascent.

    Args:
        individual: The current individual to update
        gradient_dict: Dictionary containing gradients for each key
        keys_for_heuristic: List of keys to consider for the heuristic
        thresholds: Dictionary of min/max thresholds for each key
        assigned_fixed_keys: Optional list of keys that should not be updated
    """
    updated_individual = {}

    for key in individual.keys():
        # Skip updating if key is in assignedFixedKeys
        if assigned_fixed_keys and key in assigned_fixed_keys:
            updated_individual[key] = individual[key]
            continue

        # adjust the gradient ascent rate based on the key thresholds
        gradient_ascent_rate = 0.002 * (max(thresholds[key]) - min(thresholds[key]))

        # Update the value using gradient ascent
        updated_individual[key] = (
            individual[key] + gradient_ascent_rate * gradient_dict[key]
        )

        # ensure the values are within the specified bounds
        updated_individual[key] = min(
            max(updated_individual[key], min(thresholds[key])), max(thresholds[key])
        )

        # Convert ndarray to scalar if necessary
        if isinstance(updated_individual[key], np.ndarray):
            updated_individual[key] = updated_individual[key].item()

    return updated_individual

def _numeric_derivative_wrt_input_key(func, heuristic_code_path, base_dict, key, thresholds, block_length):
    """
    Finite-difference derivative of L w.r.t. a variable in input_dict.
    L( x + block_length e_k ) - L( x ) / block_length
    """
    base_dict = dict(base_dict)
    x = base_dict[key]
    base_dict[key] = min(max(x, min(thresholds[key])), max(thresholds[key]))
    L0, L0_code_path = func(base_dict)
    base_dict[key] = min(max(x + block_length, min(thresholds[key])), max(thresholds[key]))
    Lp, Lp_code_path = func(base_dict)
    if L0_code_path != heuristic_code_path or Lp_code_path != heuristic_code_path:
        # try the other direction
        base_dict[key] = min(max(x - block_length, min(thresholds[key])), max(thresholds[key]))
        L0, L0_code_path = func(base_dict)
        base_dict[key] = min(max(x, min(thresholds[key])), max(thresholds[key]))
        Lp, Lp_code_path = func(base_dict)
        if L0_code_path != heuristic_code_path or Lp_code_path != heuristic_code_path:
            return 0
    return (Lp - L0) / (block_length)

def estimate_heuristic_gradient(func, heuristic_code_path, all_vars, keys_for_gradient, thresholds, block_length):
    """
    Estimate the gradient of the heuristic function.
    """
    gradient = {}
    for k, v in all_vars.items():
        if k in keys_for_gradient:
            gradient[k] = _numeric_derivative_wrt_input_key(func, heuristic_code_path, all_vars, k, thresholds, block_length)
    return gradient

def normal_gradient_ascent(
    best_sample,
    gradient_dict,
    keys_for_heuristic,
    thresholds,
    assigned_fixed_keys,
    compute_heuristic_code_path: Tuple[Callable, str],
    gradient_ascent_rate=0.2,
    num_keys_for_gradient=20,
    minimize_is_better=False,
    block_length=0.1,
    ignore_code_path=False,
    early_stop=False
):
    # print("Gradient ascent rate ", gradient_ascent_rate)
    if DEBUG:
        print(f"Using direct derivative, minimize_is_better: {minimize_is_better}, block_length: {block_length}")
    exec, heuristic_code_path = compute_heuristic_code_path
    selected_keys_for_gradient = random.sample(keys_for_heuristic, min(num_keys_for_gradient, len(keys_for_heuristic)))
    if assigned_fixed_keys:
        selected_keys_for_gradient = list(set(selected_keys_for_gradient) - set(assigned_fixed_keys))
    start_time = time.time()
    est_grad = estimate_heuristic_gradient(exec, heuristic_code_path, best_sample, selected_keys_for_gradient, thresholds, block_length)
    end_time = time.time()
    if DEBUG:
        print("selected_keys_for_gradient: ", selected_keys_for_gradient)
        print("est_grad: ", est_grad)
        print(f"Time taken to estimate gradient: {end_time - start_time} seconds")


    for key in est_grad.keys():
        if key in gradient_dict:
            gradient_dict[key] -= est_grad[key]
        if minimize_is_better:
            gradient_dict[key] = -gradient_dict[key]
    for key in keys_for_heuristic:
        if key in gradient_dict and key not in selected_keys_for_gradient:
            gradient_dict[key] = 0

    if assigned_fixed_keys:
        for key in assigned_fixed_keys:
            gradient_dict[key] = 0

    if DEBUG:
        print("est_grad sum: ", sum(est_grad.values()))
        print("gradient_dict sum: ", sum(gradient_dict.values()))

    new_best_sample = best_sample.copy()
    for key in best_sample.keys():
        if key in gradient_dict:
            new_best_sample[key] = best_sample[key] + gradient_ascent_rate * gradient_dict[key]
            new_best_sample[key] = min(
                max(new_best_sample[key], min(thresholds[key])), max(thresholds[key])
            )
    code_path, _ = exec(new_best_sample)
    if code_path == heuristic_code_path or ignore_code_path:
        if all(new_best_sample[key] == best_sample[key] for key in keys_for_heuristic) and not early_stop:
            print("Best sample and new best sample are the same")
            # if the best sample and the new best sample are at the boundary, return all zero to indicate convergence
            return {key: 0 for key in best_sample.keys()}
        else:
            # if the best sample and the new best sample are not at the boundary, return the new best sample
            return new_best_sample
    else:
        print("Code path is different")
        # if the code path is different, return None to indicate that the gradient ascent is not successful
        return None

# TODO: this function needs documentation that explains what we are doing and why.
def update_with_closest_angle_to_gradient(
    population,
    best_sample,
    gradient_dict,
    keys_for_heuristic,
    thresholds,
    assigned_fixed_keys=None,
    compute_heuristic_code_path: Optional[Tuple[Callable, str]] = None,
    gradient_ascent_rate=0.2,
    disable_guassian_process=False,
    minimize_is_better=False,
    block_length=0.1,
    ignore_code_path=False,
    early_stop=False
):
    if compute_heuristic_code_path is not None and disable_guassian_process:
        # the gradient_dict is incomplete and we need to compute it
        x =  normal_gradient_ascent(
            best_sample,
            gradient_dict,
            keys_for_heuristic,
            thresholds,
            assigned_fixed_keys,
            compute_heuristic_code_path,
            gradient_ascent_rate=gradient_ascent_rate,
            minimize_is_better=minimize_is_better,
            block_length=block_length,
            ignore_code_path=ignore_code_path,
            early_stop=early_stop
        )
        if x is not None:
            return x

    # pick the individual that has the closest angle to the gradient
    best_angle = 180
    best_individual = None
    keys_for_heuristic_gradient = np.array(
        [gradient_dict[key] for key in keys_for_heuristic]
    )
    angles = []

    for i, ind in enumerate(population):
        if ind == best_sample:
            continue
        individual_array = np.array(
            [ind[key] - best_sample[key] for key in keys_for_heuristic]
        )
        angle = abs(
            np.arccos(
                np.dot(individual_array, keys_for_heuristic_gradient)
                / (
                    np.linalg.norm(individual_array)
                    * np.linalg.norm(keys_for_heuristic_gradient)
                    + 1e-12
                )
            )
            * 180
            / np.pi
        )
        if np.isnan(angle):
            angle = 89
        else:
            angle = angle.item()

        angles.append(angle)
        if angle < best_angle:
            best_angle = angle
            best_individual = ind

    if len(angles) == 0:
        print("No individuals in the population")
        return best_sample

    if DEBUG:
        print("Angles max ", max(angles), " min ", min(angles))
        print("Best angle ", best_angle)

    # Now do a normal gradient ascent on the keys of the best individual that are not in the keys_for_heuristic
    allkeys = best_individual.keys()
    remaning_keys = [key for key in allkeys if key not in keys_for_heuristic]
    for key in remaning_keys:
        # Skip updating if key is in assignedFixedVariables
        if assigned_fixed_keys:
            if key in assigned_fixed_keys:
                best_individual[key] = best_sample[key]
                continue


        gradient_ascent_rate = max(1, 0.02 * (max(thresholds[key]) - min(thresholds[key])))
        best_individual[key] = (
            best_sample[key] + gradient_dict[key] * gradient_ascent_rate
        )
        best_individual[key] = min(
            max(best_individual[key], min(thresholds[key])), max(thresholds[key])
        )

    if best_angle >= 90 and np.linalg.norm(keys_for_heuristic_gradient) != 0:
        if not early_stop:
            # We don't want to early stop here, do it again
            return best_sample
        # No individual has an angle less than 90 degrees
        # The path-based equivalance class of heuristic, has no more points in the direction of the gradient
        # return a zero sample
        print("No individual has an angle less than 90 degrees")
        return {key: 0 for key in allkeys}
    return best_individual


def subtract_values(optimal, heuristic):
    if isinstance(optimal, dict) and isinstance(heuristic, dict):
        return {
            subkey: subtract_values(optimal[subkey], heuristic[subkey])
            for subkey in optimal.keys() & heuristic.keys()
        }
    elif isinstance(optimal, (int, float, np.ndarray)) and isinstance(
        heuristic, (int, float, np.ndarray)
    ):
        return optimal - heuristic
    elif (
        isinstance(optimal, dict)
        and isinstance(heuristic, (int, float, np.ndarray))
        and heuristic == 0
    ):
        return {subkey: optimal[subkey] for subkey in optimal.keys()}
    else:
        raise ValueError("Invalid type for optimal and heuristic values")


def gradient_ascent_with_optimal_gradient(
    population,
    fitness_scores,
    gp,
    scaler_x,
    scaler_y,
    best_sample,
    gradient_ascent_rate,
    thresholds,
    optimal_gradient,
    max_iterations,
    keys_for_heuristic,
    heu_first=False,
    only_dual=False,
    assigned_fixed_keys=None,
    minimize_is_better=False,
):
    """
    Note that the optimal has the dual and auxiliary variables as well, so the keys_for_heuristic should be the original demand keys for the heuristic gradient
    This function is used when the optimal gradient is available, and is used for only updating the best sample
    """
    updated_population = []
    individual = best_sample
    # best_sample_keys = best_sample.keys()
    best_sample_array = np.array([best_sample[key] for key in best_sample])
    # print("Keys for heuristic ", keys_for_heuristic)
    if gp is not None and DEBUG:
        print(
            "Best sample GP estimate: ",
            predict_gaussian_process(gp, scaler_x, scaler_y, best_sample, keys_for_heuristic),
        )
    optimal_gradient_array = np.array([optimal_gradient[key] for key in best_sample])

    # do gradient ascent for best_sample
    individual_array = best_sample_array
    heuristic_array = np.array([individual[key] for key in keys_for_heuristic])
    if only_dual:
        heuristic_gradient_array = np.zeros_like(heuristic_array)
    else:
        heuristic_gradient_array = estimate_gradient_with_gp(gp, scaler_x, scaler_y, heuristic_array)
        # heuristic_gradient_array = estimate_gradient_with_samples(heuristic_function, heuristic_array, population, fitness_scores, keys_for_heuristic)
    heuristic_gradient_dict = {
        key: heuristic_gradient_array[i] for i, key in enumerate(keys_for_heuristic)
    }
    for key in individual.keys():
        if key not in keys_for_heuristic:
            heuristic_gradient_dict[key] = 0

    heuristic_gradient = np.array(
        [heuristic_gradient_dict[key] for key in individual.keys()]
    )

    # print("Heuristic gradient array ", heuristic_gradient)
    if not heu_first:
        gradient_dict = {
            key: optimal_gradient[key]
            for key in best_sample.keys()
        }
        # gradient = optimal_gradient_array - heuristic_gradient
    else:
        gradient_dict = {
            key: subtract_values(heuristic_gradient_dict[key], optimal_gradient[key])
            for key in best_sample.keys()
        }

        # gradient = heuristic_gradient - optimal_gradient_array
    if assigned_fixed_keys:
        for key in assigned_fixed_keys:
            gradient_dict[key] = 0
            heuristic_gradient_dict[key] = 0

    if DEBUG:
        print(f"{RED}Best sample before gradient ascent{RESET} ", individual)
        print(f"{RED}Optimal gradient from file{RESET} ", optimal_gradient)
        print(f"{RED}Heuristic gradient{RESET} ", heuristic_gradient_dict)
        # print(f"{RED}Gradients{RESET} ", gradient_dict)
    # best_sample = update_with_gradient(
    #     individual,
    #     gradient_dict,
    #     keys_for_heuristic,
    #     thresholds,
    #     assigned_fixed_keys
    # )
    best_sample = update_with_closest_angle_to_gradient(
        population,
        individual,
        gradient_dict,
        keys_for_heuristic,
        thresholds,
        assigned_fixed_keys,
        gradient_ascent_rate,
        minimize_is_better,
    )
    if DEBUG:
        print(f"{RED}Best sample after gradient ascent{RESET} ", best_sample)
    # print("Changed values ", {key: best_sample[key] for key in best_sample if best_sample[key] != individual[key]})
    # Add the best sample to the updated population
    # if best_sample values are lists, flatten them
    updated_population.append(flatten_dict(best_sample))
    return updated_population


def flatten_dict_values(data):
    flattened_dict = {}
    if isinstance(data, dict):
        for key, value in data.items():
            # If the value is a list, extract the first element, otherwise keep it as is
            if isinstance(value, list):
                flattened_dict[key] = value[0] if value else None
            else:
                flattened_dict[key] = value
    elif isinstance(data, list):
        flattened_dict = []
        for data_in_list in data:
            temp_dict = {}
            for key, value in data_in_list.items():
                if isinstance(value, list):
                    temp_dict[key] = value[0] if value else None
                else:
                    temp_dict[key] = value
            flattened_dict.append(temp_dict)
    return flattened_dict


def main():
    parser = argparse.ArgumentParser(
        description="Perform gradient ascent using Gaussian Process Regression."
    )

    parser.add_argument(
        "--best-sample-file",
        type=str,
        help="Path to the JSON file containing the best sample.",
    )
    parser.add_argument(
        "--population-file",
        type=str,
        help="Path to the JSON file containing the population.",
    )
    parser.add_argument(
        "--original-demand-file",
        type=str,
        help="Path to the JSON file containing the original demands, does not contain dual variables etc.",
    )
    parser.add_argument(
        "--fitness-scores-file",
        type=str,
        help="Path to the JSON file containing fitness scores.",
    )
    parser.add_argument(
        "--gradient-ascent-rate",
        type=float,
        help="Rate at which to perform gradient ascent.",
    )
    parser.add_argument(
        "--thresholds-file",
        type=str,
        help="Path to the JSON file containing the thresholds.",
    )
    parser.add_argument(
        "--optimal-gradient-file",
        type=str,
        default=None,
        help="Optional path to the file containing the gradient of the optimal.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of iterations for gradient ascent.",
    )
    parser.add_argument(
        "--heu-first",
        action="store_true",
        help="Whether to use the heuristic gradient first.",
    )
    parser.add_argument(
        "--only-dual",
        action="store_true",
        help="Whether to use only the optimal gradient.",
    )
    parser.add_argument(
        "--retrain-gp",
        action="store_true",
        help="Whether to retrain the GP again or not.",
    )
    parser.add_argument(
        "--path-to-assigned-fixed-keys",
        type=str,
        help="Path to the file that contains the fixed points",
        default=None,
    )
    args = parser.parse_args()
    save_dir = os.path.dirname(args.best_sample_file)
    optimal_gradient_available = False
    optimal_gradient = None
    # TODO: fix for USE_DIRECT_DERIVATIVE and when not
    if args.optimal_gradient_file:
        optimal_gradient_available = True
        with open(args.optimal_gradient_file, "r") as f:
            optimal_gradient = flatten_dict_values(json.load(f)[0])
        with open(args.original_demand_file, "r") as f:
            original_demand = flatten_dict_values(json.load(f))

    with open(args.best_sample_file, "r") as f:
        best_sample = flatten_dict_values(json.load(f))

    with open(args.population_file, "r") as f:
        population = flatten_dict_values(json.load(f))

    keys_for_heuristic = best_sample.keys()
    if optimal_gradient_available:
        keys_for_heuristic = original_demand.keys()

    assigned_fixed_keys = None
    if args.path_to_assigned_fixed_keys:
        with open(args.path_to_assigned_fixed_keys, "r") as f:
            assigned_fixed_keys = json.load(f)
        keys_for_heuristic = [
            key for key in keys_for_heuristic if key not in assigned_fixed_keys
        ]
        # print("Keys for heuristic ", keys_for_heuristic)

    # fitness_scores_file belongs to heuristic values if optimal_gradient_file is provided
    with open(args.fitness_scores_file, "r") as f:
        fitness_scores = json.load(f)

    with open(args.thresholds_file, "r") as f:
        thresholds = json.load(f)

    # Fit the Gaussian Process
    if args.only_dual:
        gp, scaler_x, scaler_y = None, None, None
    else:
        if f"{save_dir}/gp_model.pkl" not in os.listdir():
            args.retrain_gp = True

        if args.retrain_gp and not USE_DIRECT_DERIVATIVE:
            if DEBUG:
                # for each key in keys_for_heuristic, print the max and min values in the population
                for key in keys_for_heuristic:
                    print(
                        f"Max {key}: {max([individual[key] for individual in population])}, Min {key}: {min([individual[key] for individual in population])}"
                    )

                print(
                    "Max fitness score: ",
                    max(fitness_scores),
                    ", Min fitness score: ",
                    min(fitness_scores),
                )
                print("Num population samples:", len(population))
            # start_time = time.time()
            gp, scaler_x, scaler_y = fit_gaussian_process(
                population, fitness_scores, keys_for_heuristic
            )
            # end_time = time.time()
            # print(f"Time taken to fit the GP: {end_time - start_time} seconds")
            # Save the GP model as a pickle file
            with open(f"{save_dir}/gp_model.pkl", "wb") as f:
                pickle.dump(gp, f)
            with open(f"{save_dir}/scaler_x.pkl", "wb") as f:
                pickle.dump(scaler_x, f)
            with open(f"{save_dir}/scaler_y.pkl", "wb") as f:
                pickle.dump(scaler_y, f)
            # calculate the gp error
            if gp is not None:
                gp_error = 0
                for i in range(len(population)):
                    individual = population[i]
                    individual_array = np.array(
                        [individual[key] for key in keys_for_heuristic]
                    )
                    y, _ = predict_gaussian_process(
                        gp, scaler_x, scaler_y, individual, keys_for_heuristic
                    )
                    gp_error += (y.item() - fitness_scores[i]) ** 2
                gp_error /= len(population)
                if DEBUG:
                    print("GP error: ", gp_error)
        elif not USE_DIRECT_DERIVATIVE:
            with open(f"{save_dir}/gp_model.pkl", "rb") as f:
                gp = pickle.load(f)
            with open(f"{save_dir}/scaler_x.pkl", "rb") as f:
                scaler_x = pickle.load(f)
            with open(f"{save_dir}/scaler_y.pkl", "rb") as f:
                scaler_y = pickle.load(f)
        else:
            gp, scaler_x, scaler_y = None, None, None

    if optimal_gradient_available:
        updated_population = gradient_ascent_with_optimal_gradient(
            population,
            fitness_scores,
            gp,
            scaler_x,
            scaler_y,
            best_sample,
            args.gradient_ascent_rate,
            thresholds,
            optimal_gradient,
            args.max_iterations,
            keys_for_heuristic,
            args.heu_first,
            args.only_dual,
            assigned_fixed_keys,
            minimize_is_better,
        )
    else:
        # Perform gradient ascent using the GP
        updated_population = gradient_ascent(
            gp, scaler_x, scaler_y, best_sample, population, args.gradient_ascent_rate, thresholds
        )

    # Save the updated population as JSON
    updated_population_file = args.population_file.replace(
        "population", "updated_population"
    )
    with open(updated_population_file, "w") as f:
        f.write(json.dumps(updated_population))


if __name__ == "__main__":
    main()
