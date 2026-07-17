import numpy as np
import torch


def import_data(data_file: str, device: torch.device, requires_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Imports simulated data from a CSV file and returns the input features (X) and target outputs (Y) as PyTorch tensors. The function also performs sanity checks on the data.

    :param data_file: Path of the CSV file containing the simulated data.
    :param device: The PyTorch device (CPU or GPU) to which the tensors will be moved.
    :param requires_grad: If True, the input features tensor will have requires_grad set to True, allowing for gradient computation.
    :return: Tuple of PyTorch tensors (X_raw, Y_raw) where X_raw contains the input features and Y_raw contains the target outputs.
    """
    simu_data = np.genfromtxt(data_file, delimiter=",")
    X_raw = torch.tensor(simu_data[:, 0:3], dtype=torch.float32).to(device)
    Y_raw = torch.tensor(simu_data[:, 3:], dtype=torch.float32).to(device)

    assert float(Y_raw.std()) > 0.01, f"Field std={float(Y_raw.std()):.6f} — data generation (may be zero)"

    print(f"  Loaded: {data_file}")
    print(f"  X range: [{float(X_raw.min()):.3f}, {float(X_raw.max()):.3f}]")
    print(f"  Y mean:  {Y_raw.mean(dim=0).tolist()}")
    print(f"  Y std:   {Y_raw.std(dim=0).tolist()}")
    print(f"  |B| range: [{float(Y_raw.norm(dim=-1).min()):.3f}," f" {float(Y_raw.norm(dim=-1).max()):.3f}]")

    if requires_grad:
        X_raw.requires_grad_(True)
    return X_raw, Y_raw


def mean_standardized_log_loss(
    mean: np.ndarray, sigma_2: np.ndarray, test_y: np.ndarray, train_y: np.ndarray, eps=1e-6
) -> np.ndarray:
    """
    Computes the mean standardized log loss between the predicted mean and variance and the true test values, normalized by the variance of the training data.
    The MSLL is calculated as the per component log loss of the model minus the log loss of a trivial model that predicts the mean and variance of the training data.

    :param mean: Predicted mean values (tensor).
    :param sigma_2: Predicted variance values (tensor).
    :param test_y: True test values (tensor).
    :param train_y: True training values (tensor).
    :param eps: Small value to prevent division by zero (default: 1e-6).
    :return: Mean standardized log loss (scalar tensor).
    """
    # Check if the sigma 2 tensor is a variance tensor, not a standard deviation tensor, and ensure it is non-negative
    if np.any(sigma_2 < 0):
        raise ValueError("Predicted variance (sigma_2) must be non-negative.")

    loss_model = 0.5 * np.log(2 * torch.pi * sigma_2) + (test_y - mean) ** 2 / (2 * sigma_2)
    loss_model = loss_model.sum(axis=-1).mean()

    data_mean = train_y.mean(axis=0, keepdims=True)
    data_var = np.maximum(train_y.var(axis=0, keepdims=True), eps)

    loss_triv = 0.5 * np.log(2 * torch.pi * data_var) + (test_y - data_mean) ** 2 / (2 * data_var)
    loss_triv = loss_triv.sum(axis=-1).mean()

    return loss_model - loss_triv


def generate_metrics(mean, sigma_2, y_test, y_train):
    """
    Generates a dictionary of metrics including mean standardized log loss (MSLL) and root mean squared error (RMSE) based on the predicted mean and variance, as well as the true test and training values.

    :param mean: Predicted mean values (tensor).
    :param sigma_2: Predicted variance values (tensor).
    :param y_test: True test values (tensor).
    :param y_train: True training values (tensor).
    :return: Dictionary containing MSLL and RMSE metrics.
    """
    msll = mean_standardized_log_loss(mean, sigma_2, y_test, y_train)
    rmse = np.sqrt(np.mean((mean - y_test) ** 2))
    residual = np.abs(y_test - mean)
    coverage_68 = (residual < np.sqrt(sigma_2)).mean() * 100
    coverage_95 = (residual < 2 * np.sqrt(sigma_2)).mean() * 100

    print(
        f"MSLL: {msll.item():.4f}, RMSE: {rmse.item():.4f}, Coverage 68%: {coverage_68:.2f}%, Coverage 95%: {coverage_95:.2f}%"
    )

    return {"msll": msll.item(), "rmse": rmse.item(), "coverage_68": coverage_68, "coverage_95": coverage_95}
