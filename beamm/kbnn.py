import torch
import numpy as np
from tqdm import tqdm
import math


class BayesianLinear(torch.nn.Linear):
    """The Bayesian Linear class just simply adds a new buffer that stores the covariance of the weights
    and biases for the Kalman BNN training."""

    def __init__(self, in_features, out_features, bias=True, cov_init: float = None):
        super().__init__(in_features, out_features, bias)

        cov_val = cov_init if cov_init is not None else 0.5 / in_features

        self.register_buffer(
            "cov_weights",
            torch.ones((out_features, in_features + 1)) * cov_val,
        )
        torch.nn.init.normal_(self.weight, 0, math.sqrt(2.0 / in_features))
        torch.nn.init.zeros_(self.bias)


def probit(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(a / np.sqrt(2)))


def gauss_prob_pwl(mu: torch.Tensor, sigma: torch.Tensor, a: torch.Tensor = 0) -> torch.Tensor:
    gauss_a = (1 / torch.sqrt(2 * np.pi * sigma**2)) * torch.exp(-((a - mu) ** 2) / (2 * sigma**2))
    p_a = sigma**2 * gauss_a
    return p_a


# @torch.no_grad()
def forward(x_i: torch.Tensor, model: torch.nn.Module, activation: str = "relu"):
    """
    `forward` completes the forward pass (or predict pass) for the neural network.
    Here for the KalmanBNN, the backward pass uses Bayesian/Kalman update rules
    to probabilistically update the means and covariances of the weights and biases
    of the network. This is completed as a vectorized operation, so it is highly efficient.

    :param x_i: Training inputs for each
    :param model: torch.nn.Module model that include weights, biases and covariance of both in
                  each layer.
    :param activation: A string that defines the activation function. Available functions are `tanh`,
                        `relu` and `sigmoid`.
    """

    device = next(model.parameters()).device
    num_layers = len(list(model.children()))

    z_list = [x_i]
    cov_z_list = [torch.tensor([1e-8], device=device).repeat((z_list[0].shape))]
    mu_a_list = []
    sigma2_a_list = []
    lambda_ = math.sqrt(torch.pi / 8)
    for i in range(num_layers):
        weights = model[i].weight
        bias = model[i].bias
        model_cov = model[i].cov_weights
        W = torch.hstack([bias.unsqueeze(1), weights])

        sigma2_W = model_cov
        z_l = z_list[i]
        sigma2_z = cov_z_list[i]
        z_l_bias = torch.cat([torch.ones(1, device=device), z_l])
        sigma2_z_bias = (
            torch.cat([torch.zeros(1, device=device), sigma2_z]) if i > 0 else torch.zeros_like(z_l_bias, device=device)
        )

        mu_a = W @ z_l_bias
        sigma2_a = torch.clamp(
            (W**2 * sigma2_z_bias).sum(dim=1)
            + (z_l_bias**2 * sigma2_W).sum(dim=1)
            + (sigma2_W * sigma2_z_bias).sum(dim=1),
            min=1e-12,
        )
        t = torch.sqrt(1 + (lambda_**2) * sigma2_a)
        if i == num_layers - 1:
            mu_z = mu_a
            sigma2_z = sigma2_a
        elif activation == "relu":
            alpha, beta = (0, 1)
            mu_z = alpha * mu_a + (beta - alpha) * (
                mu_a * probit(mu_a / torch.sqrt(sigma2_a)) + gauss_prob_pwl(mu_a, torch.sqrt(sigma2_a))
            )

            e2 = mu_a**2 + sigma2_a
            c = beta**2 - alpha**2
            sigma2_z = (
                alpha**2 * e2
                + c * (e2 * probit(mu_a / torch.sqrt(sigma2_a)) + mu_a * gauss_prob_pwl(mu_a, torch.sqrt(sigma2_a)))
                - mu_z**2
            )
        elif activation == "sigmoid":
            mu_z = probit(lambda_ * mu_a / t)
            sigma2_z = mu_z * (1 - mu_z) * (1 - (1 / t))
        elif activation == "tanh":
            # mu_z = 2 * probit(lambda_ * mu_a / t) - 1
            # sigma2_z = (1 - mu_z**2) * (1 - (1 / t))
            mu_z = torch.tanh(mu_a)
            # 2. Use the first-order Taylor expansion for variance
            # This uses the derivative of tanh: (1 - tanh^2)
            grad_tanh = 1 - mu_z**2
            sigma2_z = (grad_tanh**2) * sigma2_a
        elif activation == "softplus":
            mu_z = torch.nn.functional.softplus(mu_a)
            # derivative of softplus is sigmoid
            grad_sp = torch.sigmoid(mu_a)
            sigma2_z = (grad_sp**2) * sigma2_a
        z_list.append(mu_z)
        cov_z_list.append(sigma2_z)
        mu_a_list.append(mu_a)
        sigma2_a_list.append(sigma2_a)

    return mu_z, sigma2_z, z_list, cov_z_list, mu_a_list, sigma2_a_list


@torch.no_grad()
def backward(
    model: torch.nn.Module,
    x_i: torch.Tensor,
    y_i: torch.Tensor,
    measurement_noise=0.1,
    activation: str = "relu",
):
    """
    `backward` completes the backward pass (or fit pass) for the neural network.
    Here for the KalmanBNN, the backward pass uses Bayesian/Kalman update rules
    to probabilistically update the means and covariances of the weights and biases
    of the network. This is completed as a vectorized operation, so it is highly efficient.

    :param model: torch.nn.Module model that include weights, biases and covariance of both in
                  each layer.
    :param x_i: Training inputs for each
    :param y_i: Description
    :param activation: Activation function: 'relu', 'tanh', or 'sigmoid'
    """

    def sigma_az(mu_a: torch.Tensor, sigma2_a: torch.Tensor, mu_z: torch.Tensor, activation: str = "relu"):
        """
        Calculates the cross covariance between the activations and the outputs
        of each layer. The return is a diagonal vector of the direct
        cross-covariances.

        :param mu_a: Mean of activations
        :param sigma2_a: Diagonal of the covariance matrix for the activations
        :param mu_z: Mean vector of the outputs of a layer
        :param activation: Activation is a string that describes the activation function to be used. Supported are
                            'relu', 'tanh' or 'sigmoid'.
        """
        if activation == "tanh":
            # Cov(a, tanh(a)) ≈ tanh'(mu_a) * Var(a) = (1 - tanh(mu_a)^2) * sigma2_a
            grad_tanh = 1 - mu_z**2  # mu_z = tanh(mu_a)
            return grad_tanh * sigma2_a

        elif activation == "softplus":
            # Cov(a, softplus(a)) ≈ sigmoid(mu_a) * sigma2_a
            return torch.sigmoid(mu_a) * sigma2_a

        elif activation == "relu":
            alpha, beta = (0, 1)

        elif activation == "linear":
            alpha, beta = (1, 1)
        sigma_a = torch.sqrt(torch.clamp(sigma2_a, min=1e-6))
        t = mu_a / sigma_a
        e2 = mu_a**2 + sigma2_a
        E_af = alpha * e2 + (beta - alpha) * (e2 * probit(t) + mu_a * gauss_prob_pwl(mu_a, sigma_a))
        return E_af - mu_a * mu_z

    device = next(model.parameters()).device
    _, _, z_list, cov_list, mu_a_list, sigma2_a_list = forward(x_i.to(device), model, activation=activation)
    # Epsilon is purely for numerical stability to prevent division by 0.
    eps = 1e-7

    num_layers = len(list(model.children()))
    mu_z_next_upd = y_i
    sigma2_z_next_upd = torch.full(y_i.shape, 0, device=device)
    for i in range(num_layers - 1, -1, -1):
        weights = model[i].weight
        bias = model[i].bias
        weights_cov = model[i].cov_weights

        # Adding bias to weights
        mu_W = torch.hstack([bias.unsqueeze(1), weights])
        sigma2_W = weights_cov.squeeze()

        mu_a = mu_a_list[i]
        sigma2_a = sigma2_a_list[i]
        mu_z_next = z_list[i + 1]
        sigma2_z_next = cov_list[i + 1] + measurement_noise if i == num_layers - 1 else cov_list[i + 1] + eps
        mu_z_curr_bias = torch.cat([torch.ones(1, device=device), z_list[i]])
        sigma2_z_curr_bias = torch.cat([torch.zeros(1, device=device), cov_list[i]])

        act_for_layer = "linear" if i == num_layers - 1 else activation
        sigma2_az_curr = sigma_az(mu_a, sigma2_a, mu_z_next, activation=act_for_layer)
        k_curr = sigma2_az_curr / sigma2_z_next

        mu_a_plus = mu_a + k_curr * (mu_z_next_upd - mu_z_next)
        sigma2_a_plus = sigma2_a + k_curr**2 * (sigma2_z_next_upd - sigma2_z_next)

        c_w_mu_z = sigma2_W * mu_z_curr_bias
        c_z_mu_w = sigma2_z_curr_bias * mu_W
        L_w = c_w_mu_z / (sigma2_a.reshape(-1, 1) + eps)
        L_z = c_z_mu_w / (sigma2_a.reshape(-1, 1) + eps)

        innovation = (mu_a_plus - mu_a).reshape(-1, 1) * 0.8
        mu_w_plus = mu_W + L_w * innovation
        mu_z_plus = mu_z_curr_bias + (L_z.T @ innovation).flatten()

        cov_innovation = (sigma2_a_plus - sigma2_a).reshape(-1, 1) * 0.8

        C_w_plus = sigma2_W + L_w**2 * cov_innovation

        C_z_plus = sigma2_z_curr_bias + (L_z.T**2 @ cov_innovation).flatten()

        model[i].weight.data.copy_(mu_w_plus[:, 1:])
        model[i].cov_weights.copy_(torch.clamp(C_w_plus, min=1e-6))
        model[i].bias.data.copy_(mu_w_plus[:, 0])
        mu_z_next_upd = mu_z_plus.squeeze()[1:]
        sigma2_z_next_upd = torch.clamp(C_z_plus.flatten()[1:], min=1e-6)


class KalmanBNN:
    """This is a class to implement the functions and passes from the paper
    'Kalman Bayesian Neural Networks for Closed-Form Online Learning' by
     Philipp Wagner, Xinyang Wu1 and Marco F. Huber.

     The KalmanBNN class wraps the forward and backward into a fit and predict
     function, and implements a custom torch.Linear layer that includes a buffer
     for the covariances of the weights.

     The fit function will compute through your given Dataset from torch.utils.data
     Dataset. This will compute the weights for one epoch. The system assumes your
     Dataset is of the form Dataset.X includes all your training inputs and Dataset.Y
     includes all your training outputs.

     The predict function will estimate the mean and variance of your output features.
    """

    def __init__(
        self,
        layer_sizes: list,
        device: str = "cpu",
        model: torch.nn.Module = None,
        cov_init=None,
    ):
        """
        Initialization for the KalmanBNN. Builds a model using the BayesianLinear layer class
        and stores the model internally. The model can also be initialized using the model
        parameter if a different description of the model is desired.

        :param self: Description
        :param layer_sizes: list of ints that include the layer size pairs ([int,int,int,int])
        :param device: Device to complete computations on. Default is 'cpu'.
        :param model: torch.nn.Module model if a custom model is desired. This model must include
                      the BayesianLinear torch.nn.Linear layers in order to store the covariance weights.
        """

        self.layer_sizes = layer_sizes
        self.device = device
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(BayesianLinear(layer_sizes[i], layer_sizes[i + 1], cov_init=cov_init))

        self.model = torch.nn.Sequential(*layers).to(device) if model is None else model

    def fit(self, dataset: torch.utils.data.Dataset, reset_cov=False):
        """
        The fit function iterates through the given dataset and runs the backwards pass on
        the training data. This is only a single epoch, if desired the fit function can be
        wrapped in a loop for multiple epochs.

        :param dataset: Dataset of type torch.utils.Dataset with training inputs defined as
                        dataset.X and outputs as dataset.Y
        :param reset_cov: Boolean to reset the covariance of the weights and biases to their initial values.
        """
        if reset_cov:
            for layer in self.model.children():
                if hasattr(layer, "cov_weights"):
                    in_features = layer.weight.shape[1]
                    layer.cov_weights.data += 1e-5

        idx = torch.randperm(len(dataset))
        shuffled = torch.utils.data.Subset(dataset, idx)
        for data in tqdm(shuffled):
            x_i, y_i = data[0], data[1]
            self._backward(
                x_i=x_i.to(self.device),
                y_i=y_i.to(self.device),
            )

    def predict(self, x_t):
        """
        The predict function runs the forward pass on the given input data and returns the mean and variance of the output features.

        :param x_t: Input tensor of shape [N, input_features] where N is the number of samples.
        :return: Tuple of (mu_z, sigma_z) where mu_z is the mean of the output features and sigma_z is the variance of the output features.
        """
        mu_z, sigma_z, _, _, _, _ = self._forward(x_t.to(self.device))
        return mu_z, sigma_z

    def _forward(self, x_t):
        return forward(x_t, self.model)

    def _backward(self, x_i, y_i):
        return backward(self.model, x_i, y_i)
