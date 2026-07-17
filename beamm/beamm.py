import os
from .kbnn import BayesianLinear, backward
import numpy as np
import hamiltorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import tomllib

local_config = "config.toml"

if os.path.exists(local_config):
    warnings.warn(f"WARNING: using local config.toml at {local_config}")
    config_path = local_config
else:
    warnings.warn(f"WARNING: {local_config} not found, using package config.toml")
    package_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(package_dir, "config.toml")

with open(config_path, "rb") as file:
    config = tomllib.load(file)

KBNN_NOISE_MULT = config["params"]["kbnn_noise_mult"]
PHI_NOISE_FLOOR = config["params"]["phi_noise_floor"]
KBNN_EPOCHS = config["params"]["kbnn_epochs"]
KBNN_SUBSAMPLE = config["params"]["kbnn_subsample"]

SIGMA_HMC_BEAMM = config["params"]["sigma_hmc_beamm"]
PRIOR_VAR_SCALE = config["params"]["prior_var_scale"]
PRIOR_VAR_FLOOR = 0.2 * SIGMA_HMC_BEAMM
SIGMA_1D = config["params"]["sigma_1d"]

HIDDEN = config["params"]["hidden"]
ADAM_LR = config["params"]["adam_lr"]
ADAM_STEPS = config["params"]["adam_steps"]

NUTS_SAMPLES_BEAMM = config["params"]["nuts_samples_beamm"]
NUTS_BURN_BEAMM = config["params"]["nuts_burn_beamm"]
NUTS_STEP_BEAMM = config["params"]["nuts_step_beamm"]
NUTS_STEP_REFRESH = config["params"]["nuts_step_refresh"]
NUTS_STEPS_PER_SAMPLE_BEAMM = config["params"]["nuts_steps_per_sample_beamm"]
NUTS_ACCEPT_BEAMM = config["params"]["nuts_accept_beamm"]

N_EFF_THRESHOLD = config["params"]["n_eff_threshold"]


class ScalarPotNet(nn.Module):
    """
    A simple MLP to represent the scalar potential phi(x,y,z) in 3D space. The magnetic field B is computed as the negative gradient of phi.
    """

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def _prepare_local_targets(
    X: torch.Tensor, Y: torch.Tensor, sigma: float, noise_floor: float, noise_mult: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment delta-phi targets and per-segment noise

    :param X: [N,3]  (x,y,z) positions
    :param Y: [N,3]  (Bx,By,Bz) magnetic field vectors
    :param sigma: float, measurement noise stddev for B
    :param noise_floor: float, minimum delta-phi noise
    :param noise_mult: float, multiplier for delta-phi noise

    :return X_s: [M,3] sorted positions (segment endpoints)
    :return d_phi: [M-1] delta-phi values for each segment
    :return d_phi_noise: [M-1] noise variance for each segment
    """
    sort_idx = X[:, 0].argsort()
    X_s = X[sort_idx]
    Y_s = Y[sort_idx]
    dx = X_s[1:] - X_s[:-1]
    B_avg = 0.5 * (Y_s[1:] + Y_s[:-1])
    d_phi = torch.sum(B_avg * dx, dim=1)
    d_phi_noise = (torch.norm(dx, dim=1) * 0.5) ** 2 * noise_mult * sigma**2 + noise_floor
    return X_s, d_phi, d_phi_noise


def _kbnn_diag_var_flat(kbnn_model) -> torch.Tensor:
    """
    Flatten KBNN diagonal variance in ScalarPotNet parameter order: [W1,b1,W2,b2,W3,b3].
    BayesianLinear.cov_weights shape: [out, in+1]  (col 0 = bias, cols 1: = weights)

    :param kbnn_model: nn.Sequential of BayesianLinear layers
    :return flat_var: [num_params] flattened variance vector
    """
    var_parts = []
    for layer in kbnn_model:
        if not isinstance(layer, BayesianLinear):
            continue
        cov = layer.cov_weights.detach()
        var_parts.append(cov[:, 1:].reshape(-1))  # weight var
        var_parts.append(cov[:, 0].reshape(-1))  # bias var
    return torch.cat(var_parts)


def _load_params(model: nn.Module, params: torch.Tensor) -> None:
    """
    Assign a flat parameter vector to the model's parameters in order. The input params is a 1D tensor containing
    all parameters concatenated.

    :param model: nn.Module whose parameters will be updated
    :param params: 1D tensor of parameters to load into the model
    """
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(params[offset : offset + n].view(p.shape))
        offset += n


@torch.enable_grad()
def _grad_phi(model, x: torch.Tensor) -> torch.Tensor:
    """
    Gradient of the scalar potential phi with respect to input x, which gives the magnetic field B = -grad(phi).

    :param model: ScalarPotNet model
    :param x: [N,3] input positions
    :return B: [N,3] magnetic field vectors at positions x
    """
    x_req = x.detach().requires_grad_(True)
    phi = model(x_req)
    B = -torch.autograd.grad(phi, x_req, grad_outputs=torch.ones_like(phi))[0]
    return B.detach()


def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    :return: Compute the L2 norm squared between two tensors a and b.
    """
    return ((a - b) ** 2).sum()


class BEAMMMapper:
    """
    Bayesian Expandable Ambient Magnetic Mapping (BEAMM) class for mapping magnetic fields using a Bayesian approach.
    It maintains a set of particles representing the posterior distribution over model parameters, updates the prior
    using observed data, and performs importance reweighting and resampling to refine the particle set. The class
    also provides methods for making predictions at new locations based on the current particle set.
    """

    def __init__(self, device: torch.device, logger=None) -> None:
        self.device = device
        self.logger = logger
        self.model = ScalarPotNet(HIDDEN).to(device)
        self.particles = None
        self.log_weights = None
        self.prior_mean = None
        self.prior_var = None

    def update_prior(self, xd_all: torch.Tensor, yd_all: torch.Tensor):
        """
        Update the prior distribution of the model parameters using the provided training data (xd_all, yd_all).
        This method performs an Adam optimization to find the MAP estimate of the model parameters and then uses
        a KBNN to estimate the prior variance of the parameters based on the training data.

        :param xd_all: [N,3] tensor of input positions (x,y,z)
        :param yd_all: [N,3] tensor of observed magnetic field vectors (Bx,By,Bz)

        :return: Updates self.prior_mean and self.prior_var with the estimated mean and variance of the model parameters.

        """

        opt = torch.optim.Adam(self.model.parameters(), lr=ADAM_LR)

        for _ in range(ADAM_STEPS):
            opt.zero_grad()
            x_req = xd_all.detach().requires_grad_(True)
            phi = self.model(x_req)
            B = -torch.autograd.grad(phi, x_req, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
            loss = ((B - yd_all) ** 2).mean() / (2 * SIGMA_1D)
            loss.backward()
            opt.step()

        self.prior_mean = torch.cat([p.detach().flatten() for p in self.model.parameters()])

        kbnn = nn.Sequential(
            BayesianLinear(3, HIDDEN),
            BayesianLinear(HIDDEN, HIDDEN),
            BayesianLinear(HIDDEN, 1),
        ).to(self.device)

        for kl, dl in zip(kbnn, [self.model.net[0], self.model.net[2], self.model.net[4]]):
            kl.weight.data.copy_(dl.weight.data)
            kl.bias.data.copy_(dl.bias.data)

        X_s, d_phi, d_phi_noise = _prepare_local_targets(
            xd_all,
            yd_all,
            sigma=SIGMA_1D,
            noise_floor=PHI_NOISE_FLOOR,
            noise_mult=KBNN_NOISE_MULT,
            device=self.device,
        )
        num_segments = X_s.shape[0] - 1
        n_sub = min(KBNN_SUBSAMPLE, num_segments)

        for ep in range(KBNN_EPOCHS):
            order = torch.randperm(num_segments, device=self.device)[:n_sub]
            for k in order:
                backward(
                    kbnn,
                    X_s[k + 1],  # single input point
                    d_phi[k : k + 1],  # scalar delta-phi target
                    measurement_noise=float(d_phi_noise[k]),
                    activation="softplus",
                )

        raw_var = _kbnn_diag_var_flat(kbnn)
        self.prior_var = (raw_var * PRIOR_VAR_SCALE).clamp_min(PRIOR_VAR_FLOOR)

    def importance_reweight(self, xd_new: torch.Tensor, yd_new: torch.Tensor):
        """
        The SMC importance reweighting step. Given new data (xd_new, yd_new), this method updates
        the log weights of the particles based on the likelihood of the new data under each particle's
        parameters. If a particle's parameters lead to NaN or infinite likelihood, its log weight is
        set to a large negative value to effectively remove it from consideration.

        :param xd_new: [M,3] tensor of new input positions (x,y,z)
        :param yd_new: [M,3] tensor of new observed magnetic field vectors (Bx,By,Bz)


        :return: Updates self.log_weights with the new log weights for each particle.
        """
        if self.particles is None:
            return

        model_eval = ScalarPotNet(HIDDEN).to(self.device)

        for i, p in enumerate(self.particles):
            if torch.isnan(p).any():
                self.log_weights[i] = -50.0
                continue

            _load_params(model_eval, p)
            B = _grad_phi(model_eval, xd_new)
            err = _l2(B, yd_new)

            if torch.isnan(err) or torch.isinf(err):
                self.log_weights[i] = -50.0
                continue

            log_lik = -err.clamp(max=1e4) / (2 * SIGMA_HMC_BEAMM)
            self.log_weights[i] += log_lik

        self.log_weights -= self.log_weights.max()
        self.log_weights = torch.nan_to_num(self.log_weights, nan=-50.0)
        self.log_weights = torch.clamp(self.log_weights, min=-50.0)

    def compute_neff(self) -> float:
        """
        Computes the effective sample size (N_eff) of the particle set based on the current log weights.
        :return N_eff: float, the effective number of particles, calculated as 1 / sum(w^2)
        """
        if self.log_weights is None:
            return 0.0
        w = torch.softmax(self.log_weights, dim=0)
        return (1.0 / (w.pow(2).sum() + 1e-12)).item()

    def resample(self):
        """
        Resamples the particles based on their current log weights. This method uses multinomial resampling to
        select particles according to their weights, effectively focusing on the more probable particles and
        discarding those with low weights. After resampling, the log weights are reset to zero for all particles.
        """

        w = torch.softmax(self.log_weights, dim=0)
        idx = torch.multinomial(w, len(w), replacement=True)
        self.particles = self.particles[idx]
        self.log_weights = torch.zeros_like(self.log_weights)

    def _make_neg_log_post(self, xd_all: torch.Tensor, yd_all: torch.Tensor) -> callable:
        """Shared closure for cold_start and nuts_refresh. Returns a function that computes the negative log
        posterior for given parameters.

        :param xd_all: [N,3] tensor of input positions (x,y,z)
        :param yd_all: [N,3] tensor of observed magnetic field vectors (Bx,By,By)
        :return: function that computes the negative log posterior for given parameters
        """
        shapes = [p.shape for p in self.model.parameters()]
        sizes = [p.numel() for p in self.model.parameters()]
        x_fixed = xd_all.detach().clone()
        y_fixed = yd_all.detach().clone()
        prior_mean_snap = self.prior_mean.detach().clone()
        prior_var_snap = self.prior_var.detach().clone()

        def neg_log_post(params):
            """
            Compute the negative log posterior for the given flat parameter vector.
            """
            if torch.isnan(params).any():
                return torch.tensor(float("inf"), device=self.device, requires_grad=True)

            parts = torch.split(params, sizes)
            rebuilt = [p.view(s) for p, s in zip(parts, shapes)]

            x_inp = x_fixed.clone().requires_grad_(True)
            h = F.softplus(F.linear(x_inp, rebuilt[0], rebuilt[1]))
            h = F.softplus(F.linear(h, rebuilt[2], rebuilt[3]))
            phi = F.linear(h, rebuilt[4], rebuilt[5])

            B = -torch.autograd.grad(phi, x_inp, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

            err = ((B - y_fixed) ** 2).clamp(max=1e4)
            data_term = err.sum() / (2 * SIGMA_HMC_BEAMM)
            prior_term = 0.5 * (((params - prior_mean_snap) ** 2) / (prior_var_snap + 1e-6)).sum()

            return data_term + prior_term

        return neg_log_post

    def nuts_refresh(self, xd_all: torch.Tensor, yd_all: torch.Tensor):
        """
        The refresh of the weighted posterior in the case of particle degeneracy. This method uses the No-U-Turn Sampler (NUTS)
        to generate new particles from the posterior distribution defined by the current data (xd_all, yd_all) and the prior.
        It initializes the sampler using the mean and spread of the current particles, runs the NUTS algorithm to generate new
        samples, and replaces the current particles with the new ones.

        :param xd_all: [N,3] tensor of input positions (x,y,z)
        :param yd_all: [N,3] tensor of observed magnetic field vectors (Bx,By,Bz)

        :return: Updates self.particles with the new samples from the posterior distribution.
        """
        if self.particles is None:
            return
        self.update_prior(xd_all, yd_all)
        TARGET_N = NUTS_SAMPLES_BEAMM - NUTS_BURN_BEAMM
        neg_log_post = self._make_neg_log_post(xd_all, yd_all)

        base = self.particles.mean(0)
        spread = self.particles.std(0).clamp(min=1e-3)
        init = (base + spread * torch.randn_like(base)).requires_grad_(True)

        burn = NUTS_BURN_BEAMM
        samples = hamiltorch.sample(
            log_prob_func=lambda p: -neg_log_post(p),
            params_init=init,
            num_samples=TARGET_N + burn,
            step_size=NUTS_STEP_REFRESH,
            num_steps_per_sample=NUTS_STEPS_PER_SAMPLE_BEAMM,
            burn=burn,
            sampler=hamiltorch.Sampler.HMC_NUTS,
            desired_accept_rate=NUTS_ACCEPT_BEAMM,
            integrator=hamiltorch.Integrator.IMPLICIT,
        )

        new_particles = torch.stack(samples[burn:]).detach()
        valid = ~torch.isnan(new_particles).any(dim=1)

        if valid.sum() < 5:
            warnings.warn("WARNING: nuts_refresh too few valid particles — keeping old")
            return
        clean = new_particles[valid]

        if len(clean) < TARGET_N:
            pad_idx = torch.randint(0, len(clean), (TARGET_N - len(clean),))
            clean = torch.cat([clean, clean[pad_idx]], dim=0)

        self.particles = clean
        self.log_weights = torch.zeros(len(self.particles), device=self.device)

    def cold_start(self, xd_all: torch.Tensor, yd_all: torch.Tensor):
        """
        Cold start the particle set using NUTS sampling from the posterior defined by the current data (xd_all, yd_all) and the prior.
        This method initializes the sampler using the prior mean, runs the NUTS algorithm to generate new samples, and sets the particles to the new samples. If the sampling fails to produce enough valid
        samples, it falls back to using the prior mean with added noise.

        :param xd_all: [N,3] tensor of input positions (x,y,z)
        :param yd_all: [N,3] tensor of observed magnetic field vectors (Bx,By,Bz)

        :return: Updates self.particles with the new samples from the posterior distribution.
        """
        assert self.prior_mean is not None, "update_prior() must run before cold_start()"

        neg_log_post = self._make_neg_log_post(xd_all, yd_all)

        init = self.prior_mean.clone().detach().requires_grad_(True)

        samples = hamiltorch.sample(
            log_prob_func=lambda p: -neg_log_post(p),
            params_init=init,
            num_samples=NUTS_SAMPLES_BEAMM,
            step_size=NUTS_STEP_BEAMM,
            num_steps_per_sample=NUTS_STEPS_PER_SAMPLE_BEAMM,
            burn=NUTS_BURN_BEAMM,
            sampler=hamiltorch.Sampler.HMC_NUTS,
            desired_accept_rate=NUTS_ACCEPT_BEAMM,
            integrator=hamiltorch.Integrator.IMPLICIT,
        )

        TARGET_N = NUTS_SAMPLES_BEAMM - NUTS_BURN_BEAMM

        clean = torch.stack(samples[NUTS_BURN_BEAMM:]).detach()
        valid = ~torch.isnan(clean).any(dim=1)

        if valid.sum() < 5:
            warnings.warn("WARNING: cold start fallback to prior mean + noise")
            clean = self.prior_mean.unsqueeze(0) + 0.01 * torch.randn(
                TARGET_N, self.prior_mean.numel(), device=self.device
            )
        else:
            clean = clean[valid]
            if len(clean) < TARGET_N:
                pad_idx = torch.randint(0, len(clean), (TARGET_N - len(clean),))
                clean = torch.cat([clean, clean[pad_idx]], dim=0)

        self.particles = clean
        self.log_weights = torch.zeros(len(clean), device=self.device)

    def update_and_predict(
        self, xd_all: torch.Tensor, yd_all: torch.Tensor, xd_new: torch.Tensor, yd_new: torch.Tensor, xtd: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Update the particle set with new data and make predictions at new locations.

        **NOTE**: The method returns the mean and std of the magnetic field as numpy arrays,
        not torch tensors. This is to avoid issues with GPU memory management and to ensure
        compatibility with downstream code that may expect numpy arrays.

        :param xd_all: [N,3] tensor of all input positions (x,y,z) seen so far
        :param yd_all: [N,3] tensor of all observed magnetic field vectors (Bx,By,Bz) seen so far
        :param xd_new: [M,3] tensor of new input positions (x,y,z) to update the model with
        :param yd_new: [M,3] tensor of new observed magnetic field vectors (Bx,By,Bz) to update the model with
        :param xtd: [K,3] tensor of target positions (x,y,z) where predictions are desired

        :return mean: [K,3] array of predicted mean magnetic field vectors at xtd
        :return std: [K,3] array of predicted standard deviation of magnetic field vectors at xtd
        :return neff: float, effective number of particles after the update
        """
        if self.particles is None:
            warnings.warn("BEAMM cold start")
            self.update_prior(xd_all, yd_all)
            self.cold_start(xd_all, yd_all)

            neff = self.particles.shape[0]

        else:
            if xd_new.shape[0] > 0:
                # Reset weights to score only this batch's incremental likelihood
                self.log_weights = torch.zeros_like(self.log_weights)
                self.importance_reweight(xd_new, yd_new)

            neff = self.compute_neff()

            if neff < N_EFF_THRESHOLD * len(self.particles):
                self.resample()
                self.nuts_refresh(xd_all, yd_all)

        w = torch.softmax(self.log_weights, dim=0)
        model_eval = ScalarPotNet(HIDDEN).to(self.device)

        preds = []
        for p in self.particles:
            _load_params(model_eval, p)
            preds.append(_grad_phi(model_eval, xtd))

        preds = torch.stack(preds)
        mean = (w.view(-1, 1, 1) * preds).sum(0)
        var = (w.view(-1, 1, 1) * (preds - mean.unsqueeze(0)) ** 2).sum(0)

        std = (var).sqrt()

        return mean.cpu().numpy(), std.cpu().numpy(), neff
