# BEAMM

Bayesian Expandable Ambient Magnetic Mapping: Continual Learning with a Physics-Constrained Bayesian Neural Network is a framework for mapping indoor ambient magnetic fields. The system implments a continual learning BNN that enables fast interpolations of magnetic fields while maintaining constraints set by Maxwells equations.

This repository houses the main network and framework for BEAMM. The code is abstracted to a library set containing BEAMM and the KalmanBNN which is a dependency. The repository also includes some examples to run and test the code for yourself along with a submoduled dataset that can be found at https://github.com/beamm-map/beamm-dataset

If you use the code please cite:

``` bibtext
@Provided Upon Acceptance
```

## Environment Setup

A `uv` environment is included to handle all the dependencies needed to run BEAMM. If you'd like to abstract BEAMM to run on your own framework, please check out the `pyproject.toml` for any dependencies you may need.

Setting up the `uv` environment is straight forward and will require the installation of [Astral UV](https://docs.astral.sh/uv/#installation). To add an install an environment you can simply do 

```bash
uv add beamm
uv sync
```

That's it!

## Running BEAMM

To run the BEAMM mapper two ways are configured. The simplest way is to run it offline (no continual learning) through the command provided below:

```python
from beamm import BEAMMMapper
offline_mapper = BEAMMMapper(device=device)

mean_offline, std_offline, _ = offline_mapper.update_and_predict(
    xd_all=X_train,
    yd_all=Y_train,
    xd_new=torch.empty((0, 3), device=device),
    yd_new=torch.empty((0, 3), device=device),
    xtd=X_test,
)
```

`xd_new` and `yd_new` are empty tensors to ensure that it doesn't think that there is new data coming in. You will have to import data and format it as the datasets with

```csv
# x,y,z,Bx,By,Bz
```

To import the datasets you can use the given `import_data` function. For example:

```python
from beamm import import_data

torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(42)

X_train, Y_train = import_data(
    "beamm-dataset/large-scale-magnetic-mapping/simu1D2D3D/simu1D_training.csv", device=device
)
X_test, Y_test = import_data("beamm-dataset/large-scale-magnetic-mapping/simu1D2D3D/simu_test.csv", device=device)
```

### Continual learning

For continual learning, instead of having the empty datasets in `xd_new` and `yd_new` you simply add the new data there, and the accumulated data in the `xd_all` and `yd_all`

```python
    mean_continual, std_continual, _ = continual_beamm.update_and_predict(
        xd_all=x_acc,
        yd_all=y_acc,
        xd_new=xd_batch,
        yd_new=yd_batch,
        xtd=X_test,
    )
```

Please checkout the examples folder for a more detailed example given there. The example folder may be updated with a ROS 2 package in the future for users to access and use.

## Citations

If you use any of the work please cite the `bibtex` entry above. If you use the large-scale-magnetic-mapping dataset please cite the orginal authors:

```bibtex
@InProceedings{abdul-raouf_large_2024,
  title = 	 {Large Scale Mapping of Indoor Magnetic Field by Local and Sparse Gaussian Processes},
  author =       {Abdul-Raouf, Iad and Gay-Bellile, Vincent and Joly, Cyril and Bourgeois, Steve and Paljic, Alexis},
  booktitle = 	 CORL,
  pages = 	 {2104--2119},
  year = 	 {2024},
  volume = 	 {270},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v270/abdul-raouf25a.html},
  abstract = 	 {Magnetometer-based indoor navigation uses variations in the magnetic field to determine the robot’s location. For that, a magnetic map of the environment has to be built beforehand from a collection of localized magnetic measurements. Existing solutions built on sparse Gaussian Process (GP) regression do not scale well to large environments, being either slow or resulting in discontinuous prediction. In this paper, we propose to model the magnetic field of large environments based on GP regression. We first modify a deterministic training conditional sparse GP by accounting for magnetic field physics to map small environments efficiently. We then scale the model on larger scenes by introducing a local expert aggregation framework. It splits the scene into subdomains, fits a local expert on each, and then aggregates expert predictions in a differentiable and probabilistic way. We evaluate our model on real and simulated data and show that we can smoothly map a three-story building in a few hundred milliseconds.}
}
```

If you use the implmentation of the Kalman-BNN provided in this python file please cite our work along with the authors of the original work give below:

```bibtex
@article{wagner_kalman_2023,
    title = {Kalman {Bayesian} {Neural} {Networks} for {Closed}-{Form} {Online} {Learning}},
    volume = {37},
    copyright = {Copyright (c) 2023 Association for the Advancement of Artificial Intelligence},
        url = {https://ojs.aaai.org/index.php/AAAI/article/view/26200},
    doi = {10.1609/aaai.v37i8.26200},
    abstract = {Compared to point estimates calculated by standard neural networks, Bayesian neural networks (BNN) provide probability distributions over the output predictions and model parameters, i.e., the weights. Training the weight distribution of a BNN, however, is more involved due to the intractability of the underlying Bayesian inference problem and thus, requires efficient approximations. In this paper, we propose a novel approach for BNN learning via closed-form Bayesian inference. For this purpose, the calculation of the predictive distribution of the output and the update of the weight distribution are treated as Bayesian filtering and smoothing problems, where the weights are modeled as Gaussian random variables. This allows closed-form expressions for training the network's parameters in a sequential/online fashion without gradient descent. We demonstrate our method on several UCI datasets and compare it to the state of the art.},
    language = {en},
    number = {8},
    urldate = {2026-02-05},
    journal = AAAI,
    author = {Wagner, Philipp and Wu, Xinyang and Huber, Marco F.},
    month = jun,
    year = {2023},
    keywords = {ML: Deep Learning Theory},
    pages = {10069--10077},
}
```
