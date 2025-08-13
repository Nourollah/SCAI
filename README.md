<h1 align="center"> <code>SCAI</code> by Cardiff University </h1>

<h4 align="center">

![tests](https://github.com/Nourollah/SCAI/actions/workflows/test.yml/badge.svg?branch=main)
![documentation](https://github.com/Nourollah/SCAI/actions/workflows/deploy_docs.yml/badge.svg?branch=main)
![Static Badge](https://img.shields.io/badge/python-3.12%2B-blue)

This repository is a clone of `fairchem` developed by FAIR Chemistry. You can find documents and resources about core of this project it can be found at `Open-Catalyst-Project/ocp` and [here](https://fair-chem.github.io/).


### Installation
The main difference of this packages with original fairchem is that it is being supported by `Pixi` and it can be installed in your environment by the following:
```
pixi install
pixi run post_install
```

### Quick Start
Pretrained models can be used directly with ASE through our `OCPCalculator` interface:

```python
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import OCPCalculator

# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
    local_cache="pretrained_models",
    cpu=False,
)
slab.calc = calc

# Set up LBFGS dynamics object
dyn = LBFGS(slab)
dyn.run(0.05, 100)
```

### Docker build
To facilitate the usage of this repository we added docker to build, train, fine-tune your models on available datasets or your custom dataset. To begin with, make sure you have installed docker/podman properly. It can be happened by following:

```sh
docker build -e scai .
```

You can modify the command you need to run with docker in the `Dockerfile`. Then to run the image run following command (you can always adjust it regarding your needs):

```sh
docker run -it \
--name scai-interactive-job \
--gpus '"device=0"' \# Specify the number of GPU here if you have any
-v "$(pwd)/checkpoints:/app/checkpoints" \
-v "$(pwd)/logs:/app/logs" \
-v "$(pwd)/results:/app/results" \ 
-v "$(pwd)/configs:/app/configs" \
-v "$(pwd)/databases:/app/databases" \# Location of your data
-v "$(pwd)/wandb:/app/wandb" \# If you need to run on W&B
scai
```

### Train a model
Many configs are already available at `configs`. If you need any specify you can make one. After approaching right a config for your requirement you can easily start the train by following:

```sh
pixi run train /path/to/your/config
```

### Hyperparameters Optimization
SCAI supports `W&B` sweep to run hyperparameters optimization tasks. Refer to `W&B` documentation to learn more about.
