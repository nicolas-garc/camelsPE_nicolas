# camelsPE

CAMELS parameter-estimation research workflows using PyTorch models and
simulation observables.

## Environment setup

Each clone can use the same dependency file while keeping its Python
environment outside the repository. The following commands create a dedicated
environment and register it as a Jupyter kernel:

```bash
cd ~/camelsPE_nicolas
python3 -m venv ~/.venvs/camelspe
source ~/.venvs/camelspe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name camelspe \
  --display-name "Python (camelsPE)"
```

Select `Python (camelsPE)` as the kernel when opening this repository's
notebooks. If the environment already exists, activate it and rerun the
installation command after pulling dependency changes:

```bash
source ~/.venvs/camelspe/bin/activate
cd ~/camelsPE_nicolas
python -m pip install -r requirements.txt
```

The exact clone path may differ between machines. Adjust the `cd` command, but
keep the environment separate from environments used by other repositories.

## Data location

Set the data root for the current terminal session when the mounted data path
differs between systems:

```bash
export CAMELS_DATA_DIR=/path/to/mounted/data
```

Notebook code must read `CAMELS_DATA_DIR` for this setting to take effect.
Existing notebooks still contain environment-specific paths, so this variable
documents the intended interface for the later data-loading cleanup.

## Platform note

`requirements.txt` lists direct dependencies without exact version pins. This
allows installation in both CPU-only Binder sessions and CUDA-equipped cluster
environments. On a managed GPU cluster, use the system's supported PyTorch/CUDA
installation when required by the cluster rather than replacing it blindly.
