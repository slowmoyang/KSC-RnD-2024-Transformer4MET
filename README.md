# Differential MET Reconstruction

## Updates at Here!
[Google Docs](https://docs.google.com/presentation/d/1NfmIcNPUrPn3dtbH4TU6n-ewqgTCQL2fbUM9HTdGMfM/edit#slide=id.p)

## Recipes
### Install
#### Micromamba
* The following command will ask you a few questions.
* `Prefix location` must be located under `/scratch/$USER` because the limit on the number of files in the home directory is only 100k

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
copied from [Micromamba Installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install)

#### Python environment
You can install python packages with `environment.yaml`
```bash
micromamba create -y -f environment.yaml
```

### Setup
let's start setup!
```bash
source setup.sh
```
### Run a training code on a local machine
You can then train a deep learning model with `train.py` and a configuration .yaml file.
For example, you can a Transformer-based network that takes in L1 ParticleFlow candidates and reconstruct MET.
```bash
python train.py -c ./config/test-l1pf-transformer-neuron.yaml
```

If you want to understand how to configure it, I recommend reading [Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)

### Submit a batch job with HTCondor
Before submitting a job, u need to check available memory.
```bash
condor_status -compact -constraint 'Gpus > 0'
```

You can submit a training job to a GPU cluster using `submit.py`
```bash
submit.py --config ./config/only-high-met.yaml --data ./config/data/delphes/knu/sanity-check.yaml --trainer ./config/trainer/sanity-check.yaml -e sanity-check -m 50GB
```

```bash
condor_q $USER
```
