# [Moreh] Running SimMIM on Moreh AI Framework
![](https://badgen.net/badge/Moreh-HAC/failed/red) ![](https://badgen.net/badge/Nvidia-A100/passed/green)

## Prepare

### Data
For testing purpose, we use `imagenet_100cls`, a subset of ImageNet with 100 classes.
Get the dataset from [here](http://ref.deploy.kt-epc.moreh.io:8080/reference/dataset/imagenet_100cls.tar.gz)
and extract it. The structure of the dataset is already compatible.

### Code
```bash
git clone https://github.com/loctxmoreh/SimMIM
cd SimMIM
```

#### Pretrained models
Get these two pretrained models:
[`simmim_finetune__swin_base__img224_window7__800ep.pth`](https://drive.google.com/file/d/1xEKyfMTsdh6TfnYhk5vbw0Yz7a-viZ0w/view)
and
[`simmim_pretrain__swin_base__img192_window6__800ep.pth`](https://drive.google.com/file/d/15zENvGjHlM71uKQ3d2FbljWPubtrPtjl/view)
and put them in the root of SimMIM repo, for later runs.


### Environment
First, create a `conda` environment:
```bash
conda create -n simmim python=3.8 -y
conda activate simmim
```

#### Install `torch`

##### On A100 machine
With `torch==1.7.1` and CUDA 11.0:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f
https://download.pytorch.org/whl/torch_stable.html
```

With `torch==1.12.1` and CUDA 11.3:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

##### On HAC machine
Install `torch=1.7.1`:
```bash
conda install -y torchvision torchaudio numpy protobuf==3.13.0 pytorch==1.7.1
cpuonly -c pytorch
```
Then, force update Moreh framework (version 22.10.2 at the time of writing)
```bash
update-moreh --force --version 22.10.2
```

#### Install `apex`
Still inside the conda env, clone the `apex` repo:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

Then, install this one missing dependency for `apex`:
```bash
pip install packaging
```

##### Install in full mode with CUDA
To be able to install in full mode, `apex` requires the *exact* same version of
CUDA on the machine and the CUDA version `torch` is compiled with. On A100 VM,
this is possible with `torch==1.7.1+cu110` and CUDA version 11.0.

```bash
# explicitly pointing CUDA_HOME to CUDA 11.0,
# since normally, CUDA_HOME point to CUDA 11.2 on A100 VM
CUDA_HOME=/usr/local/cuda-11.0 pip install -v --disable-pipp-version-check
--no-cache-dir --global-option="--cpp_ext" --global-optionion="--cuda_ext" ./
```

##### Install in python-only mode
With `torch=1.12.1` and CUDA 11.3, or `torch==1.7.1` with Moreh solution,
`apex` can only be installed in python-only mode
```bash
pip install -v --disable-pip-version-check --no-cache-dir ./
```

#### The rest of requirements
```bash
pip install -r requirements.txt
```

## Run
Before running the following scripts:
- Edit them and change the `--data-path` parameter to point to
  `imagenet_100cls` dataset.
- Edit the corresponding `.yml` config file and change `TRAIN.EPOCHS` to a
  small value

### Evaluate existing models
Run the `evaluate-*` scripts to evaluate the pretrained
`simmim_finetune__swin_base__img224_window7__800ep.pth` on `imagenet_100cls`
dataset:

```bash
# On HAC machine:
./evaluate-hac

# On A100 machine:
./evaluate-a100
```

### Pretrain model with SimMIM
Run the `pretrain-*` scripts to pretrain a
`simmim_pretrain__swin_base__img192_window6__800ep.yaml` model on
`imagenet_100cls` dataset:

```bash
# On HAC machine:
./pretrain-hac

# On A100 machine:
./pretrain-a100
```

### Finetune existing models
Run the `finetune-*` scripts to finetune the pretrained
`simmim_finetune__swin_base__img224_window7__800ep.pth` model on
`imagenet_100cls` dataset:


```bash
# On HAC machine:
./finetune-hac

# On A100 machine:
./finetune-a100
```
