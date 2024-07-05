<div id="top"></div>

# VIT-VQGAN

This is an unofficial implementation of both [ViT-VQGAN](https://arxiv.org/abs/2110.04627) and [RQ-VAE](https://arxiv.org/abs/2110.04627) in Pytorch. ViT-VQGAN is a simple ViT-based Vector Quantized AutoEncoder while RQ-VAE introduces a new residual quantization scheme. Further details can be viewed in the papers

![](https://raw.githubusercontent.com/henrywoo/vim/main/vitvqgan.png)


## Installation

```python
pip install vitvqgan 
```


## Training

**Stage 1 - VQ Training:**
```
python -m vitvqgan.train_vim
```

You can add more options too:

```python
python -m vitvqgan.train_vim -c imagenet_vitvq_small -lr 0.00001 -e 10
```

It uses `Imagenette` as the training dataset for demo purpose, to change it, modify [dataloader init file](vitvqgan/dataloader/__init__.py).

**Inference:**
- download checkpoints from above in mbin folder
- Run the following command:
```
python -m vitvqgan.demo_recon
```

This code has cusomtized cuda code, so it need gcc to support C++17.

If you don't have latest gcc, please consider the following commands:

```
conda install ninja
conda install -c conda-forge gcc_linux-64 gxx_linux-64
```

## Checkpoints

- [ViT-VQGAN Small](https://drive.google.com/file/d/1jbjD4q0iJpXrRMVSYJRIvM_94AxA1EqJ/view?usp=sharing) 
- [ViT-VQGAN Base](https://drive.google.com/file/d/1syv0t3nAJ-bETFgFpztw9cPXghanUaM6/view?usp=sharing)


## Acknowledgements

The repo is modified from [here](https://github.com/thuanz123/enhancing-transformers) with updates to latest dependencies and to be easily run in consumer-grade GPU for learning purpose.

