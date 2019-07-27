## Kernel Prediction  Networks and Multi-Kernel Prediction Networks
Reimplement of [Burst Denoising with Kernel Prediction Networks](https://arxiv.org/pdf/1712.02327.pdf) and [Multi-Kernel Prediction Networks for Denoising of Image Burst](https://arxiv.org/pdf/1902.05392.pdf) by using PyTorch.

The partial work is following [https://github.com/12dmodel/camera_sim](https://github.com/12dmodel/camera_sim).

## TODO
Write the documents.

## Requirements
- Python3
- PyTorch >= 1.0.0
- Scikit-image
- Numpy
- TensorboardX (needed tensorflow support)

## How to use it?
This repo. supports training on multiple GPUs and the default setting is also multi-GPU.  

If you want to restart the train process using KPN, the command you can type as
```
CUDA_VISIBLE_DEVICES=x,x train_eval_syn.py --cuda --mGPU -nw 4 --config_file ./kpn_specs/kpn_config.conf --restart
```
If no `--restart`, the train process would be resumed.
