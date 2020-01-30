## Attention Mechanism Enhanced Kernel Prediction Networks (AME-KPNs)

The official implementation of AME-KPNs in PyTorch, and our paper is accepted by ICASSP 2020 (oral), it is available at [http://arxiv.org/abs/1910.08313](http://arxiv.org/abs/1910.08313).

### News
- Support KPN (Kernel Prediction Networks), MKPN (Multi-Kernel Prediction Networks) by modifing the config file.
- The current version supports training on color images.
- The noise can be generated in a simple way as the paper descirbed, and a complex way as [Jaroensri's work](https://github.com/12dmodel/camera_sim) but replacing the Halide with OpenCV and scikit-image.

### TODO
Write the documents.

### Requirements
- Python3
- PyTorch >= 1.0.0
- Scikit-image
- Numpy
- TensorboardX (needed tensorflow support)

### How to use it?
This repo. supports training on multiple GPUs and the default setting is also multi-GPU.  

If you want to restart the train process using KPN, the command you can type as
```
CUDA_VISIBLE_DEVICES=x,x train_eval_syn.py --cuda --mGPU -nw 4 --config_file ./kpn_specs/kpn_config.conf --restart
```
If no `--restart`, the train process would be resumed.

### Citation
'''
@article{zhang2019attention,
    title={Attention Mechanism Enhanced Kernel Prediction Networks for Denoising of Burst Images},
    author={Bin Zhang and Shenyao Jin and Yili Xia and Yongming Huang and Zixiang Xiong},
    year={2019},
    journal={arXiv preprint arXiv:1910.08313}
}
'''