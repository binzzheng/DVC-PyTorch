# DVC-PyTorch

A PyTorch reimplemetation for the paper:

DVC: An End-to-end Deep Video Compression Framework, Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, Zhiyong Gao, CVPR 2019 (**Oral**). [[arXiv]](https://arxiv.org/abs/1812.00101)

## Requirements

- Python==3.6.2
- PyTorch==1.7.1
- CompressAI==1.1.6

## Training Data Preparation

Vimeo90k is used for training, you can download it online and put the video sequence in `./data/vimeo90k/sequences`.

## Train

You can simply perform the most basic training with the following commands 

```
python train_dvc.py --config DVC_1024.json
```

If you want to change the training settings, you can modify the corresponding json file.

## Testing Data Preparation
