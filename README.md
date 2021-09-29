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
python main_dvc.py --config DVC_1024.json
```

If you want to change the training settings, you can modify the corresponding json file.

## Testing Data Preparation

The HEVC test sequence is used to test the video compression performance, you can download it and put it in `./data/h265_compress/videos/`. The test process here is based on the class B in the HEVC test sequence. You can test other common datasets such as UVG in a similar way. 

The video processed by DVC has a resolution limit, and here you need to manually crop the video to a multiple of 64. Take the class B in the hevc test sequence as an example, you need to crop 1920x1080 to 1920x1024.

```
cd data/h265_compress/
ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/xxxx.yuv -vf crop=1920:1024:0:0 ./videos_crop/xxxx.yuv
```

Split the yuv video file into frames. 
```
python convert.py
```

Create I frames. We need to create I frames by H.265 with QP of 22,27,32,37. The setting of the compressed I frame affects the compression performance. According to this setting (set QP as 22,27,32,37), the results obtained when testing is consistent with the DVC paper. 

```
cd CreateI
sh h265.sh $qp 1920 1024
```
After finished the generating of I frames of each QP, you need to use bpps of each video in `result.txt` to fill the bpps in Class `HEVCDataSet` in `dataset.py`.

## Test

Set the path of the trained model in the `main_dvc.py` file, and enable `args.hevctest`. 

```
python main_dvc.py --config test_1024.json
```

## Improve on the basis of DVC 

You can edit the `net_dvc.py` file. The specific components that build it are in the `subnet` folder. Only `forward` is executed during training, and `compress` and `decompress` need to be executed during testing. Make sure that the two parts correspond. 


## Citation
If you find this paper useful, please cite:
```
@article{lu2018dvc,
  title={DVC: An End-to-end Deep Video Compression Framework},
  author={Lu, Guo and Ouyang, Wanli and Xu, Dong and Zhang, Xiaoyun and Cai, Chunlei and Gao, Zhiyong},
  journal={arXiv preprint arXiv:1812.00101},
  year={2018}
}
```

## Acknowledement
Most of the code comes from [[PyTorchVideoCompression]](https://github.com/binzzheng/PyTorchVideoCompression/tree/master/DVC). Thanks to [[ZhihaoHu]](https://github.com/ZhihaoHu) for the open source code. I replaced the entropy model used in it with the `EntropyBottleneck` class in [[CompressAI]](https://github.com/InterDigitalInc/CompressAI), and performed the actual compression and decompression during testing. 
