# [TGRS'2026] Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion

> **Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion** [[Paper]](https://ieeexplore.ieee.org/document/11471838)
>
> Nan Xiao, Xiyou Fu†, [Qi Ren](https://github.com/renqi1998), [Wangquan He](https://github.com/Hewq77), Siqi Wei, [Sen Jia](https://scholar.google.com/citations?user=UxbDMKoAAAAJ&hl=zh-CN&oi=ao)
>
> College of Computer Science and Software Engineering, Shenzhen University

## ⚙️ Environment
```
conda create -n RAMoE python=3.9.19
conda activate RAMoE
pip install -r requirements.txt
```
## 🛫 Usage
1.  **Datasets setting**
   - Download  Datasets : [WDCM](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html) / [Chikusei](https://naotoyokoya.com/Download.html) / [Xiongan](https://github.com/NIM-NMDC/PSTUN)

2.  **Training**
 ```
 CUDA_VISIBLE_DEVICES=0 
 python train.py  \
```

3.  **Inference**
```
set train_test in train.py as 0 \
python train.py  \
```

## 🎓 Citations
Please cite us if our work is useful for your research.
```
@ARTICLE{Xiao2026RAMoE,
  author={Xiao, Nan and Fu, Xiyou and Ren, Qi and He, Wangquan and Wei, Siqi and Jia, Sen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
}

