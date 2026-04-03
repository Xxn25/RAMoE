# @Time    : 2025/10/16 17:02
# @Author  : Nan Xiao
# @File    : v4_8_1_36_ywX4.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from STM_standard_train_model import STM
from RAMoE import RAMoEN

# ================== Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True
cudnn.deterministic = True


if __name__ == '__main__':
    lr = 2e-4
    strat_epoch = 1
    patch_size = 64
    epochs = 500
    ckpt_step = 25
    batch_size = 8
    net_type = 'RAMoEN'
    upscale_factor = 4
    stride = 64
    dataset = 'WDCM'
    dim = 96
    deepth = 3
    val = False
    train_test = 1
    load_epoch = 0
    num_experts = 8
    num_shared = 1
    dim_moe = 36
    value_case = 'sum'
    ffn_mode = 'RAMoE'
    residual_type = '3conv'
    act_func = 'gelu'
    save_dir = os.path.join("{}_{}".format(net_type,dataset))

    model = RAMoEN(hsi_channels=110, msi_channels=4, upscale_factor=upscale_factor, dim_moe=dim_moe, num_experts=num_experts, num_shared=num_shared,
                    residual_type=residual_type, n_feats=dim, deepth=deepth, value_case=value_case, act_func=act_func, ffn_mode=ffn_mode).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = MultiStepLR(optimizer, milestones=[150, 200, 250, 300, 350, 400, 450, 500], gamma=0.5)
    train_loss = nn.L1Loss().cuda()
    stm = STM(save_dir=save_dir, epoch=strat_epoch, dataset=dataset, model=model, patch_size=patch_size, batch_size=batch_size,
              scale=upscale_factor, train_loss=train_loss, optimizer=optimizer, scheduler=lr_scheduler, val=val, num_workers=4)
    if train_test == 1:
        for epoch in range(stm.epoch, epochs+1):
            stm.train_one_epoch(epoch)
            if epoch % ckpt_step == 0:
                stm.save(epoch)
                stm.test_per_epoch(epoch)
        print("train successful!")
    else:
        stm.load(load_epoch)
        stm.test_per_epoch(load_epoch, save=False)