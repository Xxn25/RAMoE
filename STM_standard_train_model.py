import os.path
import torch
from STM_writer import STMWriter
from Process_data import ProcessData
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.io import savemat
from IQA.hsi_metrics import *
from torch.autograd import Variable


class STM:
    def __init__(self, save_dir, epoch, dataset, model, patch_size, batch_size, scale, train_loss, optimizer, scheduler, val, num_workers=1):
        self.epoch = epoch
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_loss = train_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.scale = scale
        self.writer = STMWriter(save_dir)
        train_set = ProcessData(os.path.join('img_sets', dataset, 'train'), train=True, patch_size=patch_size, scale=scale)
        self.train_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True)
        if val:
            val_set = ProcessData(os.path.join('img_sets', dataset, 'val'), train=False, patch_size=patch_size, scale=scale)
            self.val_data_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1,
                                              shuffle=False, pin_memory=True, drop_last=False)

        test_set = ProcessData(os.path.join('img_sets', dataset, 'test'), train=False, patch_size=patch_size, scale=scale)
        self.test_data_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1,
                                           shuffle=False, pin_memory=True, drop_last=False)
        print("STM_stand_train_model is Successfully initialized")

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, resume):
        path_checkpoint = os.path.join(self.save_dir, 'save_model','epoch_{}.pth'.format(resume))
        checkpoint = torch.load(path_checkpoint)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']+1
        print('Network is Successfully Loaded from {} \n'.format(path_checkpoint))

    # ----------------------------------------
    # save model / optimizer / epoch / lr
    # ----------------------------------------
    def save(self, epoch):
        save_dir = os.path.join(self.save_dir, 'save_model')
        model_out_path = os.path.join(save_dir,'epoch_{}.pth'.format(epoch))
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch,
        }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(checkpoint, model_out_path)


    def train_one_epoch(self, epoch):
        start = datetime.now()
        epoch_train_loss = []
        self.model.train()
        for i, batch in enumerate(self.train_data_loader, 1):
            GT, LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            self.optimizer.zero_grad()  # fixed
            output_HRHSI = self.model(LRHSI, HRMSI)
            net_loss = self.train_loss(output_HRHSI, GT)
            epoch_train_loss.append(net_loss.item())
            net_loss.backward()
            self.optimizer.step()
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, i, len(self.train_data_loader), net_loss.item()))
        end = datetime.now()
        print("learning rate:º%f" % (self.optimizer.param_groups[0]['lr']))
        self.scheduler.step()  # update lr
        epoch_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        train_msg = "{}--> epoch={} loss={:.8f} lr={} time={}s".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),epoch, epoch_loss, self.optimizer.param_groups[0]['lr'], (end - start).total_seconds())
        self.writer.write_train(train_msg)

    def val_one_epoch(self, epoch):
        start = datetime.now()
        epoch_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_data_loader, 1):
                GT, LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                GT = Variable(GT).float()
                LRHSI = Variable(LRHSI).float()
                HRMSI = Variable(HRMSI).float()
                output_HRHSI = self.model(LRHSI, HRMSI)
                net_loss = self.train_loss(output_HRHSI, GT)
                epoch_val_loss.append(net_loss.item())
                net_loss.backward()
            end = datetime.now()
            epoch_loss = np.nanmean(np.array(epoch_val_loss))  # compute the mean value of all losses, as one epoch loss
            val_msg = "{}--> epoch={} loss={:.8f} lr={} time={:.4}s".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                           epoch, epoch_loss,
                                                                           self.optimizer.param_groups[0]['lr'],
                                                                           (end - start).total_seconds())
            self.writer.write_val(val_msg)

    def test_per_epoch(self, epoch, save=False):
        sum_time = 0.0
        avg_psnr = 0.0
        avg_ergas = 0.0
        avg_sam = 0.0
        self.model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(self.test_data_loader, 1):
                GT, LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                GT = Variable(GT).float()
                LRHSI = Variable(LRHSI).float()
                HRMSI = Variable(HRMSI).float()
                start = datetime.now()
                output_HRHSI = self.model(LRHSI, HRMSI)
                end = datetime.now()
                sum_time += (end - start).total_seconds()
                img_name = batch[3]
                img_name = img_name[0].split('/')[-1]
                if save:
                    os.makedirs(os.path.join(self.save_dir, 'save_train_details', 'result_epoch_{}'.format(epoch)), exist_ok=True)
                    savemat(os.path.join(self.save_dir, 'save_train_details', 'result_epoch_{}'.format(epoch), "{}".format(img_name)), {
                        'result': output_HRHSI.detach().cpu().numpy(),
                        'hrhsi': GT.detach().cpu().numpy(),
                        'hrmsi': HRMSI.detach().cpu().numpy(),
                        'lrhsi': LRHSI.detach().cpu().numpy()})
                current_psnr = calc_psnr(GT, output_HRHSI)
                current_sam = calc_sam(GT, output_HRHSI)
                current_ergas = calc_ergas(GT, output_HRHSI, r=self.scale)
                avg_psnr += current_psnr
                avg_sam += current_sam
                avg_ergas += current_ergas
        num_test = len(self.test_data_loader)
        avg_psnr = avg_psnr / num_test
        avg_sam = avg_sam / num_test
        avg_ergas = avg_ergas / num_test
        test_msg = "{}--> epoch={} Ave_psnr={:.2f} Ave_sam={:.2f}  Ave_ergas={:.4f} time={:.4}s".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             epoch,
            avg_psnr,
            avg_sam,
            avg_ergas,
            sum_time)
        print(test_msg)
        self.writer.write_test(test_msg)
