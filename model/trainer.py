
from pathlib import Path
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from torch.optim import Adam
from timm.models.layers import trunc_normal_

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#from utils import *
from model.flow_utils import *

# constants
import re
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
#import matplotlib.pyplot as plt

def divisible_by(numer, denom):
    return (numer % denom) == 0

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, diffusion_model, train_ds, test_ds, *,
                 train_batch_size=16, gradient_accumulate_every=1,
                 train_lr=1e-4,  ema_update_every=10,train_num_steps=100000,
                 ema_decay=0.995, adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000, 
                 results_folder='./results', amp=False,
                 mixed_precision_type='fp16', split_batches=True,
                 flow_size = [96,128],
                 max_grad_norm=1.):

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
        )

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.is_ddim_sampling = diffusion_model.is_ddim_sampling
        self.flow_h = flow_size[0]
        self.flow_w = flow_size[1]
        
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm

        self.ds = train_ds#Dataset_fastec(train_folder)
        self.test_ds = test_ds#Dataset_fastec_test(test_folder)

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        self.test_dl = DataLoader(self.test_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
        self.test_long = len(self.test_dl)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone = 1,file_name = None):
        device = self.device
        if file_name is not None:
            data = torch.load(file_name, map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train_MDM(self):
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    intra_flow = norm_flow(data['intra_flow'].to(self.device))
                    intra_flow1 = norm_flow(data['intra_flow1'].to(self.device))
                    flow12 = norm_flow(upsample2d_flow_as(data['flow12'].to(self.device), torch.ones(1,2,self.flow_h,self.flow_w), mode="bilinear", if_rate=True))
                    flow23 = norm_flow(upsample2d_flow_as(data['flow23'].to(self.device), torch.ones(1,2,self.flow_h,self.flow_w), mode="bilinear", if_rate=True))
                    intra_flow2 = upsample2d_flow_as(torch.cat((intra_flow, intra_flow1), dim=1), torch.ones(1,2,self.flow_h,self.flow_w), mode="bilinear", if_rate=True)
                    classes = torch.zeros(flow12.shape[0], dtype=torch.int64).to(self.device)
                    flow_condition = torch.cat((flow12,flow23),1)
                    with self.accelerator.autocast():
                        loss = self.model(flow=intra_flow2, classes=classes, condition = flow_condition)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f'loss: {total_loss:.4f}')
                self.accelerator.log({"total_loss": total_loss}, step=self.step)

                self.opt.step()
                self.opt.zero_grad()
                self.step += 1

                if self.accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        milestone = self.step // self.save_and_sample_every
                        self.sample_MDM(milestone)
                        self.save(milestone)
                pbar.update(1)
        self.accelerator.print('training complete')

    def sample_MDM(self, milestone = None):
        device = self.device
        if milestone == None:
            milestone = 'final'
        self.ema.ema_model.eval()
        for idx, test_data in tqdm(enumerate(self.test_dl), total=len(self.test_dl)):
            intra_flow = test_data['intra_flow'].to(device)
            intra_flow1 = test_data['intra_flow1'].to(device)
            flow12 = norm_flow(upsample2d_flow_as(test_data['flow12'].to(device), torch.ones(1,2,self.flow_h,self.flow_w), mode="bilinear", if_rate=True))
            flow23 = norm_flow(upsample2d_flow_as(test_data['flow23'].to(device), torch.ones(1,2,self.flow_h,self.flow_w), mode="bilinear", if_rate=True))
            rs2_ori = test_data['rs2_ori']
            gs2_ori = test_data['gs2_ori']
            path = test_data['path'][5]

            image_classes = torch.zeros([rs2_ori.shape[0]], dtype=torch.int64).to(device)
            flow_condition = torch.cat((flow12,flow23),1)
            out_flow = self.ema.ema_model.sample(classes=image_classes, condition = flow_condition, cond_scale=1)

            pre_rs2_gs2_flow = upsample2d_flow_as(out_flow[:,:2], rs2_ori, mode="bilinear", if_rate=True).cpu()
            true_rs2_gs2_flow = upsample2d_flow_as(intra_flow, rs2_ori, mode="bilinear", if_rate=True).cpu()
            pred_rs2_warp = flow_warp(rs2_ori, pre_rs2_gs2_flow, pad="zeros", mode="bilinear")

            pre_gs2_rs2_flow = upsample2d_flow_as(out_flow[:,2:], gs2_ori, mode="bilinear", if_rate=True).cpu()
            true_gs2_rs2_flow = upsample2d_flow_as(intra_flow1, rs2_ori, mode="bilinear", if_rate=True).cpu()
            pred_gs2_warp = flow_warp(gs2_ori, pre_gs2_rs2_flow, pad="zeros", mode="bilinear")

            for i in range(pred_rs2_warp.shape[0]):
                pred_gs2 = pred_rs2_warp[i]
                true_gs2 = gs2_ori[i]
                true_rs2 = rs2_ori[i]
                pred_rs2 = pred_gs2_warp[i]
                path_1 = path[i].split('/')
                part1 = '/'.join(path_1[-4:-1])
                frame_number = int(path_1[-1].split('_')[0])

                img_folder = os.path.join(self.results_folder / f'sample-{milestone}', part1, 'image')
                flow_folder = os.path.join(self.results_folder / f'sample-{milestone}', part1, 'flow')
                flow_npy_folder = os.path.join(self.results_folder / f'sample-{milestone}', part1, 'flow_npy')
                os.makedirs(img_folder, exist_ok=True)
                os.makedirs(flow_folder, exist_ok=True)
                os.makedirs(flow_npy_folder, exist_ok=True)

                img_path = os.path.join(img_folder, f'rs{frame_number}_to_gs{frame_number}.png')
                flow_img_path = os.path.join(flow_folder, f'rs{frame_number}_to_gs{frame_number}.png')
                flow_npy_path = os.path.join(flow_npy_folder, f'rs{frame_number}_to_gs{frame_number}.npy')
                flow_npy_path_back = os.path.join(flow_npy_folder, f'gs{frame_number}_to_rs{frame_number}.npy')

                result_image = torch.cat([true_rs2, pred_rs2, true_gs2, pred_gs2], dim=2)
                utils.save_image(result_image, img_path)

                np.save(flow_npy_path, pre_rs2_gs2_flow[i].numpy().transpose(1,2,0))
                np.save(flow_npy_path_back, pre_gs2_rs2_flow[i].numpy().transpose(1,2,0))

                pre_flow_img = flow_to_image(pre_rs2_gs2_flow[i].numpy().transpose(1,2,0))
                true_flow_img = flow_to_image(true_rs2_gs2_flow[i].numpy().transpose(1,2,0))
                pre_back_img = flow_to_image(pre_gs2_rs2_flow[i].numpy().transpose(1,2,0))
                true_back_img = flow_to_image(true_gs2_rs2_flow[i].numpy().transpose(1,2,0))

                flow_combined = np.concatenate([true_flow_img, pre_flow_img, true_back_img, pre_back_img], axis=1)
                cv2.imwrite(flow_img_path, flow_combined)
                
               
    def train_ODM(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data1 = next(self.dl)
                    rs1, rs2, rs3 = data1['rs1'].to(device), data1['rs2'].to(device), data1['rs3'].to(device)
                    gs2 = data1['gs2'].to(device)
                    flow = upsample2d_flow_as(data1['flow'].to(device), gs2, mode="bilinear", if_rate=True)
                    flow1 = upsample2d_flow_as(data1['flow1'].to(device), gs2, mode="bilinear", if_rate=True)
                    flow12 = upsample2d_flow_as(data1['flow12'].to(device), gs2, mode="bilinear", if_rate=True)
                    flow32 = upsample2d_flow_as(data1['flow32'].to(device), gs2, mode="bilinear", if_rate=True)

                    rs2_to_gs2 = flow_warp(rs2, flow, pad="zeros", mode="bilinear")
                    rs1 = flow_warp(flow_warp(rs1, flow, pad="zeros", mode="bilinear"), flow12, pad="zeros", mode="bilinear")
                    rs3 = flow_warp(flow_warp(rs3, flow, pad="zeros", mode="bilinear"), flow32, pad="zeros", mode="bilinear")
                    occ = get_occu_mask_bidirection(flow, flow1, scale=0.01, bias=0.5)
                    img_rs = torch.cat((rs1 * occ, rs2_to_gs2 * (1 - occ), rs3 * occ), 1)

                    class1 = torch.zeros(flow12.shape[0], dtype=torch.int64).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(img=gs2, classes=class1, condition=img_rs) / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.log({"total_loss": total_loss}, step=self.step)

                self.opt.step()
                self.opt.zero_grad()
                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        milestone = self.step // self.save_and_sample_every
                        self.sample_ODM(milestone)
                        self.save(milestone)
                pbar.update(1)
        accelerator.print('training complete')

    def sample_ODM(self, milestone = None):
        if milestone == None:
            milestone = 'final'
        device = self.device
        self.ema.ema_model.eval()
        for idx, test_data in tqdm(enumerate(self.test_dl), total=len(self.test_dl)):
            rs1, rs2, rs3 = test_data['rs1'].to(device), test_data['rs2'].to(device), test_data['rs3'].to(device)
            gs2 = test_data['gs2'].to(device)
            flow = upsample2d_flow_as(test_data['flow'].to(device), gs2, mode="bilinear", if_rate=True)
            flow1 = upsample2d_flow_as(test_data['flow1'].to(device), gs2, mode="bilinear", if_rate=True)
            flow12 = upsample2d_flow_as(test_data['flow12'].to(device), gs2, mode="bilinear", if_rate=True)
            flow32 = upsample2d_flow_as(test_data['flow32'].to(device), gs2, mode="bilinear", if_rate=True)

            occ = get_occu_mask_bidirection(flow, flow1, scale=0.01, bias=0.5)
            rs2_to_gs2 = flow_warp(rs2, flow, pad="zeros", mode="bilinear")
            rs1 = flow_warp(flow_warp(rs1, flow, pad="zeros", mode="bilinear"), flow12, pad="zeros", mode="bilinear")
            rs3 = flow_warp(flow_warp(rs3, flow, pad="zeros", mode="bilinear"), flow32, pad="zeros", mode="bilinear")

            img_rs = torch.cat((rs1 * occ, rs2_to_gs2 * (1 - occ), rs3 * occ), 1)
            image_classes = torch.zeros([gs2.shape[0]], dtype=torch.int64).to(device)
            with torch.inference_mode():
                pred_gs = self.ema.ema_model.sample(classes=image_classes, condition=img_rs, mask=occ, rs=rs2_to_gs2, cond_scale=1)

            for i in range(pred_gs.shape[0]):
                pred_gs1 = pred_gs[i].cpu()
                ori_gs2 = gs2[i].cpu()
                result_matrix = torch.cat([ori_gs2, pred_gs1], dim=2)

                path = test_data['path'][i] if 'path' in test_data else f'sample_{i}.png'
                output_dir = self.results_folder / f'sample-{milestone}' / 'image'
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.join(output_dir, f'{idx:05d}_image_{i:05d}.png')
                utils.save_image(result_matrix, image_path)

            
                
                
                
                
                
                
                