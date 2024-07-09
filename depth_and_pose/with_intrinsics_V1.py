import numpy as np
import torch
from kornia.geometry.depth import depth_to_normals
from pytorch_lightning import LightningModule

import losses.loss_functions as LossF
from models.DepthNet import DepthNet
from models.PoseNet import PoseNet
from vidar.vidar.arch.networks.intrinsics.IntrinsicsNet import IntrinsicsNet
from visualization import *


class with_intrinsics_v1(LightningModule):
    def __init__(self, hparams):
        super(with_intrinsics_v1, self).__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters()
        #num_epochs = self.hparams.hparams.num_epochs
        #print(f"Number of epochs: {self.num_epochs}")
        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet()
        self.intrinsics_net = IntrinsicsNet()
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"Number of available GPUs: {self.num_gpus}")
    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr*self.num_gpus},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr*self.num_gpus},
            {'params': self.intrinsics_net.parameters(), 'lr': self.hparams.hparams.lr*self.num_gpus}
        ]
        optimizer = torch.optim.Adam(optim_params)
        return [optimizer]

    

    def training_step(self, batch, batch_idx):
        tgt_img, ref_imgs = batch
        
        if self.current_epoch < 10:
            for param in self.intrinsics_net.parameters():
                param.requires_grad = False
        else:
            for param in self.intrinsics_net.parameters():
                param.requires_grad = True

        intrinsics = self.intrinsics_net(tgt_img)
        # network forward
        tgt_depth = self.depth_net(tgt_img)
        ref_depths = [self.depth_net(im) for im in ref_imgs]
        
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]

        # compute normal
        #tgt_normal = depth_to_normals(tgt_depth, intrinsics)
        #tgt_pseudo_normal = depth_to_normals(tgt_pseudo_depth, intrinsics)

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.normal_matching_weight
        #w4 = self.hparams.hparams.mask_rank_weight
        #w5 = self.hparams.hparams.normal_rank_weight

        loss_1, loss_2, dynamic_mask = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                    intrinsics, poses, poses_inv, self.hparams.hparams)

        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)
        # normal_l1_loss
        #loss_3 = (tgt_normal-tgt_pseudo_normal).abs().mean()

        # mask ranking loss
        #loss_4 = LossF.mask_ranking_loss(tgt_depth, tgt_pseudo_depth, dynamic_mask)

        # normal ranking loss
        #loss_5 = LossF.normal_ranking_loss(tgt_pseudo_depth, tgt_img, tgt_normal, tgt_pseudo_normal)
        
        loss = 1*loss_1 + 0.5*loss_2 + w3*loss_3
        
        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)
        #self.log('train/normal_l1_loss', loss_3)
        #self.log('train/mask_ranking_loss', loss_4)
        #self.log('train/normal_ranking_loss', loss_5)

        return loss
        
    def validation_step(self, batch, batch_idx):

        if self.hparams.hparams.val_mode == 'depth':
            tgt_img, gt_depth = batch
            tgt_depth = self.depth_net(tgt_img)
            errs = LossF.compute_errors(gt_depth, tgt_depth, self.hparams.hparams.dataset_name)
            
            errs = {'abs_diff': errs[0], 'abs_rel': errs[1],
                    'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        elif self.hparams.hparams.val_mode == 'photo':
            tgt_img, ref_imgs = batch
            
            intrinsics = self.intrinsics_net(tgt_img)
            tgt_depth = self.depth_net(tgt_img)
            ref_depths = [self.depth_net(im) for im in ref_imgs]
            poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
            poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]

            loss_1, _, _ = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                           intrinsics, poses, poses_inv, self.hparams.hparams)
            errs = {'photo_loss': loss_1.item()}
        else:
            print('wrong validation mode')
   
        if self.global_step < 10:
            return errs

        # plot  
        if batch_idx < 3:
            vis_img = visualize_image(tgt_img[0]) # (3, H, W)
            vis_depth = visualize_depth(tgt_depth[0,0]) # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0) # (3, 2*H, W)
            self.logger.experiment.add_images('val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)
        self.validation_step_outputs.append(errs)
        return errs
    
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.hparams.hparams.val_mode == 'depth':
            mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
            mean_diff = np.array([x['abs_diff'] for x in outputs]).mean()
            mean_a1 = np.array([x['a1'] for x in outputs]).mean()
            mean_a2 = np.array([x['a2'] for x in outputs]).mean()
            mean_a3 = np.array([x['a3'] for x in outputs]).mean()
            
            self.log('val_loss', mean_rel, prog_bar=True)
            self.log('val/abs_diff', mean_diff)
            self.log('val/abs_rel', mean_rel)
            self.log('val/a1', mean_a1, on_epoch=True)
            self.log('val/a2', mean_a2, on_epoch=True)
            self.log('val/a3', mean_a3, on_epoch=True)

        elif self.hparams.hparams.val_mode == 'photo':
            mean_pl = np.array([x['photo_loss'] for x in outputs]).mean()
            self.log('val_loss', mean_pl, prog_bar=True)
            self.validation_step_outputs.clear() 
  