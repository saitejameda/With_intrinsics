import numpy as np
import torch
from kornia.geometry.depth import depth_to_normals
from pytorch_lightning import LightningModule
from visualization import *
import losses.loss_functions as LossF 
from models.DepthNet import DepthNet
from models.PoseNet_V2 import PoseNet_V2
from visualization import *
from vidar.vidar.geometry.camera_ucm import UCMCamera
from vidar.vidar.arch.networks.intrinsics.IntrinsicsNet import IntrinsicsNet
from imageio import imread, imwrite
from pathlib import Path
from data_modules import *
 
class with_intrinsics_v2(LightningModule):
    def __init__(self, hparams=None):
        super(with_intrinsics_v2, self).__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        # 
        
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet_V2(model_version=self.hparams.hparams.model_version)
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"Number of available GPUs: {self.num_gpus}")
        
    
    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr*self.num_gpus},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr*self.num_gpus},
                        ]
        optimizer = torch.optim.Adam(optim_params)
        return [optimizer]    
  

    def training_step(self, batch, batch_idx):
        tgt_img, ref_imgs= batch
        #intrinsics = intrinsics[0].cpu().detach().numpy()
        #print('intrinsics',intrinsics[0])
        # network forward
        #output_dir = Path("/home/meda/Thesis")
        tgt_depth = self.depth_net(tgt_img)
        #vis = visualize_depth(tgt_depth[0, 0]).permute(1, 2, 0).numpy() * 255
        #imwrite(output_dir/'depth.jpg',vis.astype(np.uint8))
        ref_depths = [self.depth_net(im) for im in ref_imgs]
        # for im in ref_imgs:
        #     poses, intrinsics = self.pose_net(tgt_img, im)
        #     print('poses',poses.shape)
        #     print('intrinsics_here',intrinsics.shape)
        #print([self.pose_net(tgt_img, im) for im in ref_imgs])
        outs = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_1,intr_1 = outs[0][0],outs[0][1]
        poses_2,intr_2 = outs[1][0],outs[1][1]
        

        outs_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]
        poses_inv_1,intr_inv_1 = outs_inv[0][0],outs_inv[0][1]
        poses_inv_2,intr__inv_2 = outs_inv[1][0],outs_inv[1][1]
        
        intrinsics_1 = 0.5 * (intr_1 + intr_inv_1)
        intrinsics_2 = 0.5 * (intr_2 + intr__inv_2)
        intrinsics = [intrinsics_1,intrinsics_2]
        poses = [poses_1,poses_2]
       # print('poses',poses[0].shape)
        poses_inv = [poses_inv_1,poses_inv_2]
        #poses, intrinsics = self.pose_net(tgt_img, ref_imgs[0], intrinsics)

        # compute normal
        #tgt_normal = depth_to_normals(tgt_depth, intrinsics)
        #tgt_pseudo_normal = depth_to_normals(tgt_pseudo_depth, intrinsics)

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.normal_matching_weight
        #w4 = self.hparams.hparams.mask_rank_weight
        #w5 = self.hparams.hparams.normal_rank_weight

        loss_1, loss_2,_ = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                    intrinsics, poses, poses_inv, self.hparams.hparams)

        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)
        
        #loss_4 = LossF.edge_aware_regularization_loss(tgt_depth)
        #loss_5 = LossF.temporal_consistency_loss(ref_depths[0], ref_depths[1])
        #loss_4 = 
        # normal_l1_loss
        #loss_3 = (tgt_normal-tgt_pseudo_normal).abs().mean()

        # mask ranking loss
        #loss_4 = LossF.mask_ranking_loss(tgt_depth, tgt_pseudo_depth, dynamic_mask)

        # normal ranking loss
        #loss_5 = LossF.normal_ranking_loss(tgt_pseudo_depth, tgt_img, tgt_normal, tgt_pseudo_normal)
        print('w_1','1')
        print('w_2','0.5')
        print('w_3','0.2')
        
        loss = 1*loss_1 + 0.5*loss_2 + 0.2*loss_3

        #np.save('intrinsics.npy', intrinsics[0].cpu().detach().numpy())
        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/compute_smooth_loss', loss_3)
        #self.log('intrincis', intrinsics[0])
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
            #fx, fy, cx, cy = intrinsics
            #fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()
            #intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            #print('intrinsics',intrinsics)
            tgt_depth = self.depth_net(tgt_img)
            ref_depths = [self.depth_net(im) for im in ref_imgs]
            outs = [self.pose_net(tgt_img, im) for im in ref_imgs]
            poses_1,intr_1 = outs[0][0],outs[0][1]
            poses_2,intr_2 = outs[1][0],outs[1][1]
        

            outs_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]
            poses_inv_1,intr_inv_1 = outs_inv[0][0],outs_inv[0][1]
            poses_inv_2,intr__inv_2 = outs_inv[1][0],outs_inv[1][1]
        
            intrinsics_1 = 0.5 * (intr_1 + intr_inv_1)
            intrinsics_2 = 0.5 * (intr_2 + intr__inv_2)
            intrinsics = [intrinsics_1,intrinsics_2]
            poses = [poses_1,poses_2]
            poses_inv = [poses_inv_1,poses_inv_2]

            loss_1, _ ,_= LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
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
            self.logger.experiment.add_images(
                'val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)
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
  

