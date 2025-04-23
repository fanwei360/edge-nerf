import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import numpy as np
# import torch
from opt import get_opts
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.accelerators import accelerator

# from models.density import EdgeDensity


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.mse_loss = loss_dict['mse'](coef=10)
        self.edge_density_consistency = loss_dict['edge_density_consistency'](coef=100)
        self.ada_mse_loss = loss_dict['adaptive_mse'](coef=10)
        self.sparsity_loss = loss_dict['sparsity'](coef=1)
        # self.depth_consistency_loss = loss_dict['depth_consistency'](coef=0.01)
        # self.depth_mask_loss = loss_dict['depth_mask'](coef=0.01)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')
        # print(self.nerf_coarse)

        # self.edge2density = EdgeDensity()
        # self.models['edge2density'] = self.edge2density

        if hparams.N_importance > 0:    # number of additional fine samples
            self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays, batch_num, test_time=False):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):   # 分块计算
            rendered_ray_chunks, beta = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            test_time,
                            batch_nb=batch_num,
                            all_batch=int(len(self.train_dataset)/self.hparams['batch_size']))

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results, beta

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = 1
            kwargs['mask'] = self.hparams['no_mask']
            kwargs['sample_num'] = self.hparams['sample_num']
        if self.hparams.dataset_name == 'dtu':
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        train_dataloader_ = DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
        return train_dataloader_

    def val_dataloader(self):
        val_dataloader_ = DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,     # validate one image (H*W rays) at a time
                          pin_memory=True)
        return val_dataloader_

    def training_step(self, batch, batch_nb):  # batch_nb = batchID
        rays, rgbs, edges = batch['rays'], batch['rgbs'], batch['edges'] # rays: [1, batch_size, 8]; rgbs: torch.Size([1, batch_size, channel=1])
        rays = torch.concat([rays, edges.squeeze(0)], dim=-1)
        results, beta = self(rays, batch_nb)

        color_loss = self.mse_loss(results, rgbs)
        edge_loss = self.ada_mse_loss(results, edges)
        sem_sparsity_loss = self.sparsity_loss(results, edges)   # Sparsity loss
        sem_consistency_loss = self.edge_density_consistency(results)  # consistency loss

        total_loss = color_loss + edge_loss + sem_sparsity_loss + sem_consistency_loss
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            e_psnr_ = psnr(results[f'edge_map_{typ}'], edges)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('beta', beta, prog_bar=True)

        self.log('train/color_loss', color_loss, prog_bar=True)
        self.log('train/edge_loss', edge_loss, prog_bar=True)
        self.log('train/sem_sparsity_loss', sem_sparsity_loss, prog_bar=True)
        self.log('train/sem_consistency', sem_consistency_loss, prog_bar=True)
        self.log('train/all_loss', total_loss, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)
        self.log('train/e_psnr', e_psnr_, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, edges_gt = batch['rays'], batch['rgbs'], batch['edges']
        if self.hparams['no_mask']:
            mask = batch['mask'].squeeze(0)
        else:
            mask = None
        rays = rays.squeeze()
        rays = torch.concat([rays, edges_gt.squeeze(0)], dim=-1)
        results, _ = self(rays, batch_nb, test_time=True)  # forward

        rgbs = rgbs.squeeze(0)
        val_loss = self.mse_loss(results, rgbs)

        edges_gt = edges_gt.squeeze(0)
        # val_loss = self.edge_mse_loss(results, edges_gt)

        log = {'val_loss': val_loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu().numpy() * 255
            img_gt = rgbs.view(H, W, 3).cpu().numpy() * 255
            img = Image.fromarray(img.astype('uint8'))
            img_gt = Image.fromarray(img_gt.astype('uint8'))
            img = transforms.ToTensor()(img)
            img_gt = transforms.ToTensor()(img_gt)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
            edges_depth = visualize_depth(results[f'edge_depth_{typ}'].view(H, W))
            edges_depth_cut = visualize_depth(results[f'edge_depth_cut_{typ}'].view(H, W))

            edges = results[f'edge_map_{typ}'].view(H, W).cpu().numpy() * 255
            edges = Image.fromarray(edges.astype('uint8')).convert("RGB")
            edges = transforms.ToTensor()(edges)  # (3, H, W)
            edges_img_gt = edges_gt.view(H, W).cpu().numpy() * 255
            edges_img_gt = Image.fromarray(edges_img_gt).convert("RGB")
            edges_img_gt = transforms.ToTensor()(edges_img_gt)  # (3, H, W)

            edges_cut = results[f'edge_map_cut_{typ}'].view(H, W).cpu().numpy() * 255
            edges_cut = Image.fromarray(edges_cut.astype('uint8')).convert("RGB")
            edges_cut = transforms.ToTensor()(edges_cut)  # (3, H, W)

            stack = torch.stack([img_gt, img, depth, edges_depth_cut, edges_cut, edges_img_gt])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs, valid_mask=mask)
        log['val_psnr'] = psnr_
        edge_psnr = psnr(results[f'edge_map_{typ}'], edges_gt)
        log['val_edge_psnr'] = edge_psnr

        return log

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)

    os.makedirs(os.path.join(hparams.save_dir, hparams.exp_name), exist_ok=True)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir=hparams.save_dir,
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus == 1 else None,
                      strategy=DDPPlugin(find_unused_parameters=False) if len(hparams.num_gpus) > 1 else None,
                      val_check_interval=0.5)
                      # auto_lr_find = True
                      # log_every_n_steps=50,
                      # precision=16


    trainer.fit(system)

if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)

