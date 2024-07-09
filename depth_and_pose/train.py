from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.profilers import PyTorchProfiler
import torch
import time

from config import get_opts
from data_modules import VideosDataModule
from with_intrinsics_V1 import with_intrinsics_v1
from with_intrinsics_V2 import with_intrinsics_v2

if __name__ == '__main__':
    hparams = get_opts()
    
    if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Number of available GPUs: {num_gpus}")
    # pl model

    if hparams.model_version == 'w1':
        system = with_intrinsics_v1(hparams)
    elif hparams.model_version == 'w2':
        system = with_intrinsics_v2(hparams)


    torch.autograd.set_detect_anomaly(True)
    # pl data module
    dm = VideosDataModule(hparams)

    # pl logger
    logger = TensorBoardLogger(
        save_dir="ckpts",
        name=hparams.exp_name
    )
    
    # save checkpoints
    ckpt_dir = 'ckpts/{}/version_{:d}'.format(
        hparams.exp_name, logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{val_loss:.4f}',
                                          monitor='val_loss',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=3)

    # restore from previous checkpoints
    if hparams.ckpt_path is not None:
        print('load pre-trained model from {}'.format(hparams.ckpt_path))
        system = system.load_from_checkpoint(
            hparams.ckpt_path, strict=False, hparams=hparams)
    #profiler = PyTorchProfiler(group_by_input_shape=True)
    # set up trainer
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=hparams.num_epochs,
        limit_train_batches=hparams.epoch_size,
        limit_val_batches=8 if hparams.val_mode == 'photo' else 1.0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=logger,
        devices = num_gpus,
        benchmark=True
    )
    start_time = time.time()
    
    trainer.fit(system, dm)
    
    end_time = time.time()
    training_time =  end_time - start_time
    print("Total training time:", training_time/60, " minutes")
    #resource_stats = profiler.summary()
    #print("Resource usage:", resource_stats)
