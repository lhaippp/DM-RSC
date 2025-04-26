import yaml
import torch
from model.Unet.unet import Unet
from model.ODM.occlusion_diffusion import ODM
from model.trainer import Trainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config):
    # Model
    model_cfg = config['model']
    model = Unet(
        dim=model_cfg['dim'],
        dim_mults=tuple(model_cfg['dim_mults']),
        num_classes=model_cfg['num_classes'],
        cond_drop_prob=model_cfg['cond_drop_prob'],
        input_dim=model_cfg['input_dim'],
        condition_dim=model_cfg['condition_dim'],
        out_dim=model_cfg['out_dim']
    )

    # Diffusion
    diffusion_cfg = config['diffusion']
    diffusion = ODM(
        model,
        image_size=diffusion_cfg['image_size'],
        timesteps=diffusion_cfg['timesteps'],
        sampling_timesteps=diffusion_cfg['sampling_timesteps'],
        beta_schedule=diffusion_cfg['beta_schedule'],
        objective=diffusion_cfg['objective']
    )

    # Dataset
    dataset_cfg = config['dataset']
    dataset_name = dataset_cfg['name']

    if dataset_name == 'fastec':
        from dataset.ODM.fastec_dataset_ODM import Dataset_fastec_ODM
        DatasetClass = Dataset_fastec_ODM
    elif dataset_name == 'bsrsc':
        from dataset.ODM.BSRSC_dataset_ODM import Dataset_BSRSC_ODM
        DatasetClass = Dataset_BSRSC_ODM
    elif dataset_name == 'carla':
        from dataset.ODM.Carla_dataset_ODM import Dataset_Carla_ODM
        DatasetClass = Dataset_Carla_ODM
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    train_ds = DatasetClass(dataset_cfg['train_path'])
    test_ds = DatasetClass(dataset_cfg['test_path'])

    # Trainer
    trainer_cfg = config['trainer']
    trainer = Trainer(
        diffusion,
        train_ds=train_ds,
        test_ds=test_ds,
        train_batch_size=trainer_cfg['train_batch_size'],
        train_lr=trainer_cfg['train_lr'],
        train_num_steps=trainer_cfg['train_num_steps'],
        gradient_accumulate_every=trainer_cfg['gradient_accumulate_every'],
        ema_decay=trainer_cfg['ema_decay'],
        amp=trainer_cfg['amp'],
        save_and_sample_every=trainer_cfg['save_and_sample_every'],
        results_folder=trainer_cfg['results_folder'],
        flow_size=trainer_cfg['flow_size']
    )

   
    # Start training
    trainer.train_ODM()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
