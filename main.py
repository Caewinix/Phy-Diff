import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from torch import optim
from enum import Enum, auto
from foundation.running import ExponentialMovingAverage
from utils.scheduler import GradualWarmupScheduler
from utils import config_tool
import torch_ext as te
from foundation.running.utils.checkpoint_details_loader import checkpoint_sorted_key, load_checkpoints_details
from accelerate.utils import set_seed
from foundation import running
from torch.utils.tensorboard import SummaryWriter

class ModeType(Enum):
    train = auto()
    val = auto()
    test = auto()
    ema = auto()
    
    @staticmethod
    def arg_type(arg_str: str):
        return eval(f'ModeType.{arg_str}')

class ModelType(Enum):
    diffusion = auto()
    adapter = auto()
    
    @staticmethod
    def arg_type(arg_str: str):
        return eval(f'ModelType.{arg_str}')

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-m', '--mode', type=ModeType.arg_type, required=True,  help='The mode of this program.')
    parser.add_argument('-t', '--model-type', type=ModelType.arg_type, default=ModelType.diffusion,  help='The model type should be runned.')
    parser.add_argument('-c', '--config', type=str, default='default',  help='The filename of the config in the `./configs` or the file path of the config.')
    parser.add_argument('-s', '--seed', type=int, default=3407,  help='The seed of the random.')
    args = parser.parse_args()
    return args

def _get_model_from_config(config, model_config):
    specific_type_str = model_config.type
    exec(f'from {model_config.package} import {specific_type_str}')
    specific_parameters_dict = vars(model_config.parameters)
    for key, item in specific_parameters_dict.items():
        if isinstance(item, config_tool.script_func):
            specific_parameters_dict[key] = item.call(config)
    specific_model = eval(specific_type_str)(**specific_parameters_dict)
    return specific_model

def _get_optimizer_from_config(optimizer_config, model_parameters):
    optimizer_str = optimizer_config.type
    exec(f'from {optimizer_config.package} import {optimizer_str}')
    optimizer = eval(optimizer_str)(model_parameters, **vars(optimizer_config.parameters))
    return optimizer

def _resume(checkpoints_dir):
    ckpt_filenames = sorted(os.listdir(checkpoints_dir), key=checkpoint_sorted_key)
    ckpt_filenames = [filename for filename in ckpt_filenames if filename.endswith('.pt')]
    if len(ckpt_filenames) > 0:
        final_ckpt_filename = ckpt_filenames[-1]
        ckpt_path = os.path.join(checkpoints_dir, final_ckpt_filename)
        start_epoch, start_step, learning_rate, optim_details = load_checkpoints_details(checkpoints_dir)
        ckpt = torch.load(ckpt_path, map_location=device)
        return start_epoch, start_step, learning_rate, optim_details, ckpt
    else:
        return 0, None, None, None, None

if __name__ == '__main__':
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    
    config_path: str = args.config
    if not config_path.endswith('.yaml'):
        if args.model_type == ModelType.adapter:
            config_name = ModelType.diffusion.name
        else:
            config_name = args.model_type.name
        config_path = f'./configs/{config_name}/{config_path}.yaml'
    config = config_tool.load(config_path)
    
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    
    dataset_config = config.dataset
    dataset_type_str = dataset_config.type
    exec(f'from {dataset_config.package} import {dataset_type_str}')
    
    model_config = config.model
    
    # Device
    device_config = model_config.device
    device_id = device_config.id
    if not isinstance(device_id, list):
        device_id = [device_id]
    device = torch.device('cpu' if (device_config.type != 'cuda') or (not torch.cuda.is_available()) else f'cuda:{device_id[0]}')
    
    # Backbone
    backbone_model: nn.Module = _get_model_from_config(config, model_config.backbone).to(device)
    if model_config.backbone.pretrained is not None:
        backbone_model.load_state_dict(torch.load(model_config.backbone.pretrained, map_location=device))
    
    adapter_config = None
    adapter = None
    if config.adapter is not None:
        adapter_config = config.adapter
        if adapter_config.enabled:
            adapter_config_path = adapter_config.config_path
            if not adapter_config_path.endswith('.yaml'):
                adapter_config_path = f'./configs/adapter/{adapter_config_path}.yaml'
            adapter_config = config_tool.load(adapter_config_path)
            if args.mode != ModeType.train or args.model_type == ModelType.adapter:
                adapter: nn.Module = _get_model_from_config(adapter_config, adapter_config.model.backbone).to(device)
                adapter = nn.DataParallel(adapter, device_ids=device_id)
    config.adapter.adapter = adapter
    
    diffusion_config = model_config.diffusion
    
    if args.mode != ModeType.test:
        # Diffuser
        diffuser_type_str = diffusion_config.diffuser_type
        exec(f'from {diffusion_config.package} import {diffuser_type_str}')
        diffuser = eval(diffuser_type_str)(
            backbone_model, **vars(diffusion_config.parameters)).to(device)
        # cuda_id = int(modelConfig["device"].split(":")[1])
        # device_ids = np.arange(torch.cuda.device_count())
        # device_ids = list(np.insert(device_ids[device_ids != cuda_id], 0, cuda_id))
        diffuser = nn.DataParallel(diffuser, device_ids=device_id)
        
        if args.mode == ModeType.train:
            
            train_config = config.train
            if train_config.load_weight is not None:
                backbone_model.load_state_dict(torch.load(os.path.join(
                    config.checkpoints_dir, train_config.load_weight), map_location=device))
            data_getter = train_config.data_getter
            model_parameters = backbone_model.parameters()
            logging_path = config.log.path
            ema_object = backbone_model
            
            start_epoch, start_step, learning_rate_temp, optim_details, ckpt = 0, *[None for _ in range(4)]
            if adapter is not None and args.model_type == ModelType.adapter:
                train_config = adapter_config.train
                model_parameters = adapter.parameters()
                logging_path = adapter_config.log.path
                ema_object = adapter
                if train_config.resume:
                    try:
                        start_epoch, start_step, learning_rate_temp, optim_details, ckpt = _resume(adapter_config.checkpoints_dir)
                    except:
                        pass
                    if ckpt is not None:
                        te.refine_model(adapter).load_state_dict(ckpt)
                    model_parameters = adapter.parameters()
                adapter_condition_getter = train_config.adapter_condition_getter
            else:
                if train_config.resume:
                    try:
                        start_epoch, start_step, learning_rate_temp, optim_details, ckpt = _resume(config.checkpoints_dir)
                    except:
                        pass
                    if ckpt is not None:
                        backbone_model.load_state_dict(ckpt)
                    model_parameters = backbone_model.parameters()
                adapter_condition_getter = argparse.Namespace()
                adapter_condition_getter.call = lambda _: None
            
            dataset = eval(dataset_type_str)(*config_tool.parse_args(dataset_config.mode.train.args), **config_tool.parse_kwargs(vars(dataset_config.mode.train.kwargs)))
            
            if train_config.resume:
                if learning_rate_temp is None:
                    learning_rate = train_config.optimizer.parameters.lr
                else:
                    learning_rate = learning_rate_temp
                train_config.optimizer.parameters.lr = learning_rate
            
            optimizer: nn.Module = _get_optimizer_from_config(train_config.optimizer, model_parameters)
            if optim_details is not None:
                optimizer.load_state_dict(optim_details['optimizer'])
                scaler_state = optim_details['scaler']
            else:
                scaler_state = None
            data_loader = DataLoader(
                dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers, pin_memory=True)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=train_config.epochs, eta_min=0, last_epoch=-1)
            warm_up_scheduler = GradualWarmupScheduler(
                optimizer=optimizer, multiplier=train_config.scheduler_multiplier, warm_epoch=train_config.epochs // 10, after_scheduler=cosine_scheduler)
            
            os.makedirs(logging_path, exist_ok=True)
            writer = SummaryWriter(logging_path)
            
            if train_config.ema_decay is not None:
                ema_model = ExponentialMovingAverage(ema_object, decay=train_config.ema_decay).to(device)
            else:
                ema_model = None
            
            running.diffusion.train(
                diffuser=diffuser,
                backbone_model=backbone_model,
                adapter=adapter,
                device=device,
                data_getter=data_getter.call(config),
                adapter_condition_getter=adapter_condition_getter.call(config),
                optimizer=optimizer,
                lr_scheduler=warm_up_scheduler,
                train_data_loader=data_loader,
                grad_clip=train_config.grad_clip,
                epochs=train_config.epochs,
                start_epoch=start_epoch,
                start_step=start_step,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                mixed_precision=train_config.mixed_precision,
                use_accelerator=train_config.use_accelerator,
                ema_model=ema_model,
                checkpoints_dir=config.checkpoints_dir if args.model_type == ModelType.diffusion else adapter_config.checkpoints_dir,
                step_interval=train_config.step_interval,
                writer=writer,
                scaler_state=scaler_state
            )
            
            writer.close()
        
        elif args.mode == ModeType.val:
            val_config = config.val
            dataset = eval(dataset_type_str)(*config_tool.parse_args(dataset_config.mode.val.args), **config_tool.parse_kwargs(vars(dataset_config.mode.val.kwargs)))
            torch.manual_seed(torch.initial_seed())
            first_data_number = val_config.first_data_number
            if first_data_number is not None:
                indices = torch.randperm(len(dataset))
                indices = indices[:first_data_number]
                dataset = Subset(dataset, indices)
            data_loader = DataLoader(
                dataset, batch_size=val_config.batch_size, shuffle=False, num_workers=val_config.num_workers, pin_memory=True)
            
            if adapter is None:
                checkpoints_dir = config.checkpoints_dir
                adapter_condition_getter = argparse.Namespace()
                adapter_condition_getter.call = lambda _: None
            else:
                checkpoints_dir = adapter_config.checkpoints_dir
                adapter_condition_getter = adapter_config.train.adapter_condition_getter
            
            # Sampler
            sampler_type_str = diffusion_config.sampler_type
            exec(f'from {diffusion_config.package} import {sampler_type_str}')
            backbone_model = nn.DataParallel(backbone_model, device_ids=device_id)
            backbone_model.eval()
            sampler = eval(sampler_type_str)(
                backbone_model, **vars(diffusion_config.parameters)).to(device)
            
            # start_epoch, start_step = checkpoint_sorted_key(val_config.start_weight)
            
            running.diffusion.val(
                sampler=sampler,
                adapter=adapter,
                device=device,
                data_getter=val_config.data_getter.call(config),
                adapter_condition_getter=adapter_condition_getter.call(config),
                comparison_getter=val_config.comparison_getter.call(config),
                initial_state_getter=val_config.initial_state_getter.call(config),
                initial_noise_strength=val_config.initial_noise_strength,
                sampled_images_transform=val_config.sampled_images_transform.call(config),
                data_loader=data_loader,
                mixed_precision=val_config.mixed_precision,
                start_epoch=val_config.start_epoch,
                start_step=val_config.start_step,
                checkpoints_dir=checkpoints_dir
            )
        
        elif args.mode == ModeType.ema:
            post_train_ema_config = config.post_train_ema
            ckpt = torch.load(os.path.join(
                config.checkpoints_dir, post_train_ema_config.start_weight), map_location=device)
            backbone_model.load_state_dict(ckpt)
            ema_model = ExponentialMovingAverage(backbone_model, decay=post_train_ema_config.decay).to(device)
            start_epoch, start_step = checkpoint_sorted_key(post_train_ema_config.start_weight)
            
            running.diffusion.ema_existing_checkpoints(
                backbone_model=backbone_model,
                ema_model=ema_model,
                device=device,
                start_epoch=start_epoch,
                start_stepped_batch=start_step,
                checkpoints_dir=config.checkpoints_dir
            )
    
    else:
        test_config = config.test
        dataset = eval(dataset_type_str)(*config_tool.parse_args(dataset_config.mode.test.args), **config_tool.parse_kwargs(vars(dataset_config.mode.test.kwargs)))
        torch.manual_seed(torch.initial_seed())
        first_data_number = test_config.first_data_number
        if first_data_number is not None:
            indices = torch.randperm(len(dataset))
            indices = indices[:first_data_number]
            dataset = Subset(dataset, indices)
        data_loader = DataLoader(
            dataset, batch_size=test_config.batch_size, shuffle=False, num_workers=test_config.num_workers, pin_memory=True)
        
        if adapter is None:
            checkpoints_dir = config.checkpoints_dir
            load_weight = test_config.load_weight
            adapter_condition_getter = argparse.Namespace()
            adapter_condition_getter.call = lambda _: None
            ckpt = torch.load(os.path.join(
                checkpoints_dir, load_weight), map_location=device)
            backbone_model.load_state_dict(ckpt)
            print(f"Model loading weight done: {load_weight}")
            backbone_model.eval()
        else:
            checkpoints_dir = adapter_config.checkpoints_dir
            adapter_condition_getter = adapter_config.train.adapter_condition_getter
            load_weight = adapter_config.test.load_weight
            ckpt = torch.load(os.path.join(
                checkpoints_dir, load_weight), map_location=device)
            te.refine_model(adapter).load_state_dict(ckpt)
            print(f"Model loading weight done: {load_weight}")
            adapter.eval()
        
        # Sampler
        sampler_type_str = diffusion_config.sampler_type
        exec(f'from {diffusion_config.package} import {sampler_type_str}')
        backbone_model = nn.DataParallel(backbone_model, device_ids=device_id)
        sampler = eval(sampler_type_str)(
            backbone_model, **vars(diffusion_config.parameters)).to(device)
        
        running.diffusion.test(
            sampler=sampler,
            adapter=adapter,
            device=device,
            data_getter=test_config.data_getter.call(config),
            adapter_condition_getter=adapter_condition_getter.call(config),
            comparison_getter=test_config.comparison_getter.call(config),
            initial_state_getter=test_config.initial_state_getter.call(config),
            initial_noise_strength=test_config.initial_noise_strength,
            sampled_images_transform=test_config.sampled_images_transform.call(config),
            data_loader=data_loader,
            sampled_images_saving_path=test_config.sampled_images_saving_path,
            mixed_precision=test_config.mixed_precision,
            n_rows=test_config.n_rows
        )