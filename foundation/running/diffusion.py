import os
from typing import Callable, Optional, Sequence, Union, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from .utils.exponential_moving_average import ExponentialMovingAverage
from torch.optim.lr_scheduler import LRScheduler
import torch_ext as te
from .utils.checkpoint_details_loader import checkpoint_sorted_key
from evaluation import PSNRMetric
from monai.metrics import SSIMMetric
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.tensorboard import SummaryWriter

def train(
    diffuser: nn.Module,
    backbone_model: nn.Module,
    adapter: Optional[nn.Module],
    device: torch.device,
    data_getter: Callable[[torch.Tensor, torch.device], tuple],
    adapter_condition_getter: Optional[Callable[[torch.Tensor, torch.device], tuple]],
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    train_data_loader: DataLoader,
    grad_clip: float,
    epochs: int,
    start_epoch: int = 0,
    start_step: Optional[int] = 0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: Union[bool, str] = False,
    use_accelerator: bool = False,
    ema_model: Optional[ExponentialMovingAverage] = None,
    checkpoints_dir: str = './checkpoints',
    step_interval: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    scaler_state: Optional[Any] = None
):
    checkpoint_training_details_dir = os.path.join(checkpoints_dir, 'training_details')
    os.makedirs(checkpoint_training_details_dir, exist_ok=True)
    training_optim_details_dir = os.path.join(checkpoint_training_details_dir, 'optim')
    os.makedirs(training_optim_details_dir, exist_ok=True)
    
    if adapter_condition_getter is None:
        def blank(_, __): return tuple()
        adapter_condition_getter = blank
    
    if adapter is None:
        def backbone_zero_grad_if_enable_adapter(backbone_model: None):
            pass
        def get_adaptation(adapter: None, *_):
            return None
        def select_saving_model(backbone_model: nn.Module, adapter: None):
            return te.refine_model(backbone_model)
        main_model = backbone_model
    else:
        def backbone_zero_grad_if_enable_adapter(backbone_model: nn.Module):
            backbone_model.zero_grad()
        def get_adaptation(adapter: nn.Module, required_condition):
            return adapter(*required_condition, need_tokens=True)
        def select_saving_model(backbone_model: nn.Module, adapter: nn.Module):
            return te.refine_model(adapter)
        main_model = adapter
    
    if step_interval is None:
        def get_checkpoint_saving_name(epoch, global_step):
            return f'ckpt_epoch_{epoch}'
        step_interval = len(train_data_loader)
    else:
        def get_checkpoint_saving_name(epoch, global_step):
            return f'ckpt_epoch_{epoch}_step_{global_step}'
    
    if ema_model is None:
        def update_if_ema(model):
            pass
        
        def save_ema_if_enabled(checkpoint_name: str, _ = None):
            pass
    else:
        ema_path = os.path.join(checkpoints_dir, 'ema')
        os.makedirs(ema_path, exist_ok=True)
        
        def update_if_ema(model):
            ema_model.update_parameters(model)
        
        if use_accelerator:
            def save_ema_if_enabled(checkpoint_name: str, accelerator):
                accelerator.save(ema_model.module.state_dict(), os.path.join(ema_path, f'{checkpoint_name}.pt'))
        else:
            def save_ema_if_enabled(checkpoint_name: str, _ = None):
                torch.save(ema_model.module.state_dict(), os.path.join(ema_path, f'{checkpoint_name}.pt'))
    
    def calculate_loss(required_data, required_condition):
        pass
    
    if use_accelerator:
        scaler = None
        accelerator_logging_dir = os.path.join(writer.log_dir, 'accelerator')
        os.makedirs(accelerator_logging_dir, exist_ok=True)
        project_config = ProjectConfiguration(
            logging_dir=accelerator_logging_dir
        )
        
        # define the acceleartor
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with='tensorboard',
            project_config=project_config
        )
        
        diffuser = te.refine_model(diffuser)
        adapter = te.refine_model(adapter)
        diffuser, backbone_model, adapter, optimizer, train_data_loader, lr_scheduler = accelerator.prepare(
            diffuser, backbone_model, adapter, optimizer, train_data_loader, lr_scheduler
        )
        
        def get_losses(required_data, required_condition):
            with accelerator.accumulate(main_model):
                losses = calculate_loss(required_data, required_condition)
                return losses
        
        def backpropagate(losses, i):
            loss = accelerator.gather(losses).mean().item()
            accelerator.backward(losses.mean())
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(main_model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                update_if_ema(main_model)
            return loss
        
        def save(model: nn.Module):
            accelerator.wait_for_everyone()
            checkpoint_name = get_checkpoint_saving_name(epoch, start_step)
            accelerator.save(model.state_dict(), os.path.join(
                checkpoints_dir, f'{checkpoint_name}.pt'))
            optim_details = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict()
            }
            accelerator.save(optim_details, os.path.join(training_optim_details_dir, f'latest_optim.pt'))
            save_ema_if_enabled(checkpoint_name, accelerator)
            if accelerator.is_main_process:
                np.savez_compressed(
                    os.path.join(checkpoint_training_details_dir, f'latest_details.npz'),
                    epoch=epoch,
                    is_epoch_finished=(i == len(train_data_loader) - 1),
                    global_step=global_step,
                    learning_rate=learning_rate
                )
    else:
        scaler = GradScaler(enabled=mixed_precision)
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        def get_losses(required_data, required_condition):
            with autocast(enabled=mixed_precision):
                losses = calculate_loss(required_data, required_condition) / gradient_accumulation_steps
                return losses
        
        def backpropagate(losses, i):
            loss = losses.mean()
            # loss.backward()
            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone_model.parameters(), grad_clip)
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                update_if_ema(backbone_model)
            return loss.item()
        
        def save(model: nn.Module):
            checkpoint_name = get_checkpoint_saving_name(epoch, start_step)
            torch.save(model.state_dict(), os.path.join(
                checkpoints_dir, f'{checkpoint_name}.pt'))
            optim_details = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict()
            }
            torch.save(optim_details, os.path.join(training_optim_details_dir, f'latest_optim.pt'))
            save_ema_if_enabled(checkpoint_name)
            np.savez_compressed(
                os.path.join(checkpoint_training_details_dir, f'latest_details.npz'),
                epoch=epoch,
                is_epoch_finished=(i == len(train_data_loader) - 1),
                global_step=global_step,
                learning_rate=learning_rate
            )
    
    def calculate_loss(required_data, required_condition):
        loss = diffuser(*required_data, additional_residuals=get_adaptation(adapter, required_condition))
        return loss
    
    if start_step is None:
        global_step = 0
    else:
        global_step = start_step
    step_interval_loss = 0
    # start training
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        with tqdm(train_data_loader, dynamic_ncols=True) as tqdm_data_loader:
            for i, data in enumerate(tqdm_data_loader):
                required_data = data_getter(data, device)
                backbone_zero_grad_if_enable_adapter(backbone_model)
                losses = get_losses(required_data, adapter_condition_getter(data, device))
                loss = backpropagate(losses, i)
                
                epoch_loss += loss
                step_interval_loss += loss
                learning_rate = lr_scheduler.get_last_lr()[0] # optimizer.state_dict()['param_groups'][0]["lr"]
                if start_step is not None:
                    start_step += required_data[0].shape[0]
                tqdm_data_loader.set_postfix(ordered_dict={
                    "Epoch": epoch,
                    "Loss": loss,
                    "Step": global_step,
                    "LR": learning_rate
                })
                writer.add_scalar('Training Loss (Step)', loss, global_step)
                global_step += 1
                if global_step % step_interval == 0:
                    step_interval_loss /= step_interval
                    step_interval_loss = 0
                    save(select_saving_model(backbone_model, adapter))
        
        epoch_loss /= i + 1
        writer.add_scalar('Training Loss', epoch_loss, epoch)

def _init_checkpoint_filenames(ckpt_filenames, start_epoch, start_step):
    start_index = 0
    if start_step is None:
        def init_start_index(filename):
            nonlocal start_index
            if checkpoint_sorted_key(filename) == start_epoch:
                nonlocal init_start_index
                init_start_index = lambda _: None
            else:
                start_index += 1
    else:
        def init_start_index(filename):
            nonlocal start_index
            if checkpoint_sorted_key(filename) == (start_epoch, start_step):
                nonlocal init_start_index
                init_start_index = lambda _: None
            else:
                start_index += 1
    ckpt_filenames = [(filename, init_start_index(filename))[0] for filename in ckpt_filenames if filename.endswith('.pt')]
    return start_index, ckpt_filenames

def val(
    sampler: nn.Module,
    adapter: Optional[nn.Module],
    device: torch.device,
    data_getter: Callable,
    adapter_condition_getter: Optional[Callable[[torch.Tensor, torch.device], tuple]],
    comparison_getter: Callable,
    initial_state_getter: Callable,
    initial_noise_strength: float,
    sampled_images_transform: Callable,
    data_loader: DataLoader,
    mixed_precision: bool = False,
    start_epoch: int = 0,
    start_step: Optional[int] = 0,
    checkpoints_dir: str = './checkpoints'
):
    ckpt_filenames = sorted(os.listdir(checkpoints_dir), key=checkpoint_sorted_key)
    start_index, ckpt_filenames = _init_checkpoint_filenames(ckpt_filenames, start_epoch, start_step)
    ckpt_filenames = ckpt_filenames[start_index:]
    
    ssim = SSIMMetric(spatial_dims=2, data_range=1.0, win_size=7)
    psnr = PSNRMetric()
    
    best_ssim = [0, '', 0]
    best_psnr = [0, '', 0]
    
    if adapter_condition_getter is None:
        def blank(_, __): return tuple()
        adapter_condition_getter = blank
    
    if adapter is None:
        def get_adaptation(adapter: None, *_):
            return None
        main_model = sampler.model
    else:
        def get_adaptation(adapter: nn.Module, required_condition):
            return adapter(*required_condition, need_tokens=True)
        main_model = adapter
        te.refine_model(sampler.model).eval()
    
    data_loader_length = len(data_loader)
    
    with tqdm(ckpt_filenames, dynamic_ncols=True) as tqdm_ckpt_filenames:
        for ckpt_filename in tqdm_ckpt_filenames:
            ckpt = torch.load(os.path.join(checkpoints_dir, ckpt_filename), map_location=device)
            te.refine_model(main_model).load_state_dict(ckpt)
            te.refine_model(main_model).eval()
            ssim_arr = np.empty(data_loader_length, dtype=object)
            psnr_arr = np.empty(data_loader_length, dtype=object)
            with torch.inference_mode():
                with autocast(enabled=mixed_precision):
                    for i, data in enumerate(tqdm(data_loader, leave=False)):
                        data = next(iter(data_loader))
                        initial_state = initial_state_getter(data, device)
                        noisy_images = torch.randn(size=initial_state.shape, device=device)
                        sampled_imgs = sampler(noisy_images, *data_getter(data, device), additional_residuals=get_adaptation(adapter, adapter_condition_getter(data, device)), initial_state=initial_state, strength=initial_noise_strength)
                        sampled_imgs = sampled_images_transform(sampled_imgs)
                        # sampled_imgs = sampled_imgs * 0.5 + 0.5 # [0 ~ 1]
                        sampled_imgs = te.map_range(sampled_imgs, dim=tuple(np.arange(1, len(noisy_images.shape))))
                        original_image = te.map_range(comparison_getter(data, device), dim=tuple(np.arange(1, len(noisy_images.shape))))
                        ssim_arr[i] = ssim(sampled_imgs, original_image)
                        psnr_arr[i] = psnr(sampled_imgs, original_image)
                    ssim_value = torch.vstack(tuple(ssim_arr)).mean()
                    psnr_value = torch.vstack(tuple(psnr_arr)).mean()
                    if ssim_value > best_ssim[0]:
                        best_ssim[0] = ssim_value
                        best_ssim[1] = ckpt_filename
                        best_ssim[2] = psnr_value
                    if psnr_value > best_psnr[0]:
                        best_psnr[0] = psnr_value
                        best_psnr[1] = ckpt_filename
                        best_psnr[2] = ssim_value
                tqdm.write(f'{ckpt_filename} - SSIM: {ssim_value}, PSNR: {psnr_value}')
        print('BEST SSIM RESULTS:')
        print(f'{best_ssim[1]} - SSIM: {best_ssim[0]}, PSNR: {best_ssim[2]}')
        print('BEST PSNR RESULT:')
        print(f'{best_psnr[1]} - PSNR: {best_psnr[0]}, SSIM: {best_psnr[2]}')

def test(
    sampler: nn.Module,
    adapter: Optional[nn.Module],
    device: torch.device,
    data_getter: Callable,
    adapter_condition_getter: Optional[Callable[[torch.Tensor, torch.device], tuple]],
    comparison_getter: Callable,
    initial_state_getter: Callable,
    initial_noise_strength: float,
    sampled_images_transform: Callable,
    data_loader: DataLoader,
    sampled_images_saving_path: str = './',
    mixed_precision: bool = False,
    n_rows: int = 8
):
    ssim = SSIMMetric(spatial_dims=2, data_range=1.0, win_size=7)
    psnr = PSNRMetric()
    
    if adapter_condition_getter is None:
        def blank(_, __): return tuple()
        adapter_condition_getter = blank
    
    if adapter is None:
        def get_adaptation(adapter: None, *_):
            return None
    else:
        def get_adaptation(adapter: nn.Module, required_condition):
            return adapter(*required_condition, need_tokens=True)
    
    data_loader_length = len(data_loader)
    ssim_arr = np.empty(data_loader_length, dtype=object)
    psnr_arr = np.empty(data_loader_length, dtype=object)
    
    def in_the_iter(i, data):
        nonlocal ssim_arr
        nonlocal psnr_arr
        initial_state = initial_state_getter(data, device)
        noisy_images = torch.randn(size=initial_state.shape, device=device)
        # saveNoisy = torch.clamp(noisy_images * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampled_imgs = sampler(noisy_images, *data_getter(data, device), additional_residuals=get_adaptation(adapter, adapter_condition_getter(data, device)), initial_state=initial_state, strength=initial_noise_strength)
        sampled_imgs = sampled_images_transform(sampled_imgs)
        # sampled_imgs = sampled_imgs * 0.5 + 0.5 # [0 ~ 1]
        sampled_imgs = te.map_range(sampled_imgs, dim=tuple(np.arange(1, len(noisy_images.shape))))
        # torch.save(sampled_imgs, 'dm.t')
        original_image = comparison_getter(data, device)[0]
        original_image = te.map_range(original_image, dim=tuple(np.arange(1, len(noisy_images.shape))))
        # torch.save(original_image,'ori.t')
        ssim_arr[i] = ssim(sampled_imgs, original_image)
        psnr_arr[i] = psnr(sampled_imgs, original_image)

        return sampled_imgs
    
    # load model and evaluate
    with torch.inference_mode():
        # Sampled from standard normal distribution
        with autocast(enabled=mixed_precision):
            for i, data in enumerate(tqdm(data_loader)):
                sampled_imgs = in_the_iter(i, data)

        ssim_value = torch.vstack(tuple(ssim_arr))
        psnr_value = torch.vstack(tuple(psnr_arr))
        print(f'SSIM: {ssim_value.mean()} ± {ssim_value.std()}, PSNR: {psnr_value.mean()} ± {psnr_value.std()}')
        
        os.makedirs(os.path.dirname(sampled_images_saving_path), exist_ok=True)
        save_image(sampled_imgs, sampled_images_saving_path, nrow=n_rows)

def ema_existing_checkpoints(
    backbone_model: nn.Module,
    ema_model: ExponentialMovingAverage,
    device: torch.device,
    start_epoch: int = 0,
    start_stepped_batch: int = 0,
    checkpoints_dir: str = './checkpoints'
):
    ckpt_filenames = sorted(os.listdir(checkpoints_dir), key=checkpoint_sorted_key)
    start_index, ckpt_filenames = _init_checkpoint_filenames(ckpt_filenames, start_epoch, start_stepped_batch)
    
    saving_dir = os.path.join(checkpoints_dir, 'ema')
    os.makedirs(saving_dir, exist_ok=True)
    
    def ema_update(ckpt_filename, backbone_model: nn.Module, ema_model: ExponentialMovingAverage):
        ckpt = torch.load(os.path.join(
            checkpoints_dir, ckpt_filename), map_location=device)
        backbone_model.load_state_dict(ckpt)
        backbone_model.train()
        ema_model.update_parameters(backbone_model)
    
    ema_update(ckpt_filenames[start_index], backbone_model, ema_model)
    start_index += 1
    
    with tqdm(total=len(ckpt_filenames)) as pbar:
        def ema_from_checkpoint(ckpt_filename):
            nonlocal backbone_model
            nonlocal ema_model
            nonlocal start_index
            ema_update(ckpt_filename, backbone_model, ema_model)
            torch.save(ema_model.module.state_dict(), os.path.join(
                saving_dir, 'ckpt_' + str(start_index) + "_.pt"))
            start_index += 1
            pbar.update(1)
        
        ckpt_filenames = ckpt_filenames[start_index:]
        np.vectorize(ema_from_checkpoint, otypes=[])(ckpt_filenames)