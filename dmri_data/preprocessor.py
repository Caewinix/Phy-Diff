import numpy as np
import os
import zipfile
import shutil
from typing import Optional, Callable
import ants
import SimpleITK as sitk
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from monai.metrics import SSIMMetric
from evaluation import PSNRMetric
from .loader import DmriNpkDataset, BaseMriNpyDataset, AdcAtlasesNpkDataset
import numpy_ext as npe
import torch_ext as te


def get_image(path: str):
    return sitk.GetArrayFromImage(
        sitk.ReadImage(path)
    )


def preprocess_xtract_atlases(tract_atlases_dir: str, saving_path: str = './preprocessed/xtract_atlases.npy'):
    tract_atlas_filenames = sorted(os.listdir(tract_atlases_dir))
    tract_atlas_filenames = np.array([filename for filename in tract_atlas_filenames if filename.endswith('.nii.gz')])
    tract_atlas_image = np.expand_dims(get_image(os.path.join(tract_atlases_dir, tract_atlas_filenames[0])), axis=1)
    def process_each_tract(filename: str):
        nonlocal tract_atlas_image
        tract_atlas_image = np.concatenate([tract_atlas_image, np.expand_dims(get_image(filename), axis=1)], axis=1)
    np.vectorize(lambda filename: process_each_tract(os.path.join(tract_atlases_dir, filename)), otypes=[])(tract_atlas_filenames[1:])
    np.save(saving_path, tract_atlas_image)


def preprocess_dmri_data(
    data_dir: str,
    saving_dir: str = './preprocessed',
    # unique_b_values_save_path: str = './preprocessed/unique_b_values.npy',
    save_prefix: str = '',
    registration_ref_path: str | None = None,
    position_slice: slice | int | None = 110,
    b_value_mean_max_limit: int | None = None,
    base_correction: int = 5
):
    data_filenames = np.array(sorted(os.listdir(data_dir)))
    diffusion_save_dir = os.path.join(saving_dir, 'diffusion_images')
    base_save_dir = os.path.join(saving_dir, 'base_images')
    adc_atlases_save_dir = os.path.join(saving_dir, 'dmri_adc_atlases')
    os.makedirs(diffusion_save_dir, exist_ok=True)
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(adc_atlases_save_dir, exist_ok=True)
    temp_dir = os.path.join(saving_dir, 'tmp')
    # Get numpy format of a particular nii file.
    if registration_ref_path is None:
        def get_content(diffusion_path: str, t1w_path: str, mask_path:str=None):
            if mask_path is not None:
                mask = get_image(mask_path)
                return get_image(diffusion_path) * np.expand_dims(mask, 0), get_image(t1w_path) * mask
            return get_image(diffusion_path), get_image(t1w_path)
    else:
        def get_content(diffusion_path: str, t1w_path: str, mask_path: str=None):
            moving_mask = ants.image_read(mask_path, pixeltype='float')
            moving_mask_numpy = moving_mask.numpy()
            moving_t1w_img = ants.image_read(t1w_path, pixeltype='float')
            moving_t1w_img = ants.from_numpy(moving_t1w_img.numpy() * moving_mask_numpy,
                                        moving_t1w_img.origin,
                                        moving_t1w_img.spacing,
                                        moving_t1w_img.direction,
                                        moving_t1w_img.has_components,
                                        moving_t1w_img.is_rgb)
            moving_diffusion_img = ants.image_read(diffusion_path, pixeltype='float')
            moving_diffusion_img = ants.from_numpy(moving_diffusion_img.numpy() * np.expand_dims(moving_mask_numpy, -1),
                                        moving_diffusion_img.origin,
                                        moving_diffusion_img.spacing,
                                        moving_diffusion_img.direction,
                                        moving_diffusion_img.has_components,
                                        moving_diffusion_img.is_rgb)
            fixed_img = ants.image_read(registration_ref_path, pixeltype='float')
            reg_t1w_dict = ants.registration(fixed_img, moving_t1w_img, type_of_transform='Affine')
            reg_t1w_img = reg_t1w_dict['warpedmovout']
            fwdtransforms = reg_t1w_dict['fwdtransforms']
            for tmp_path in fwdtransforms:
                os.remove(tmp_path)
            moving_diffusion_img_single = ants.from_numpy(moving_diffusion_img.numpy()[:, :, :, 0],
                                        moving_diffusion_img.origin[:-1],
                                        moving_diffusion_img.spacing[:-1],
                                        moving_diffusion_img.direction[:-1, :-1].copy(),
                                        moving_diffusion_img.has_components,
                                        moving_diffusion_img.is_rgb)
            fwdtransforms = ants.registration(reg_t1w_img, moving_diffusion_img_single, type_of_transform='Affine')['fwdtransforms']
            reg_img = ants.apply_transforms(reg_t1w_img, moving_diffusion_img, transformlist=fwdtransforms, imagetype=3).numpy()
            reg_img = np.transpose(reg_img, np.arange(len(reg_img.shape))[::-1])
            reg_t1w_img = reg_t1w_img.numpy()
            reg_t1w_img = np.transpose(reg_t1w_img, np.arange(len(reg_t1w_img.shape))[::-1])
            for tmp_path in fwdtransforms:
                os.remove(tmp_path)
            return reg_img, reg_t1w_img
    def get_nii_array(diffusion_nii_path, t1w_nii_path: str, mask_nii_path=None):
        def move(nii_path):
            temp_ori_data_path = os.path.join(temp_dir, nii_path)
            temp_data_path = os.path.join(temp_dir, os.path.basename(nii_path))
            os.rename(temp_ori_data_path, temp_data_path)
            return temp_data_path
        img_nii_array, mask_nii_array = get_content(move(diffusion_nii_path), move(t1w_nii_path), move(mask_nii_path))
        shutil.rmtree(os.path.join(temp_dir, diffusion_nii_path.split('/')[0]))
        return img_nii_array, mask_nii_array
    # Start preprocessing.
    required_unique_b_values = np.empty((0,))
    subjects_count = len(data_filenames)
    total_base_imgs_arr = np.empty(subjects_count, dtype=object)
    total_diffusion_imgs_arr = np.empty(subjects_count, dtype=object)
    total_b_values_arr = np.empty(subjects_count, dtype=object)
    print('Preprocess each data...')
    with tqdm(total=len(data_filenames), dynamic_ncols=True) as pbar:
        subject_i = 0
        def preprocess_each_data_filename(data_filename: str):
            nonlocal subject_i
            with zipfile.ZipFile(os.path.join(data_dir, data_filename), 'r') as z:
                subject_b_values, subject_b_vectors, images_path, masks_path, t1w_images_path = None, None, None, None, None
                for file_path in z.namelist():
                    if file_path.endswith('bvals'):
                        with z.open(file_path, 'r') as file:
                            lines = file.readlines()
                            subject_b_values = np.array([list(map(float, line.split())) for line in lines]).T
                    elif file_path.endswith('bvecs'):
                        with z.open(file_path, 'r') as file:
                            lines = file.readlines()
                            subject_b_vectors = np.array([list(map(float, line.split())) for line in lines]).T
                    elif file_path.endswith('data.nii.gz'):
                        images_path = file_path
                        z.extract(images_path, temp_dir)
                    elif file_path.endswith('nodif_brain_mask.nii.gz'):
                        masks_path = file_path
                        z.extract(masks_path, temp_dir)
                    elif file_path.endswith('T1w_acpc_dc_restore_1.25.nii.gz'):
                        t1w_images_path = file_path
                        z.extract(t1w_images_path, temp_dir)
                    elif subject_b_values is not None and subject_b_vectors is not None and images_path is not None and masks_path is not None and t1w_images_path is not None:
                        break
            subject_b_vectors = subject_b_vectors * subject_b_values / np.sqrt(np.sum(np.square(subject_b_vectors), axis=1).reshape(-1, 1))
            subject_imgs, _ = get_nii_array(images_path, t1w_images_path, masks_path)
            
            base_index_mask = subject_b_values.flatten() == base_correction
            non_base_index_mask = np.logical_not(base_index_mask)
            subject_b_values = subject_b_values[non_base_index_mask]
            subject_base_imgs = np.mean(subject_imgs[base_index_mask], axis=0)
            subject_imgs = subject_imgs[non_base_index_mask]
            
            unique_values, count = np.unique(subject_b_values, return_counts=True)
            total_base_imgs_arr[subject_i] = np.expand_dims(subject_base_imgs, 0)
            total_diffusion_imgs_arr[subject_i] = np.expand_dims(subject_imgs, 0)
            total_b_values_arr[subject_i] = np.expand_dims(subject_b_values, 0)
            if b_value_mean_max_limit is not None:
                count_temp = b_value_mean_max_limit - count
                less_than_max_mask = count_temp >= 0
                less_than_max_diff = count_temp[less_than_max_mask]
                additional_num_for_max = np.sum(less_than_max_diff) // (len(count) - len(less_than_max_diff))
                unique_values_mean_new_count = additional_num_for_max + b_value_mean_max_limit
                less_than_max_indexes = np.where(less_than_max_mask)[0]
                required_indexes = np.empty((0,), dtype=int)
                for i in range(len(unique_values)):
                    value_indexes = np.where(unique_values[i] == subject_b_values)[0]
                    value_indexes_count = len(value_indexes)
                    if (unique_values_mean_new_count >= value_indexes_count) or (i in less_than_max_indexes):
                        required_indexes = np.concatenate([required_indexes, value_indexes])
                    else:
                        required_indexes = np.concatenate([required_indexes,
                                                            value_indexes[np.sort(np.random.choice(value_indexes_count,
                                                                                unique_values_mean_new_count,
                                                                                replace=False))]])
                subject_b_values = subject_b_values[required_indexes]
                subject_b_vectors = subject_b_vectors[required_indexes]
                subject_imgs = subject_imgs[required_indexes]
            
            nonlocal required_unique_b_values
            required_unique_b_values = np.concatenate([required_unique_b_values, unique_values.flatten()])
            nonlocal position_slice
            if position_slice is None:
                position_slice = slice(None, None, None)
            #     pos_start_index = 0
            else:
                if isinstance(position_slice, int):
                    pos_start_index = (subject_imgs.shape[1] - position_slice) // 2
                    pos_end_index = pos_start_index + position_slice
                    position_slice = slice(pos_start_index, pos_end_index)
            subject_imgs = subject_imgs[:, position_slice, :, :]
            subject_base_imgs = subject_base_imgs[position_slice, :, :]

            filename_base = os.path.splitext(data_filename)[0]
            np.save(os.path.join(base_save_dir, save_prefix + f'{filename_base}.npy'), subject_base_imgs)
            
            npe.savek(
                os.path.join(diffusion_save_dir, save_prefix + f'{filename_base}.npk'),
                images=subject_imgs,
                b_vectors=subject_b_vectors,
                b_values=subject_b_values)
            
            subject_i += 1
            pbar.update(1)
        np.vectorize(preprocess_each_data_filename, otypes=[])(data_filenames)
    shutil.rmtree(temp_dir)
    required_unique_b_values = np.unique(required_unique_b_values)
    # np.save(unique_b_values_save_path, required_unique_b_values)
    total_diffusion_imgs_arr = np.vstack(total_diffusion_imgs_arr)
    total_base_imgs_arr = np.expand_dims(np.vstack(total_base_imgs_arr), 1)
    total_b_values_arr = np.vstack(total_b_values_arr)
    total_b_values_arr = total_b_values_arr.reshape(*total_b_values_arr.shape[:2], 1, 1, 1)
    adc = np.zeros_like(total_diffusion_imgs_arr)
    logarithmic_mask = (total_diffusion_imgs_arr > 0) & (total_base_imgs_arr > 0)
    total_base_imgs_arr = np.broadcast_to(total_base_imgs_arr, total_diffusion_imgs_arr.shape)
    adc[logarithmic_mask] = np.log(total_base_imgs_arr[logarithmic_mask] / total_diffusion_imgs_arr[logarithmic_mask])
    adc /= total_b_values_arr
    total_b_values_arr = total_b_values_arr.reshape(total_b_values_arr.shape[:2])
    adc[adc < 0] = 0
    adc = adc[:, :, position_slice, :, :]
    print('Making ADC atlases of various b-values...')
    adc_atlases = np.empty((len(required_unique_b_values), *adc.shape[2:]))
    for i, b_value in enumerate(tqdm(required_unique_b_values)):
        adc_atlases[i] = np.mean(adc[b_value == total_b_values_arr], axis=0)
    npe.savek(
        os.path.join(adc_atlases_save_dir, save_prefix + 'adc_atlases.npk'),
        images=adc_atlases,
        b_values=required_unique_b_values)


def preprocess_dmri_unique_b_values(
    data_dir: str,
    unique_b_values_save_path: str = './preprocessed/unique_b_values.npy'
):
    data_filenames = np.array(sorted(os.listdir(data_dir)))
    # Start preprocessing.
    required_unique_b_values = np.empty((0,))
    with tqdm(total=len(data_filenames), dynamic_ncols=True) as pbar:
        subject_i = 0
        def preprocess_each_data_filename(data_filename: str):
            nonlocal subject_i
            with zipfile.ZipFile(os.path.join(data_dir, data_filename), 'r') as z:
                for file_path in z.namelist():
                    if file_path.endswith('bvals'):
                        with z.open(file_path, 'r') as file:
                            lines = file.readlines()
                            subject_b_values = np.array([list(map(float, line.split())) for line in lines]).T
                        break
            unique_values = np.unique(subject_b_values)
            nonlocal required_unique_b_values
            required_unique_b_values = np.concatenate([required_unique_b_values, unique_values.flatten()])
            pbar.update(1)
        npe.apply_from_axis(preprocess_each_data_filename, data_filenames, axis=0, otypes=[])
        np.save(unique_b_values_save_path, np.unique(required_unique_b_values))