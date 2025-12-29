import os
import json
import pickle
import glob
import traceback
import sys
from collections import defaultdict
from einops import rearrange, repeat 
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Union, Callable

from seva.geometry import get_plucker_coordinates
from sgm.data.read_write_model import read_model
from sgm.data.utils_camera import (
    read_intrinsics_colmap,
    read_extrinsics_colmap,
    read_intrinsics_nerfstudio,
    read_extrinsics_nerfstudio,
    opencv_to_opengl,
    colmap_to_nerfstudio,
    nerfstudio_to_colmap
)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import pytorch_lightning as pl
from seva.data.preprocessing import (
    update_intrinsics,
    create_transform_matrix,
    get_bbox_center_and_size,
    get_mvhumannet_extrinsics,
    load_json,
    load_pickle,
    update_intrinsics_resize,
    generate_gaussian_mixture_samples,
    generate_gaussian_samples,
    normalize_intrinsics
)
import time
from seva.data.cropper import RandomBBoxCropper
from seva.modules.autoencoder import AutoEncoder
import torchvision
import h5py
from datasets import load_from_disk

# NOTE: hardcoded camera order for each camera elevation (counter clockwise)
# use for trajectory NVS training!
# Camera IDs organized by rung elevation
TOP_RUNG = [
    'CC32871A043', 'CC32871A018', 'CC32871A012', 'CC32871A021',
    'CC32871A060', 'CC32871A006', 'CC32871A042', 'CC32871A041', 
    'CC32871A049', 'CC32871A036', 'CC32871A047', 'CC32871A019',
    'CC32871A020', 'CC32871A056', 'CC32871A009', 'CC32871A014'
]

MIDDLE_RUNG = [
    'CC32871A005', 'CC32871A033', 'CC32871A050', 'CC32871A059',
    'CC32871A017', 'CC32871A034', 'CC32871A032', 'CC32871A052',
    'CC32871A039', 'CC32871A058', 'CC32871A013', 'CC32871A004',
    'CC32871A044', 'CC32871A031', 'CC32871A055', 'CC32871A029'
]

BOTTOM_RUNG = [
    'CC32871A035', 'CC32871A016', 'CC32871A030', 'CC32871A038',
    'CC32871A023', 'CC32871A027', 'CC32871A051', 'CC32871A015',
    'CC32871A022', 'CC32871A057', 'CC32871A048', 'CC32871A008',
    'CC32871A046', 'CC32871A010', 'CC32871A040', 'CC32871A037'
]

CAMERA_RUNGS = [TOP_RUNG, MIDDLE_RUNG, BOTTOM_RUNG]
ALL_CAMERAS = sorted([cam for rung in CAMERA_RUNGS for cam in rung])
CAMERA_TO_INDEX = {cam: idx for idx, cam in enumerate(ALL_CAMERAS)}

# borrowed from @dataset.py
def center_cameras(all_c2ws, c2ws):
    # finds mean position of all_c2ws, then centers cameras by subtracting the mean
    ref_c2ws = all_c2ws
    camera_dist_2med = torch.norm(
        ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    c2ws[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
    

def scale_cameras(c2ws, camera_scale=2.0):
    camera_dists = c2ws[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    c2ws[:, :3, 3] *= translation_scaling_factor


def read_from_hdf5(hdf5_file, *args):
    """
    Read data from HDF5 file.
    """
    # each arg will be nested keys
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Navigate through nested keys
            current = f
            for arg in args:
                current = current[arg]
            
            # Read the data into memory before closing the file
            if isinstance(current, h5py.Dataset):
                # For datasets, read the actual data
                return np.array(current)
            elif isinstance(current, h5py.Group):
                # For groups, return a dict of the group structure
                return {key: current[key] for key in current.keys()}
            else:
                return current
    except KeyError:
        return None
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None


def one_hot_encode_segmentation(seg_map: torch.Tensor, num_classes: int, classes_to_use: list = []) -> torch.Tensor:
    """
    Converts a segmentation label map to a one-hot encoded tensor.

    Args:
        seg_map (torch.Tensor): The segmentation map. 
                                Expected shape (1, H, W) or (H, W).
                                Must contain class indices (e.g., 0, 1, ... N-1).
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: The one-hot encoded tensor of shape (num_classes, H, W).
    """
    if len(classes_to_use) > 0:
        seg_map_ = seg_map[classes_to_use]
    else: # otherwise, use all classes if empty
        seg_map_ = seg_map
    # Squeeze out the channel dim if it exists, (1, H, W) -> (H, W)
    if seg_map_.dim() == 3 and seg_map_.shape[0] == 1:
        seg_map_ = seg_map_.squeeze(0)
    
    seg_map_long_ = seg_map_.long()
    one_hot = F.one_hot(seg_map_long_, num_classes=num_classes)
    one_hot_ = one_hot.permute(2, 0, 1)
    return one_hot_.float()

class MVHumanNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_images,
        latents_dir=None,
        transforms=None,
        pre_scale_intrinsics=0.5,
        data_limit=None,
        only_include=None,
        exclude=None,
        random_crop=False,
        maximal_crop=False,
        white_background=False,
        step_size=60,
        preload_path=None,
        iclight_dataset_path=None,
        infu_dataset_path=None,
        face_bbox_dir=None,
        arcface_embeddings_dir=None,
        crop_padding=60, # used to prevent clipping of the humans
        use_inconsistent=False,
        random_crop_prob=0.3, # probability of using random crop over maximal
        ic_sampling_prob=0.7, # probability of randomly sampling from InfU over IC light
        fixed_sampling_ids=None,
        concatenate_sapiens_conditioning=None, # list of "depth, seg, latents" later
        sapiens_mask_loss_types=[], # list of "depth, seg, latents" later (for loss)
        sapiens_segmentation_channels_to_use=[], # face
        face_crop_prob=0.5, # probability of biasing crop towards face when face bbox available
        face_bias_strength=0.7, # strength of face bias (0.0=no bias, 1.0=fully centered on face)
    ):
        self.root_dir = root_dir             # directory of all subject directories
        self.latents_dir = latents_dir       # directory of all latents
        self.num_images = num_images         # context window T
        self.transforms = transforms         # transforms for the random crop
        self.pre_scale_intrinsics = pre_scale_intrinsics           # since MVHumanNet is downsampled, update intrinsics
        self.only_include = set(only_include) if only_include is not None else None     # TEMP -- include only these subjects (as List of strings)
        self.exclude = set(exclude) if exclude is not None else None               # TEMP -- exclude these subjects (as List of strings)
        self.data_limit = data_limit         # TEMP -- only get the first 'data_limit' (int) subjects
        self.step_size = step_size           # only processes every 'step_size' frames (timesteps)
        self.random_crop = random_crop       # NOTE: this is the toggle for probabilistic cropping 
                                             # ! unrelated to initial crop from crop_params.json
                                             # ! (human-centered 576x576 image crop) 

        self.random_crop_prob = random_crop_prob
        self.maximal_crop = maximal_crop     # initial crops to the human based on annots
        # NOTE: if the above is set to True, then latents_dir will be ignored
        # and latents will be computed on the fly!
        self.use_inconsistent = use_inconsistent
        self.ic_sampling_prob = ic_sampling_prob
        # if True, then will concatenate all clean latents with these ic latents
        # if False, then will leave conditioning "black" for target images
        # and will repeat the clean latent for the input images
        self.concatenate_sapiens_conditioning = concatenate_sapiens_conditioning
        assert self.concatenate_sapiens_conditioning is None or all(cond in ["depth", "seg_masks", "latents"] for cond in self.concatenate_sapiens_conditioning), "Invalid sapiens conditioning!"
        self.sapiens_segmentation_channels_to_use = sapiens_segmentation_channels_to_use
        self.sapiens_mask_loss_types = sapiens_mask_loss_types
        self.fixed_sampling_ids = fixed_sampling_ids
        self.adjacent_frame_sampling_prob = 0.2 # Trajectory NVS acceptance rate
        self.all_inputs_prob = 0.85
        self.white_background = white_background
        self.preload_path = preload_path
        self.iclight_dataset_path = iclight_dataset_path # IC-light output directory
        self.infu_dataset_path = infu_dataset_path # InfU output directory
        self.face_bbox_dir = face_bbox_dir # Face bounding box directory
        self.arcface_embeddings_dir = arcface_embeddings_dir # ArcFace embeddings directory
        self.face_crop_prob = face_crop_prob
        self.face_bias_strength = face_bias_strength
        # NOTE: currently we only have arcface_embeddings for MVHN gt dataset
    
        # if not None, will use the "phase 2" expected training process
        self.infu_num_images = {} # Dict[subject_id: int] number of images in infu directory
        if self.infu_dataset_path is not None:
            subjects_to_parse = os.listdir(self.infu_dataset_path)
            for subject_id in subjects_to_parse:
                if self.exclude is not None and subject_id in self.exclude:
                    continue
                if self.only_include is not None and subject_id not in self.only_include:
                    continue
                self.infu_num_images[subject_id] = len(os.listdir(os.path.join(self.infu_dataset_path, subject_id))) - 2
                # -2 is a HACK to avoid I/O checking; this accounts for the npz masks

        if self.num_images > 16: # if more than 16, disable trajectory NVS batching
            self.adjacent_frame_sampling_prob = 0.0
        self.crop_padding = crop_padding
        # actual data
        self.cam_params = {} # Dict[subject: (extrinsics, intrinsics, camera_scale)]
        self.face_bboxes = self._load_face_bboxes() if face_bbox_dir is not None else None # * needs to be loaded BEFORE scenes
        
        # Detect if preload_path is an Arrow dataset
        self.is_arrow = False
        self.dataset = None
        if self.preload_path and (self.preload_path.endswith('.arrow') or os.path.isdir(self.preload_path)):
            # Check if it looks like an Arrow dataset (directory with metadata.json or similar)
            if os.path.exists(os.path.join(self.preload_path, 'dataset_info.json')) or \
               os.path.exists(os.path.join(self.preload_path, 'state.json')) or \
               len(glob.glob(os.path.join(self.preload_path, '*.parquet'))) > 0:
                self.is_arrow = True
                print(f"Detected Arrow dataset at {self.preload_path}")
        
        if self.is_arrow:
            self.scenes = self._load_scenes_arrow()
        else:
            self.scenes = self._load_preloaded_filepaths()
            
        self.image_shape = (1500, 2048) # MVHumanNet images are 2048x1500

        # from SD 2.1 VAE
        self.downsample_factor = 8
        self.scale_factor = 0.18215              
        self.target_shape = (576, 576)
        self.latent_shape = (self.num_images, 4, self.target_shape[0] // self.downsample_factor, 
                          self.target_shape[1] // self.downsample_factor)

        if self.transforms is None:
            # default (no probabilistic crop), only CenterCrop
            self.transform = T.Compose([
                T.CenterCrop(self.image_shape[0]), # Center crop to square
                T.Resize(self.target_shape),       # Resize to target shape
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),                      # Convert to tensor
                T.Normalize([0.5], [0.5])          # Normalize to [-1, 1]
            ])
            self.mask_transform = T.Compose([
                T.CenterCrop(self.image_shape[0]), # Center crop to square
                T.Resize(self.target_shape),       # Resize to target shape
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),  # Convert to tensor, keep in [0, 1]
            ])

        if self.random_crop or self.maximal_crop:
            self.cropper = RandomBBoxCropper(
                random_crop=self.random_crop,
                random_crop_prob=self.random_crop_prob,
                padding=self.crop_padding,
                face_crop_prob=self.face_crop_prob,
                face_bias_strength=self.face_bias_strength
            )
            self.transform = T.Compose([
                T.Resize(self.target_shape),
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                T.Normalize([0.5], [0.5])
            ])
            self.mask_transform = T.Compose([
                T.Resize(self.target_shape),       # Resize to target shape
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),  # Convert to tensor, keep in [0, 1]
            ])

        if self.pre_scale_intrinsics != 0.5:
            print("WARNING: pre_scale_intrinsics is not 0.5, which is expected for MVHumanNet!")
        print("MVHN::init done!")


    def _clean_camera_keys(self, data):
        # Create new dictionary with cleaned keys
        cleaned_data = {}
        for key, value in data.items():
            # Extract just the camera ID number
            camera_id = key[2:-4] # remove "1_" and ".png" from camera_extrinsics.json
            cleaned_data[camera_id] = value
        return cleaned_data

    def _load_face_bboxes(self):
        """
        Load face bboxes from a directory of sharded jsons, each structured as:
        {
            "<subject_id>": {
                "<camera_id>": {
                    "<timestep>": {
                        "x1, y1, x2, y2, ley_eye, right_eye, left_lip, right_lip" keys
                        (NOTE: x1, y1, x2, y2 are int, all others are 2-tuples of ints
                    }...
                }
            }
        }
        NOTE: assumes that subject keys are all UNIQUE (should be true)
        """
        all_face_info = {}
        for face_json in os.listdir(self.face_bbox_dir):
            all_face_info.update(load_json(os.path.join(self.face_bbox_dir, face_json)))

        return all_face_info

    def _read_arcface_embeddings(self, *args):
        """
        Reads from chosen HDF5 file
        """
        # * UPDATE: lazy load from HDF5
        return read_from_hdf5(os.path.join(self.arcface_embeddings_dir, "arcface_embeddings_merged.hdf5"), *args)

    def _load_preloaded_filepaths(self):
        """
        Load preloaded filepaths from a custom json file structured as:
        WARNING: relies on accurate priors that all subjects have the exact same structure!
        {
            "subject_id": {
                "timesteps": <int: number of timesteps within first camera> (or list)
                "cameras": <list: all camera IDs> (validated prior)
                "extrinsics": <dict: camera extrinsics>
                "intrinsics": <list: camera intrinsics>
                "camera_scale": <float: camera scale>
                "annots": {"bbox": <list: bbox coords.>, "bbox_face": <list: face bbox coords. NOTE: not used since they're unreliable>}
            }
        }
        The above structure gives all that we need to load the filepaths (reducing metadata reads by a lot!)
        """
        assert self.preload_path is not None, "Preload path must be provided!"
        print("Loading preloaded filepaths...")
        preload_path = self.preload_path
        subjects = load_json(preload_path) # preloaded subject data
        scenes = []

        subjects_with_latents = None
        if self.latents_dir is not None:
            subjects_with_latents = set([subject for subject in os.listdir(self.latents_dir) if os.path.exists(os.path.join(self.latents_dir, subject, f"{subject}.npz"))])
            print(f"Found {len(subjects_with_latents)} subjects with latents")

        # for each subject (key) in the preloaded data (should be of available latents and metadata):
        for i, subject in tqdm(enumerate(subjects), total=len(subjects), desc="Loading scenes"):
            if subject == "metadata":
                continue
            subject_path = os.path.join(self.root_dir, subject)
            if self.only_include is not None and subject not in self.only_include:
                continue
            if self.exclude is not None and subject in self.exclude:
                continue  # exclude the given subjects
            if self.data_limit is not None and i >= self.data_limit:
                break
            if len(subjects[subject]['cameras']) != 48:
                print(f"Skipping subject {subject} because it does not have all 48 cameras!")
                continue # if no latents precomputed for this subject, or if not all cameras are present, then skip this
            if (subjects_with_latents is not None and subject not in subjects_with_latents):
                print(f"Skipping subject {subject} because it does not have latents precomputed!")
                continue # if no latents precomputed for this subject, or if not all cameras are present, then skip this

            extrinsics = subjects[subject]['extrinsics'] # should be pre-cleaned!
            intrinsics = subjects[subject]['intrinsics']['intrinsics']
            camera_scale = subjects[subject]['camera_scale']
            annots = subjects[subject]['annots']

            # for each subject, store camera parameters separately
            # ! NOTE: intrinsics need to be downscaled by 2 later!
            # ! AND extrinsics [t] needs to be scaled by camera_scale later!
            self.cam_params[subject] = {
                'extrinsics': extrinsics, # Dict[camera_id: extrinsic params]
                'intrinsics': intrinsics, # List[List] (turn to matrix)
                'camera_scale': camera_scale # float
            }

            # get image, mask, annots
            num_timesteps = subjects[subject]['timesteps'] # usually int, but this may be a LIST!
            # ensure only camera dirs were captured
            cameras = [cam for cam in subjects[subject]['cameras'] if cam in annots['bbox']]
            step_size = subjects["metadata"]["step_size"]
            subject_map = defaultdict(dict)

            # build frames_info for each timestep, camera combination
            # NOTE: num_timesteps is based on the FIRST camera; some subjects have differing numbers of timesteps for their cameras!
            iterator = range(1, num_timesteps, self.step_size) if isinstance(num_timesteps, int) else num_timesteps
            is_list_type = isinstance(num_timesteps, list)
            for timestep in iterator:
                try: # to get all cameras for this timestep (and ENSURE all cameras are present)
                    for camera in cameras:
                        if isinstance(timestep, str) and timestep.endswith("_img.jpg"):
                            timestep = int(timestep.split("_")[0])
                        time_id = f"{timestep * 5:04d}"
                        image_path = os.path.join(subject_path, "images_lr", camera, f"{time_id}_img.jpg")
                        mask_path = os.path.join(subject_path, "fmask_lr", camera, f"{time_id}_img_fmask.png")
                        # annots_path = os.path.join(subject_path, "annots", camera, f"{time_id}_img.json")
                        bbox = annots['bbox'][camera][time_id]
                        # Initialize defaults
                        face_bbox = [-1, -1, -1, -1]
                        arcface_embedding = None
                        
                        if self.face_bboxes is not None:
                            # Handle missing face bbox data gracefully
                            try:
                                face_bbox_dict = self.face_bboxes[subject][camera][f"{time_id}_img.jpg"]
                                if face_bbox_dict == {}:
                                    face_bbox = [-1, -1, -1, -1] # indicates no face detected
                                else:
                                    face_bbox = [face_bbox_dict['x1'], face_bbox_dict['y1'], face_bbox_dict['x2'], face_bbox_dict['y2']]
                            except KeyError:
                                # Subject/camera/timestep not in face_bboxes - tag as no face
                                face_bbox = [-1, -1, -1, -1]

                            if face_bbox != [-1, -1, -1, -1] and ((bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0):
                                print(f"Skipping subject {subject} camera {camera} timestep {timestep} because bbox is invalid")
                                continue

                        subject_map[time_id][camera] = {
                                    'image_path': image_path,
                                    'mask_path': mask_path,
                                    'annots': {
                                        'bbox': bbox,
                                        'bbox_face': face_bbox if self.face_bboxes is not None else None,
                                    }
                                }
                except Exception as e: # NOTE: this is a hack to ignore missing timesteps
                    print(f"Error loading subject {subject} camera {camera} timestep {timestep}: {e}")
                    subject_map.pop(time_id, None) # remove this timestep from the subject_map
                    break 

            sorted_timesteps = sorted(subject_map.keys())
            for i in range(0, len(sorted_timesteps)):
                timestep = sorted_timesteps[i]
                frames_info = subject_map[timestep]

                if len(frames_info.keys()) < self.num_images: # not enough cameras for this timestep to sample
                    continue # then skip this timestep

                scenes.append({
                    'subject_id': subject,
                    'frames_info': frames_info,
                    'timestep': timestep
                })
        print("Loading preloaded filepaths completed!")
        return scenes

    def _load_scenes_arrow(self):
        """
        Load scenes from Arrow dataset (memory mapped).
        Returns a list of scene metadata (subject_idx, timestep).
        """
        print("Loading Arrow dataset...")
        self.dataset = load_from_disk(self.preload_path)
        print(f"Loaded Arrow dataset with {len(self.dataset)} subjects")
        
        scenes = []
        
        subjects_with_latents = None
        if self.latents_dir is not None:
            subjects_with_latents = set([subject for subject in os.listdir(self.latents_dir) if os.path.exists(os.path.join(self.latents_dir, subject, f"{subject}.npz"))])
            print(f"Found {len(subjects_with_latents)} subjects with latents")
            
        # Create lightweight index
        for i in tqdm(range(len(self.dataset)), desc="Indexing Arrow dataset"):
            # We need to check subject_id first to apply filters
            # Accessing a single column is fast in Arrow
            subject_id = self.dataset[i]['subject_id']
            
            if self.only_include is not None and subject_id not in self.only_include:
                continue
            if self.exclude is not None and subject_id in self.exclude:
                continue
            if self.data_limit is not None and len(scenes) >= self.data_limit * 100: # approx limit (subjects * timesteps)
                 # This logic is slightly different from JSON loader which limits *subjects*
                 # But we can check if we've processed enough subjects
                 pass 
                 
            if (subjects_with_latents is not None and subject_id not in subjects_with_latents):
                # print(f"Skipping subject {subject_id} because it does not have latents precomputed!")
                continue

            # Get timesteps for this subject
            timesteps = self.dataset[i]['timesteps']
            
            # Apply step_size if needed (though likely already applied in dataset creation)
            # Use all available timesteps in the dataset
            # If the user wants to subsample further, we could do it here
            # But assuming dataset is prepared with desired step_size
            
            for timestep in timesteps:
                if isinstance(timestep, str) and timestep.endswith("_img.jpg"):
                    timestep = int(timestep.split("_")[0])
                
                time_id = f"{timestep * 5:04d}"                
                if isinstance(timestep, str):
                    # e.g. "0005_img.jpg"
                    time_id = timestep.split("_")[0] # "0005"
                else:
                    # Should not happen with Arrow dataset from preload_paths.py
                    time_id = f"{timestep:04d}"

                scenes.append({
                    'subject_id': subject_id,
                    'timestep': time_id,
                    'arrow_idx': i, # Store index to retrieve row later
                    'is_arrow': True
                })

        print(f"Loaded {len(scenes)} scenes from Arrow dataset")
        return scenes

    #! DEPRECATED: online reading of the dataset takes many hours per run just to load!
    #! therefore, only use preloaded filepaths.
    def _load_scenes(self):
        """
        NEW -- compact loading using priors:
        - all subjects are continuous between timesteps (no gaps)
        - all subjects have the same cameras
        - all subjects have the same number of timesteps
        - 
        For each subject in MVHumanNet, load dict:
        - frames_info: list of dicts, each with keys:
            - image_path
            - mask_path
            - annots (bbox, bbox_face)
        (Implicitly also updates self.cam_params)
        """
        scenes = []
        valid_latent_scenes = None
        if self.latents_dir is not None:    
            valid_latent_scenes = [
                subject for subject in os.listdir(self.latents_dir)
                if subject in os.listdir(self.root_dir)
            ]
        else:
            valid_latent_scenes = os.listdir(self.root_dir)

        for i, subject in tqdm(enumerate(valid_latent_scenes), total=len(valid_latent_scenes), desc="Loading scenes"):
            if self.data_limit is not None and i >= self.data_limit:
                break
            subject_path = os.path.join(self.root_dir, subject)  
            if not os.path.isdir(subject_path): # ignore non-directories
                continue
            if self.only_include is not None and subject not in self.only_include:
                continue  # include the given subjects only
            if self.exclude is not None and subject in self.exclude:
                continue  # exclude the given subjects

            # get subject metadata
            # NOTE: for MVHumanNet, all cameras have the same intrinsics
            # if different dataset, then may need to generalize this!
            extrinsics_path = os.path.join(subject_path, 'camera_extrinsics.json')
            intrinsics_path = os.path.join(subject_path, 'camera_intrinsics.json')
            extrinsics = self._clean_camera_keys(load_json(extrinsics_path))
            intrinsics = load_json(intrinsics_path)['intrinsics'] # same for all cameras
            camera_scale = load_pickle(os.path.join(subject_path, 'camera_scale.pkl'))

            # for each subject, store camera parameters separately
            # ! NOTE: intrinsics need to be downscaled by 2 later!
            # ! AND extrinsics [t] needs to be scaled by camera_scale later!
            self.cam_params[subject] = {
                'extrinsics': extrinsics, # Dict[camera_id: extrinsic params]
                'intrinsics': intrinsics, # List[List] (turn to matrix)
                'camera_scale': camera_scale # float
            }

            # annots, images, masks share the same camera directory names
            annots_path = os.path.join(subject_path, 'annots')
            images_path = os.path.join(subject_path, 'images_lr')
            masks_path = os.path.join(subject_path, 'fmask_lr')

            # NOTE: assumes the same cameras exist for each subject
            camera_dirs = [d for d in os.listdir(masks_path)]
            subject_map = defaultdict(dict)

            for camera in camera_dirs:
                cam_path = os.path.join(images_path, camera)

                try:
                    for entry in os.scandir(cam_path):
                        if entry.is_file() and entry.name.endswith('_img.jpg'):
                            timestep = entry.name.split('_')[0]
                            annots_json = load_json(os.path.join(annots_path, camera, f"{timestep}_img.json"))['annots'][0]
                            bbox = annots_json['bbox']
                            bbox_face = annots_json['bbox_face2d'] # ! not used
                            if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
                                print(f"Skipping subject {subject} camera {camera} timestep {timestep} because bbox is invalid")
                                continue
                            subject_map[timestep][camera] = {
                                'image_path': entry.path,
                                'mask_path': os.path.join(masks_path, camera, f"{timestep}_img_fmask.png"),
                                'annots': {
                                    'bbox': bbox,
                                    'bbox_face': bbox_face
                                }
                            }
                except Exception as e:
                    print(f"Error loading subject {subject} camera {camera}: {e}")
                    continue
            
            sorted_timesteps = sorted(subject_map.keys())
            for i in range(0, len(sorted_timesteps)):
                timestep = sorted_timesteps[i]
                frames_info = subject_map[timestep]

                if len(frames_info.keys()) < self.num_images: # not enough cameras for this timestep to sample
                    continue # then skip this timestep

                scenes.append({
                    'subject_id': subject,
                    'frames_info': frames_info,
                    'timestep': timestep
                })
        return scenes

    def _get_infu_path(self, subject_id, timestep):
        if self.infu_dataset_path is None:
            return None
        return self.infu_dataset_path + f"/{subject_id}/{timestep}_{subject_id}_img.png"

    def _get_iclight_path(self, subject_id, timestep, camera):
        if self.iclight_dataset_path is None:
            return None
        return self.iclight_dataset_path + f"/{subject_id}/images_lr/{camera}/{timestep}_img.png"

    def _sapiens_get(self, cond, subject_id, camera, timestep, dataset_type="mvhn"):
        try:
            if dataset_type == "mvhn":
                npz_path =  os.path.join(self.root_dir, subject_id, f"{subject_id}_{cond}.npz")
            elif dataset_type == "infu":
                npz_path = os.path.join(self.infu_dataset_path, subject_id, f"{subject_id}_{cond}.npz")
            elif dataset_type == "iclight":
                npz_path = os.path.join(self.iclight_dataset_path, subject_id, f"{subject_id}_{cond}.npz")
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
            # Use context manager to properly close file handles and prevent leaks
            with np.load(npz_path) as data:
                # if INFU, timestep is the given sample ID instead!
                query = timestep if dataset_type == "infu" else f"{camera}_{timestep}"
                # Copy data before context manager closes the file
                return torch.tensor(np.array(data[query], copy=True))
        except Exception as e:
            print(f"Error loading sapiens conditioning: {e}")
            return None

    def _load_raw_frames(self, image_paths, mask_paths):
        """
        Load multi-view frames from a list of 'image_paths' and their corresponding 'mask_paths'
        Returns a tensor of shape (num_images, 3, image_shape[0], image_shape[1]).

        Note: these are raw images straight from the dataset.
        No transform is applied except if needing to resize to expected input shape.
        """
        frames = torch.zeros((self.num_images, 3, self.image_shape[0],  self.image_shape[1]))
        img_masks = torch.zeros((self.num_images, self.image_shape[0], self.image_shape[1]))
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            image = Image.open(img_path).convert("RGB")
            img_mask = Image.open(mask_path)

            if img_mask.size != image.size: # ensure matching size! (note: subject 100681 has different sizes!)
                image = image.resize(img_mask.size, Image.BILINEAR)

            # Create masked image by compositing with black background
            background = Image.new(
                'RGB', image.size, (255, 255, 255) if self.white_background else (0, 0, 0)
            )

            masked_image = Image.composite(image, background, img_mask)
            frames[i] = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(masked_image)
            img_masks[i] = T.Compose([T.ToImage()])(img_mask)
            del image, img_mask, masked_image, background  # free PIL images
        return frames, img_masks

    def _sample_multiview_image_paths(self, frames_info: dict, use_iclight: bool = False, use_infu: bool = False) -> Tuple[list[str], list[str], list[str], Union[list[int], np.ndarray]]:
        """
        Sample multi-view frame + subject mask indices (as string paths) from a frames_info dictionary.
        Returns the sampled images, masks, and camera order.
        Args:
            frames_info: dictionary of frames_info for a subject
            use_iclight: whether to use IC-light
            use_infu: whether to use InfU
        Returns:
            sampled_image_paths: list of sampled image paths
            sampled_image_mask_paths: list of sampled image mask paths
            camera_order: list of camera order
            images_permutation: permutation used to select these subsets
        """
        # Sample multi-view frame + subject mask indices (as string paths)
        camera_order = [cam for cam in list(frames_info.keys())] # 48 sorted camera IDs
        sampled_image_paths = [frames_info[cam]['image_path'] for cam in camera_order]
        sampled_image_mask_paths = [frames_info[cam]['mask_path'] for cam in camera_order]

        # NOTE: if num_images>16, then trajectory NVS will default to using all in rung
        if np.random.rand() <= self.adjacent_frame_sampling_prob: # for trajectory NVS
            # choose which rung of cameras to sample from (top/mid/bot)
            # this is only because these paths are the most apparently continuous
            which_rung = np.random.randint(0, len(CAMERA_RUNGS))
            rung_of_cameras = CAMERA_RUNGS[which_rung]
            start_idx = np.random.randint(0, len(rung_of_cameras)) # out of 16 cameras
            images_permutation = np.roll(np.arange(len(rung_of_cameras)), -start_idx)[:self.num_images]
            images_permutation = [CAMERA_TO_INDEX[rung_of_cameras[i]] for i in images_permutation]
        else: # for set NVS
            # simply uniform sampling of all views
            images_permutation = np.random.choice(len(sampled_image_paths), self.num_images, replace=False)

        # (NOTE: mainly for debugging: overwrites selected samples to a fixed user-defined set)
        if self.fixed_sampling_ids is not None:
            images_permutation = self.fixed_sampling_ids

        # Select subset of multi-view frames (num_images) from the full set of cameras
        camera_order = [camera_order[i] for i in images_permutation] # ordered subset
        sampled_image_paths = [sampled_image_paths[i] for i in images_permutation]
        sampled_image_mask_paths = [sampled_image_mask_paths[i] for i in images_permutation]

        return sampled_image_paths, sampled_image_mask_paths, camera_order, images_permutation

    def _sample_all_masks(self, use_iclight: bool = False, use_infu: bool = False):
        # * Sample input/target frame split
        if not self.use_inconsistent:
            num_input_frames = np.random.randint(1, self.num_images) # at least 1 input frame
        else:
            # using IC, make it higher probability that all num_images are inputs
            if np.random.rand() <= self.all_inputs_prob:
                # more likely to have all inputs (inconsistent images)
                num_input_frames = self.num_images
            else:
                # otherwise, target views without IC are to be predicted (>=1 input frames)
                num_input_frames = np.random.randint(1, self.num_images)

        input_frames_indices = np.random.choice(self.num_images, num_input_frames, replace=False)
        input_target_mask = torch.zeros(self.num_images, dtype=torch.bool)
        input_target_mask[input_frames_indices] = True # 1: input, 0: target

        # * create ref_mask (one-hot)
        if not self.use_inconsistent: # not used in our case
            # since inputs are all consistent in dataset, we can use multiple "references"
            ref_mask = input_target_mask.clone()
        else:
            # inputs will all be inconsistent, and we can only fix to a single reference frame
            ref_mask = torch.zeros(self.num_images, dtype=torch.bool)
            fix_frame_idx = input_frames_indices[np.random.choice(len(input_frames_indices), 1).item()]
            ref_mask[fix_frame_idx] = True # this becomes the fixed frame

        # * create masks for inconsistent images from all synthetic sources
        ic_masks = {}
        if self.use_inconsistent:
            # based on the input/target mask (only inputs can be sampled)
            # ic_sampling_prob is the probabiilty of sampling from IC-light over InfU
            if use_iclight and use_infu:
                ic_masks['iclight'] = torch.rand(self.num_images) < self.ic_sampling_prob
                ic_masks['infu'] = ~ic_masks['iclight']
                # zero-out target frames (these should not have any conditioning)
                ic_masks['iclight'] = ic_masks['iclight'] * input_target_mask
                ic_masks['infu'] = ic_masks['infu'] * input_target_mask
                ic_masks['iclight'][ref_mask] = False
                ic_masks['infu'][ref_mask] = False
            elif use_iclight and not use_infu:
                ic_masks['iclight'] = ~ref_mask * input_target_mask
                ic_masks['infu'] = torch.zeros(self.num_images, dtype=torch.bool)
            elif not use_iclight and use_infu:
                ic_masks['iclight'] = torch.zeros(self.num_images, dtype=torch.bool)
                ic_masks['infu'] = ~ref_mask * input_target_mask
            else:
                raise ValueError(f"use_inconsistent is True, but neither iclight or infu paths are provided!")
        else:
            # zero masks indicating no inconsistent frames
            ic_masks['iclight'] = torch.zeros(self.num_images, dtype=torch.bool)
            ic_masks['infu'] = torch.zeros(self.num_images, dtype=torch.bool)
        return input_target_mask, ref_mask, ic_masks


    def _load_inconsistent_frames(
        self, img_paths, img_mask_paths, cam_order, subject_id, timestep,
        ref_mask, ic_masks):
        """
        Replace the sampled MVHN image paths with inconsistent paths (from IC-light and InfU synthetic data).
        Args:
            img_paths: list of sampled image paths
            img_mask_paths: list of sampled image mask paths
            cam_order: list of camera order
            subject_id: subject ID
            timestep: timestep
            ref_mask: reference mask
            ic_masks: dictionary of IC-light and InfU masks
        Returns:
            ic_rgb: list of actual model input image data (GT, IC-light, InfU)
            ic_paths: list of actual filepaths to inconsistent images
        """
        if not self.use_inconsistent:
            return torch.zeros((self.num_images, 3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)

        ic_paths = []
        # Pre-sample unique InfU indices to avoid duplicates
        num_infu_frames = ic_masks['infu'].sum().item() if isinstance(ic_masks['infu'], torch.Tensor) else ic_masks['infu'].sum()
        if ic_masks['infu'].sum() > 0 and num_infu_frames > 0:
            infu_num_images_in_directory = self.infu_num_images[subject_id]
            # Sample exactly as many unique indices as we need
            num_samples = min(num_infu_frames, infu_num_images_in_directory)
            infu_indices = list(np.random.choice(infu_num_images_in_directory, num_samples, replace=False) + 1)
        else:
            infu_indices = []
        
        # NOTE: these images are of different sizes/shapes!
        for path, is_iclight, is_infu, is_ref, camera in zip(img_paths, ic_masks['iclight'], ic_masks['infu'], ref_mask, cam_order):
            if is_ref:
                ic_path = path # GT frame
            elif is_iclight:
                ic_path = self._get_iclight_path(subject_id, timestep, camera)
            elif is_infu:
                # timestep is treated differently for InfU as a random index
                infu_random_index = infu_indices.pop(0)
                ic_path = self._get_infu_path(subject_id, f"{infu_random_index:06d}")
            else: # target frame (no inconsistent frame)
                ic_path = None
            ic_paths.append(ic_path)

        ic_rgb = []
        tensorize = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        for ic_path in ic_paths:
            if ic_path is not None:
                ic_image = Image.open(ic_path).convert("RGB")
                ic_rgb.append(tensorize(ic_image)) # these can be different image shapes originally
                del ic_image  # free PIL image to prevent memory leak
            else: # if not, then just send the 0 tensor
                ic_rgb.append(torch.zeros((3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32))
        return ic_rgb, ic_paths

    def _crop_and_transform_frames_and_intrinsics(
        self, frames_info, frames, image_masks, ic_rgb, subject_id, cam_order, intrinsics, ref_mask, ic_masks, sapiens_conditionings):
        """
        Crop and transform frames while updating intrinsics as needed.
        Args:
            frames: list of frames
            ic_rgb: list of inconsistent frames
            cam_order: list of camera order
        """
        # create intrinsics tensor, update intrinsics
        if self.random_crop or self.maximal_crop:
            annots_jsons = [frames_info[cam]["annots"] for cam in cam_order]
            crop_params = []
            face_bboxes_adjusted = []
            faces_present_mask = []
            for annots_json in annots_jsons:
                bbox = annots_json['bbox'][:4]
                if self.face_bboxes is not None and annots_json.get('bbox_face') is not None:
                    face_bbox = annots_json['bbox_face'][:4]
                    faces_present_mask.append(True if face_bbox != [-1, -1, -1, -1] else False)
                    face_bboxes_adjusted.append(face_bbox)
                crop_params.append(bbox)
            # account for mvhn downsampling (hence the 0.5)
            # ! big HACK: after 103000+, the annotations are not scaled by 0.5 anymore!
            bbox_annot_scale = 0.5 if int(subject_id) < 103000 else 1.0
            bbox_params = torch.stack([torch.tensor(bbox) * bbox_annot_scale for bbox in crop_params])
            face_params = []
            for face_bbox, is_face in zip(face_bboxes_adjusted, faces_present_mask):
                face_bbox = torch.tensor(face_bbox)
                if is_face: 
                    face_bbox = face_bbox * bbox_annot_scale
                face_params.append(face_bbox)
                
            face_params = torch.stack(face_params) if len(face_params) > 0 else None
            frames, Ks, rel_bbox, face_bboxes_result, new_bbox, bbox_before_pad = self.cropper(frames, bbox_params, torch.from_numpy(intrinsics).float(), face_bboxes=face_params)
            # ! this is bugged; ensure that padding is correctly done 
            image_masks, _ = self.cropper._possibly_pad_img(image_masks.unsqueeze(1), bbox_before_pad[:,0], bbox_before_pad[:,1], bbox_before_pad[:,2], bbox_before_pad[:,3])
            image_masks = self.cropper.crop_images(image_masks, new_bbox[:,0], new_bbox[:,1], new_bbox[:,2], new_bbox[:,3])
            if face_bboxes_result is not None:
                face_bboxes_adjusted = face_bboxes_result
            # NOTE: rel_bbox is the delta from the deterministic crop to the random crop
            # this would then be all 0 if not using random_crop
            # for ic-light, we would later need to scale these by the scale factor (1024->576)

            # later, we resize using transform, so we update cropped intrinsics here accordingly
            scale = np.array([self.target_shape[0] / cropped_img.shape[-2] for cropped_img in frames])
            Ks = update_intrinsics_resize(Ks, scale)
            Ks = normalize_intrinsics(Ks, self.target_shape[0], self.target_shape[1]) # normalize intrinsics (H,W)
            if len(Ks.shape) == 2: # if one shared intrinsic matrix, then repeat it for all
                Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images)
            Ks = torch.from_numpy(Ks).float()
        else: # simply CenterCrop
            # if CenterCrop, then crop original size image to square
            min_dim = min(*self.image_shape)
            max_dim = max(*self.image_shape)
            crop_amount  = (max_dim - min_dim) // 2 # (W-H)/2, cropped from each side (only left-part relevant)
            scale_amount = (self.target_shape[0] / min_dim)
            Ks = update_intrinsics(np.array(intrinsics), crop_x=crop_amount, crop_y=0, scale=scale_amount)
            Ks = normalize_intrinsics(Ks, self.target_shape[0], self.target_shape[1]) # normalize intrinsics (H,W)
            Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images) # assumes all intrinsics are the same 
            Ks = torch.from_numpy(Ks).float()

            # if self.concatenate_sapiens_conditioning is not None:
            #     for cond, is_ref in zip(sapiens_conditionings, ref_mask):
            #         for cond_tensor in sapiens_conditionings[cond]:
            #             if is_ref: # crop first (for MVHN seg/depth maps)
            #                 cond_tensor = cond_tensor[crop_amount:self.target_shape[0]-crop_amount, :]
            #             cond_tensor = torch.nn.functional.interpolate(cond_tensor.unsqueeze(0), size=(self.target_shape[0], self.target_shape[1]), mode='bilinear', align_corners=False).squeeze(0)

            # face bboxes can be found
            annots_jsons = [frames_info[cam]["annots"] for cam in cam_order]
            face_bboxes_adjusted = []
            for annots_json in annots_jsons:
                face_bbox = annots_json['bbox_face'][:4] # this is from facebbox dir
                face_bboxes_adjusted.append(face_bbox)
                if face_bbox != [-1, -1, -1, -1]:
                    # x1,x2 are affected by the center crop
                    # ! assumes center crop is always horizontal (HARDCODED)
                    face_bbox[0] = face_bbox[0] - crop_amount
                    face_bbox[2] = face_bbox[2] - crop_amount
            # account for mvhn downsampling (hence the 0.5)
            # ! big HACK: after 103000+, the annotations are not scaled by 0.5 anymore!
            face_params = torch.stack([torch.tensor(face_bbox) for face_bbox in face_bboxes_adjusted])
            face_bboxes_adjusted = face_params * scale_amount

        # * actual cropping
        # NOTE: the behavior of transform will change depending on whether random crop is used
        # frames = [self.transform(frame) for frame in frames]
        if self.random_crop or self.maximal_crop:
            frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames, dim=0)
            image_masks = [self.mask_transform(img_mask) for img_mask in image_masks]
            image_masks = torch.stack(image_masks, dim=0)
            
            ic_rgb_tensor = torch.zeros((self.num_images, 3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
            for i, (ic_image, bbox, is_ref, is_iclight, is_infu) in enumerate(zip(ic_rgb, rel_bbox, ref_mask, ic_masks['iclight'], ic_masks['infu'])):
                # First resize ic_image to target shape
                ic_image = torch.nn.functional.interpolate(ic_image.unsqueeze(0), size=(self.target_shape[0], self.target_shape[1]), mode='bilinear', align_corners=False).squeeze(0)
                dx1, dy1, dx2, dy2 = bbox.int()
                ic_image_ = ic_image[:,0+dy1:self.target_shape[0]+dy2, 0+dx1:self.target_shape[1]+dx2] # crop
                ic_image_ = self.transform(ic_image_)
                ic_rgb_tensor[i] = ic_image_

                # same for the sapiens conditionings
                if (self.concatenate_sapiens_conditioning is not None and len(self.concatenate_sapiens_conditioning) > 0) or len(self.sapiens_mask_loss_types) > 0:
                    # only the ref images are cropped in this way   
                    ref_idx = torch.where(ref_mask == True)[0][0].item()
                    types = self.concatenate_sapiens_conditioning if (self.concatenate_sapiens_conditioning is not None and len(self.concatenate_sapiens_conditioning) > 0) else self.sapiens_mask_loss_types
                    for cond in types:
                        cond_tensor = sapiens_conditionings[cond][i]
                        if ref_idx == i:
                            # if MVHN, crop using new_bbox 
                            padded_img, refbbox = self.cropper._possibly_pad_img(
                                cond_tensor.unsqueeze(0), 
                                new_bbox[ref_idx][0].unsqueeze(0), 
                                new_bbox[ref_idx][1].unsqueeze(0), 
                                new_bbox[ref_idx][2].unsqueeze(0), 
                                new_bbox[ref_idx][3].unsqueeze(0)
                            )
                            if isinstance(padded_img, list):
                                padded_img = padded_img[0]
                            else:
                                padded_img = padded_img.squeeze(0)
                            refbbox = refbbox.squeeze(0).int()
                            cropped = padded_img[:, refbbox[1]:refbbox[3], refbbox[0]:refbbox[2]]
                            sapiens_conditionings[cond][ref_idx] = T.Resize((self.target_shape[0], self.target_shape[1]))(cropped)
                        else:
                            # if IClight/InfU, crop and resize (but NO normalize)
                            scale = cond_tensor.shape[-2] / self.target_shape[0]
                            cropped_cond_tensor = cond_tensor[:, 0+int(dy1*scale):int((self.target_shape[0]+dy2)*scale), 0+int(dx1*scale):int((self.target_shape[1]+dx2)*scale)]
                            sapiens_conditionings[cond][i] = T.Resize((self.target_shape[0], self.target_shape[1]))(cropped_cond_tensor)
                        if cond == "seg_masks":
                            sapiens_conditionings[cond][i] = one_hot_encode_segmentation(sapiens_conditionings[cond][i], 28)

            sapiens_conditionings = {cond: torch.stack(cond_tensor, dim=0) for cond, cond_tensor in sapiens_conditionings.items()}
            ic_rgb = ic_rgb_tensor
        else: # center crop + resize only (NOTE: set transform=None for this default behavior)
            frames = self.transform(frames)
            ic_rgb = [self.transform(ic_image) for ic_image in ic_rgb] # do the same thing as frames
            ic_rgb = torch.stack(ic_rgb, dim=0)
            image_masks = self.mask_transform(image_masks)
            image_masks = torch.stack(image_masks, dim=0)

            # frames = torch.stack(frames, dim=0) # resize to 576x576 normalized [-1, 1] image tensors
            if self.concatenate_sapiens_conditioning is not None or len(self.sapiens_mask_loss_types) > 0:
                types = self.concatenate_sapiens_conditioning if (self.concatenate_sapiens_conditioning is not None and len(self.concatenate_sapiens_conditioning) > 0) else self.sapiens_mask_loss_types
                for cond in types:
                    for is_ref, cond_tensor in zip(ref_mask, sapiens_conditionings[cond]):
                        cond_tensor = T.Resize((self.target_shape[0], self.target_shape[1]))(cond_tensor) # both follows this old behavior
                        if cond == "seg_masks":
                            cond_tensor = one_hot_encode_segmentation(cond_tensor, 28)
            sapiens_conditionings = {cond: torch.stack(cond_tensor, dim=0) for cond, cond_tensor in sapiens_conditionings.items()}

        if face_bboxes_adjusted is not None:
            if not isinstance(face_bboxes_adjusted, torch.Tensor):
                face_bboxes_adjusted = torch.tensor(face_bboxes_adjusted)
        else:
            # Provide a dummy tensor if no face bboxes are available to keep batch structure consistent
            face_bboxes_adjusted = torch.full((self.num_images, 4), -1, dtype=torch.int32)

        return frames, image_masks, ic_rgb, Ks, sapiens_conditionings, face_bboxes_adjusted

    def _get_arcface_embeddings(self, subject_id, timestep, cam_order, input_target_mask, ref_mask, ic_masks):
        arcface_embeddings = []
        if self.arcface_embeddings_dir is not None:
            for is_ref, is_iclight, is_infu, cam in zip(ref_mask, ic_masks['iclight'], ic_masks['infu'], cam_order):   
                try:
                    if is_ref: # not implemented yet
                        arcface_embedding = self._read_arcface_embeddings("mvhn", subject_id, cam, f"{timestep}_img.jpg")
                    elif is_infu:
                        arcface_embedding = self._read_arcface_embeddings("infu", subject_id, cam, f"{timestep}_img.png")
                    elif is_iclight: # is iclight
                        arcface_embedding = self._read_arcface_embeddings("iclight", subject_id, cam, f"{timestep}_img.png")
                    else:
                        arcface_embedding = None
                except KeyError:
                    arcface_embedding = None
                except Exception as e:
                    print(f"Error reading arcface embedding: {e}")
                    arcface_embedding = None
                arcface_embeddings.append(arcface_embedding)
            
            # Convert to tensor [T, 512]
            # Handle None values by replacing with zeros
            arcface_embeddings = [
                torch.tensor(emb) if emb is not None else torch.zeros(512, dtype=torch.float32)
                for emb in arcface_embeddings
            ]
            arcface_embeddings = torch.stack(arcface_embeddings)
            arcface_embeddings[~input_target_mask] *= 0
            # don't use arcface embedding for target frames
            # NOTE: it's possible that we can just average all of these out
            # to get the average face embedding (which proves to be effective in the InstantID paper)
        else: # zero tensor
            arcface_embeddings = torch.zeros((self.num_images, 512), dtype=torch.float32)
        
        return arcface_embeddings

    def _get_sapiens_conditionings(self, subject_id, timestep, cam_order, ic_paths, input_target_mask, ref_mask, ic_masks):
        """
        Get sapiens conditionings.
        Args:
            subject_id: subject ID
            timestep: timestep
            cam_order: list of camera order
        """
         # get sapiens conditionings; NOTE: these are of original image size (need to crop later)
        sapiens_conditionings = {}
        if self.concatenate_sapiens_conditioning is not None and len(self.concatenate_sapiens_conditioning) > 0 or len(self.sapiens_mask_loss_types) > 0:
            types = self.concatenate_sapiens_conditioning if (self.concatenate_sapiens_conditioning is not None and len(self.concatenate_sapiens_conditioning) > 0) else self.sapiens_mask_loss_types
            for cond in types:
                sapiens_conditionings[cond] = []
                for is_ref, is_iclight, is_infu, camera, ic_path in zip(ref_mask, ic_masks['iclight'], ic_masks['infu'], cam_order, ic_paths):
                    try: 
                        cond_tensor = None
                        if is_ref:
                            # The reference frame always uses MVHN ground truth
                            cond_tensor = self._sapiens_get(cond, subject_id, camera, timestep, dataset_type="mvhn")
                        elif is_iclight: 
                            cond_tensor = self._sapiens_get(cond, subject_id, camera, timestep, dataset_type="iclight")
                        else: # infu
                            cond_tensor = self._sapiens_get(cond, subject_id, camera, f"{os.path.basename(ic_path).split('_')[0]}", dataset_type="infu")

                        cond_tensor = torch.nan_to_num(cond_tensor, nan=0) # masks have 'nan' as background values
                    except Exception as e:
                        pass
                    if cond_tensor is None:
                        if cond == "depth":
                            cond_tensor = torch.zeros((1, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
                        elif cond == "seg_masks":
                            # expand later using one_hot_encode_segmentation (in crop_and_transform_frames_and_intrinsics)
                            # refactor and decouple later 
                            cond_tensor = torch.zeros((1, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
                        elif cond == "latents":
                            cond_tensor = torch.zeros((4, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
                    else:
                        cond_tensor = cond_tensor.unsqueeze(0)
                    sapiens_conditionings[cond].append(cond_tensor)
        return sapiens_conditionings


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        try:
            return self.create_batch(idx)
        except Exception as e:
            print(f"Error creating batch at index {idx}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            # Force flush to ensure error is visible before potential segfault
            sys.stdout.flush()
            sys.stderr.flush()
            return None
    
    def create_batch(self, idx):
        """
        Collect multi-views + conditioning data for a scene at a fixed timestep,
        preprocess for training loop.
        """
        scene = self.scenes[idx] # get scene content info
        subject_id = scene['subject_id'] # ex. 100001
        timestep = scene['timestep'] # ex. 0005
        
        # Reconstruct frames_info and other metadata
        if self.is_arrow:
            row = self.dataset[scene['arrow_idx']]
            
            # Parse stored JSON metadata
            extrinsics = json.loads(row['extrinsics'])
            intrinsics = json.loads(row['intrinsics'])
            camera_scale = float(row['camera_scale'])
            
            # Parse annots for bboxes
            annots_bbox = json.loads(row['annots_bbox'])
            annots_bbox_face = json.loads(row['annots_bbox_face2d'])
            
            # Reconstruct frames_info
            frames_info = {}
            cameras = row['cameras'] # List of camera IDs
            subject_path = os.path.join(self.root_dir, subject_id)
            
            for camera in cameras:
                # Create time_id matching the format in JSON/keys
                # timestep is "0005" string
                time_id = timestep 
                
                # Check if this camera has annotation for this timestep
                if camera in annots_bbox and time_id in annots_bbox[camera]:
                    bbox = annots_bbox[camera][time_id]
                    bbox_face = [-1,-1,-1,-1]
                    if camera in annots_bbox_face and time_id in annots_bbox_face[camera]:
                        bbox_face = annots_bbox_face[camera][time_id]
                        if bbox_face[:4] == [0.0,0.0,100.0,100.0]:
                            bbox_face = [-1,-1,-1,-1]
                    # Validate bbox (same check as in JSON loader)
                    if (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
                        continue

                    # Construct paths                    
                    image_filename = f"{time_id}_img.jpg"
                    mask_filename = f"{time_id}_img_fmask.png"
                    
                    frames_info[camera] = {
                        'image_path': os.path.join(subject_path, "images_lr", camera, image_filename),
                        'mask_path': os.path.join(subject_path, "fmask_lr", camera, mask_filename),
                        'annots': {
                            'bbox': bbox,
                            'bbox_face': bbox_face
                        }
                    }
            
            # Convert intrinsics list to array (if needed)
            intrinsics = np.array(intrinsics['intrinsics'] if isinstance(intrinsics, dict) else intrinsics)
            
        else:
            # Legacy JSON loader path
            frames_info = dict(sorted(scene['frames_info'].items())) # camera dict data (annots)
            subject_path = os.path.join(self.root_dir, subject_id)
            # get camera parameters from cache
            extrinsics = self.cam_params[subject_id]['extrinsics']
            intrinsics = np.array(self.cam_params[subject_id]['intrinsics'])
            camera_scale = self.cam_params[subject_id]['camera_scale'] 

        if self.pre_scale_intrinsics != 1:
            # update intrinsics (required for MVHumanNet; default 0.5x prescaling)
            intrinsics = update_intrinsics_resize(intrinsics, scale=self.pre_scale_intrinsics)
            
        # Ensure intrinsics is numpy array for downstream processing
        if isinstance(intrinsics, list):
             intrinsics = np.array(intrinsics)

        # Sample multi-view frame + subject mask indices (as string paths)
        img_paths, img_mask_paths, cam_order, sample_permutation = self._sample_multiview_image_paths(frames_info)
        
        # generate masks for input/target frames split, reference frame, and IC/InfU frames
        input_target_mask, ref_mask, ic_masks = self._sample_all_masks(
            use_iclight=self.iclight_dataset_path is not None,
            use_infu=self.infu_dataset_path is not None,
        )
        iclight_mask = ic_masks["iclight"]
        infu_mask = ic_masks["infu"]

        # load raw GT frames from MVHN
        frames, image_masks = self._load_raw_frames(img_paths, img_mask_paths)

        # loads inconsistent frames from IC-light and InfU synthetic data
        ic_rgb, ic_paths = self._load_inconsistent_frames(
            img_paths, img_mask_paths, cam_order,
            subject_id, timestep, ref_mask, ic_masks
        )

        # arcface conditionings
        arcface_embeddings = self._get_arcface_embeddings(
            subject_id, timestep, cam_order, input_target_mask, ref_mask, ic_masks
        )

        # sapiens conditionings
        sapiens_conditionings = self._get_sapiens_conditionings(
            subject_id, timestep, cam_order, ic_paths, input_target_mask, ref_mask, ic_masks
        )

        # transform GT and inconsistent frames + intrinsics to desired shape
        # do these (special) crops for the synthetic data as well
        frames, img_masks, ic_rgb, Ks, sapiens_conditionings, face_bboxes_adjusted = self._crop_and_transform_frames_and_intrinsics(
            frames_info, frames, image_masks, ic_rgb,
            subject_id, cam_order, intrinsics,
            ref_mask, ic_masks, sapiens_conditionings
        )
        img_masks = img_masks / 255.0 # convert to [0, 1]

        # this will always be constant 1-tensor (according to pretrained model authors)
        camera_mask = torch.ones(self.num_images, dtype=torch.bool)

        def get_c2w(cam):
            tf_matrix = create_transform_matrix(
                np.array(extrinsics[cam]['rotation']),
                np.array(extrinsics[cam]['translation']) * camera_scale,
                homogeneous=True
            )

            return np.linalg.inv(tf_matrix) # w2c -> c2w

        # Read extrinsics (w2c -> c2w) to fit SEVA camera convention
        all_c2ws = np.array([
            get_c2w(cam) for cam in frames_info.keys() # these keys are SORTED
        ])

        all_c2ws = torch.from_numpy(all_c2ws).float() # (total_cameras=48, 4, 4)
        c2ws = all_c2ws[sample_permutation] # extrinsics for sampled cameras
        center_cameras(all_c2ws, c2ws) # mean center
        scale_cameras(c2ws)

        # plucker coordinates for all cameras
        w2cs = torch.linalg.inv(c2ws) # c2w -> w2c
        # Find the first input frame (first True in input_target_mask) to use as source camera
        # argmax returns the first True index, or 0 if all False
        src_camera_idx = input_target_mask.to(torch.int).argmax().item()
        pluckers = get_plucker_coordinates( # relative to the first camera in the sample
            extrinsics_src=w2cs[src_camera_idx],
            extrinsics=w2cs,
            intrinsics=Ks.clone(),
            target_size=(self.target_shape[0] // self.downsample_factor, 
                         self.target_shape[1] // self.downsample_factor),
        )

        # load preloaded latents if provided (mostly deprecated)
        if self.latents_dir is not None and os.path.exists(os.path.join(self.latents_dir, subject_id, f"{subject_id}.npz")) and not self.maximal_crop:
            npz_file = os.path.join(self.latents_dir, subject_id, f"{subject_id}.npz")
            # npz_data = np.load(npz_file) # this is already for the current subject
            with np.load(npz_file) as npz_data:
                latent_tensors = [npz_data[f"{sample_cam}.{timestep}"] for sample_cam in cam_order]
                clean_latents = torch.stack([torch.from_numpy(latent_tensor) for latent_tensor in latent_tensors]) # (B, 4, 72, 72)
        else: # else encode frames on the fly (loaded in diffusion.py)
            clean_latents = 0 # indicates to diffusion.py to encode on the fly


        concat = torch.cat( # binary masks (inp/tgt + ref) and pluckers
            [
                repeat(
                    input_target_mask,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
                repeat(
                    ref_mask, 
                    "n -> n 1 h w", 
                    h=pluckers.shape[2],
                    w=pluckers.shape[3] 
                ),
            ],
            dim=1,
        ) # (T, 6 + 1, 72, 72), where 6 is for plucker coords and 1 for binary mask

        # Why sapiens cond not in concat? => needs to be processed extra via a projection layer
        # then, in diffusion.py, we process and concatenate over there

        if type(clean_latents) == int and clean_latents == 0:
            replace = 0
        else:
            replace = torch.cat(
                [
                    clean_latents * self.scale_factor,
                    # repeat(
                    #     input_frames_mask,
                    #     "n -> n 1 h w",
                    #     h=pluckers.shape[2],
                    #     w=pluckers.shape[3],
                    # ),
                    repeat(
                        ref_mask, 
                        "n -> n 1 h w", 
                        h=pluckers.shape[2],
                        w=pluckers.shape[3] 
                    ),
                ],
                dim=1,
            )

        try:
            # ensure in shared_step:
            # - clean_latents gets encoded on the fly (if not found)
            # - update concat with ic latents
            # - replace gets updated
            output_dict = {
                "clean_latent": clean_latents, # unscaled clean latents
                "mask": input_target_mask,
                "ref_mask": ref_mask, # "one hot" mask for reference images 
                "ic_rgb": ic_rgb, # ! NOTE: normalized!
                "plucker": pluckers,
                "camera_mask": camera_mask,
                "concat": concat,
                "frames": frames,  # transformed frames (post-cropping, normalized)
                "frames_masks": img_masks, # corresponding masks
                "replace": replace, # contains pre-scaled clean latents!
                "c2w": c2ws,
                "K": Ks,
                "use_inconsistent": self.use_inconsistent,
                "face_bbox": face_bboxes_adjusted,  # Face bounding boxes [T, 4] in pixel coords (x1, y1, x2, y2)
                "subject_id": subject_id,
                "timestep": timestep,
                # NOTE: face_bbox is w.r.t post-cropping, resized 576^2 image!
            }
            # NOTE: sapiens conditioning will be processed internally to 

            # Add ArcFace embeddings if available
            if self.arcface_embeddings_dir is not None:
                output_dict["arcface_embedding"] = arcface_embeddings  # [T, 512]
                # where None values are replaced with zero tensor

            if self.concatenate_sapiens_conditioning is not None \
                or len(self.sapiens_mask_loss_types) > 0:
                output_dict["sapiens_conditioning"] = sapiens_conditionings
        except Exception as e:
            print(f"Error creating output_dict: {e}")
            raise

        return output_dict

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

def expand_only_include(only_include):
    if isinstance(only_include, str): # in the format ex: "100001-102000,102020-104000"
        only_include = only_include.split(",")
        expanded_includes = []
        for subrange in only_include:
            start, end = [int(num) for num in subrange.split("-")]
            expanded_includes.extend([str(i).zfill(6) for i in range(start, end + 1)])
        return expanded_includes
    else:
        return only_include

class MVHumanNetLoader(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        num_images: int,
        batch_size: int,
        latents_dir: str = None,
        num_workers: int = 0,
        shuffle: bool = True,
        image_size: int = 576,
        data_limit: int = None,
        only_include: list = None,
        exclude: list = None,
        step_size: int = 150,
        preload_path: str = None,
        iclight_dataset_path: str = None,
        infu_dataset_path: str = None,
        face_bbox_dir: str = None,
        arcface_embeddings_dir: str = None,
        random_crop: bool = False,
        maximal_crop: bool = True,
        val_include: list = None,
        use_inconsistent: bool = False,
        random_crop_prob: float = 0.3,
        ic_sampling_prob: float = 0.7,
        fixed_sampling_ids: list = None,
        concatenate_sapiens_conditioning: list = None,
        sapiens_segmentation_channels_to_use: list = None,
        sapiens_mask_loss_types: list = None,
        face_crop_prob: float = 0.5,
        face_bias_strength: float = 0.7,
    ):
        super().__init__()
        print("init of DATALOADER")
        self.root_dir = root_dir
        self.latents_dir = os.path.join(self.latents_dir) if latents_dir is not None else None
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_limit = data_limit
        self.only_include = only_include
        self.exclude = exclude
        self.step_size = step_size
        self.preload_path = preload_path
        self.iclight_dataset_path = iclight_dataset_path
        self.infu_dataset_path = infu_dataset_path
        self.face_bbox_dir = face_bbox_dir
        self.arcface_embeddings_dir = arcface_embeddings_dir
        self.random_crop = random_crop
        self.maximal_crop = maximal_crop
        self.val_include = val_include
        self.use_inconsistent = use_inconsistent
        self.random_crop_prob = random_crop_prob
        self.ic_sampling_prob = ic_sampling_prob
        self.fixed_sampling_ids = fixed_sampling_ids
        self.concatenate_sapiens_conditioning = concatenate_sapiens_conditioning
        self.sapiens_segmentation_channels_to_use = sapiens_segmentation_channels_to_use
        self.sapiens_mask_loss_types = sapiens_mask_loss_types if sapiens_mask_loss_types is not None else []
        self.face_crop_prob = face_crop_prob
        self.face_bias_strength = face_bias_strength
        # Define transforms
        # self.transform = T.Compose([
        #     T.Resize(image_size), # whatever final resolution we want here
        #     T.ToTensor(),
        # ])
        self.transform = None # let corresponding Dataset handle this
        if isinstance(self.only_include, str): # in the format ex: "100001-102000,102020-104000"
            if os.path.exists(self.only_include): # if passed in a file (subject numbers on each line)
                with open(self.only_include, 'r') as f:
                    self.only_include = [line.strip() for line in f]
            else:
                self.only_include = expand_only_include(self.only_include)
        if isinstance(self.exclude, str): # in the format ex: "100001-102000,102020-104000"
            if os.path.exists(self.exclude): # if passed in a file (subject numbers on each line)
                with open(self.exclude, 'r') as f:
                    self.exclude = [line.strip() for line in f]
            else:
                self.exclude = expand_only_include(self.exclude)
        if isinstance(self.val_include, str): # in the format ex: "100001-102000,102020-104000"
            if os.path.exists(self.val_include): # if passed in a file (subject numbers on each line)
                with open(self.val_include, 'r') as f:
                    self.val_include = [line.strip() for line in f]
            else:
                self.val_include = expand_only_include(self.val_include)

    def setup(self, stage: Optional[str] = None):
        print("setup of DATALOADER")
        print("stage: ", stage)
        if stage == "fit" or stage is None:
            print("train is reached")
            self.train_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir),
                latents_dir=self.latents_dir,
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.only_include,
                exclude=self.exclude,
                step_size=self.step_size,
                preload_path=self.preload_path,
                iclight_dataset_path=self.iclight_dataset_path,
                infu_dataset_path=self.infu_dataset_path,
                face_bbox_dir=self.face_bbox_dir,
                arcface_embeddings_dir=self.arcface_embeddings_dir,
                random_crop=self.random_crop,
                maximal_crop=self.maximal_crop,
                use_inconsistent=self.use_inconsistent,
                random_crop_prob=self.random_crop_prob,
                ic_sampling_prob=self.ic_sampling_prob,
                fixed_sampling_ids=self.fixed_sampling_ids,
                concatenate_sapiens_conditioning=self.concatenate_sapiens_conditioning,
                sapiens_segmentation_channels_to_use=self.sapiens_segmentation_channels_to_use,
                sapiens_mask_loss_types=self.sapiens_mask_loss_types,
                face_crop_prob=self.face_crop_prob,
                face_bias_strength=self.face_bias_strength,
            )

        if stage == "validate" or stage is None:
            print("val_dataset reached")
            self.val_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir),
                latents_dir=self.latents_dir,
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.val_include,
                exclude=self.exclude,
                step_size=self.step_size, # don't want too many samples for validation set
                preload_path=self.preload_path,
                iclight_dataset_path=self.iclight_dataset_path,
                infu_dataset_path=self.infu_dataset_path,
                face_bbox_dir=self.face_bbox_dir,
                arcface_embeddings_dir=self.arcface_embeddings_dir,
                random_crop=self.random_crop,
                maximal_crop=self.maximal_crop,
                use_inconsistent=self.use_inconsistent,
                ic_sampling_prob=self.ic_sampling_prob,
                random_crop_prob=self.random_crop_prob,
                fixed_sampling_ids=self.fixed_sampling_ids,
                concatenate_sapiens_conditioning=self.concatenate_sapiens_conditioning,
                sapiens_segmentation_channels_to_use=self.sapiens_segmentation_channels_to_use,
                sapiens_mask_loss_types=self.sapiens_mask_loss_types,
                face_crop_prob=self.face_crop_prob,
                face_bias_strength=self.face_bias_strength,
            )
        if stage == "test" or stage is None:
            self.test_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir, "test"),
                latents_dir=self.latents_dir,
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.only_include,
                exclude=self.exclude,
                step_size=self.step_size,
                preload_path=self.preload_path,
                iclight_dataset_path=self.iclight_dataset_path,
                infu_dataset_path=self.infu_dataset_path,
                face_bbox_dir=self.face_bbox_dir,
                arcface_embeddings_dir=self.arcface_embeddings_dir,
                random_crop=self.random_crop,
                maximal_crop=self.maximal_crop,
                use_inconsistent=self.use_inconsistent,
                ic_sampling_prob=self.ic_sampling_prob,
                random_crop_prob=self.random_crop_prob,
                fixed_sampling_ids=self.fixed_sampling_ids,
                concatenate_sapiens_conditioning=self.concatenate_sapiens_conditioning,
                sapiens_segmentation_channels_to_use=self.sapiens_segmentation_channels_to_use,
                sapiens_mask_loss_types=self.sapiens_mask_loss_types,
                face_crop_prob=self.face_crop_prob,
                face_bias_strength=self.face_bias_strength,
            )
            
    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        print("dataloader train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=custom_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, 'val_dataset'):
            self.setup("validate")
        k = 1 # fixed for now, sample randomly once from val set
        sampler = RandomSampler(self.val_dataset, num_samples=self.batch_size * k, replacement=True)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=custom_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=custom_collate,
        )
