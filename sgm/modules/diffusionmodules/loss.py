from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        use_face_perceptual: bool = False,
        face_perceptual_weight: float = 0.3,
        face_crop_size: int = 128,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level
        self.use_face_perceptual = use_face_perceptual
        self.face_perceptual_weight = face_perceptual_weight
        self.face_crop_size = face_crop_size
        
        # Store reference to first_stage_model for RGB decoding (set externally)
        self.first_stage_model = None
        self.scale_factor = None

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()
        
        # Initialize LPIPS for face perceptual loss if needed
        if self.use_face_perceptual:
            if not hasattr(self, 'lpips'):
                self.lpips = LPIPS().eval()
            for param in self.lpips.parameters():
                param.requires_grad = False

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        # print("\nStandardDiffusionLoss::forward cond:\n", cond)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor, # this is clean_latent
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        # print("\nStandardDiffusionLoss::forward cond2:\n", cond)
        if "mask" in cond: #  
            # if SevaWeighting, then uncomment out
            # w = append_dims(self.loss_weighting(sigmas, batch["ref_mask"]), input.ndim) # replace with ref_mask
            w = append_dims(self.loss_weighting(sigmas, cond["mask"], batch["ref_mask"]), input.ndim) # replace with ref_mask
        else:
            w = append_dims(self.loss_weighting(sigmas), input.ndim)
        
        # Compute base loss
        base_loss = self.get_loss(model_output, input, w)
        
        # Add face perceptual loss if enabled
        if self.use_face_perceptual and "face_bbox" in batch:
            # input is "clean_latent"
            # in face loss, we work in RGB space, so we use batch["frames"]
            # for LPIPS comparisons over the face
            face_loss = self.get_face_crop_perceptual_loss(model_output, input, batch)
            total_loss = base_loss + self.face_perceptual_weight * face_loss
            return total_loss
        
        return base_loss

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

    def get_face_crop_perceptual_loss(
        self,
        model_output: torch.Tensor,  # Predicted latents [B, T, C, H, W]
        target: torch.Tensor,        # Target latents [B, T, C, H, W] (not used, gt_frames is used instead)
        batch: Dict,
    ) -> torch.Tensor:
        """
        Compute perceptual loss on face-cropped regions in RGB space.
        1. Decodes full latents to full RGB images.
        2. Crops face regions from the decoded RGB images and ground-truth frames.
        3. Resizes all crops to a uniform size.
        4. Computes LPIPS loss on the RGB face crops.
        """
        # Check if decoder is available
        if self.first_stage_model is None or self.scale_factor is None:
            raise RuntimeError(
                "first_stage_model and scale_factor must be set before using face perceptual loss."
            )

        B, T, C, H, W = model_output.shape
        
        # Unscale the predicted latents before decoding (taken care of by first_stage_model)
        # model_output_unscaled = model_output / self.scale_factor
        
        # Reshape for decoding: [B*T, C, H, W]
        # Gradients need to flow through the predicted RGBs
        pred_rgbs_flat = self.first_stage_model.decode(model_output.reshape(B * T, C, H, W))
        gt_rgbs_flat = batch["frames"].reshape(B * T, 3, pred_rgbs_flat.shape[2], pred_rgbs_flat.shape[3])
        
        # Get face bounding boxes and flatten them
        face_bboxes_flat = batch["face_bbox"].reshape(B * T, 4)

        cropped_pred_rgbs = []
        cropped_gt_rgbs = []

        # 2. Loop through the batch to CROP the RGB images
        for i in range(B * T):
            x1, y1, x2, y2 = face_bboxes_flat[i].long()
            
            # Skip invalid bboxes (-1 placeholder and others)
            if x1 >= x2 or y1 >= y2 or x1 < 0:
                continue
                
            # Crop the face region directly from the full RGB images
            pred_crop = pred_rgbs_flat[i:i+1, :, y1:y2, x1:x2]
            gt_crop = gt_rgbs_flat[i:i+1, :, y1:y2, x1:x2]
            
            cropped_pred_rgbs.append(pred_crop)
            cropped_gt_rgbs.append(gt_crop)

        # If no valid faces were found in the batch, return zero loss
        if not cropped_pred_rgbs:
            return torch.tensor(0.0, device=model_output.device, dtype=model_output.dtype)

        # 3. Resize all collected RGB crops to a uniform size for LPIPS
        resized_pred_crops = torch.cat([
            F.interpolate(crop, size=(self.face_crop_size, self.face_crop_size), mode='bilinear', align_corners=False)
            for crop in cropped_pred_rgbs
        ], dim=0)
        
        resized_gt_crops = torch.cat([
            F.interpolate(crop, size=(self.face_crop_size, self.face_crop_size), mode='bilinear', align_corners=False)
            for crop in cropped_gt_rgbs
        ], dim=0)

        # 4. Compute LPIPS loss on the batched, resized RGB crops
        lpips_loss = self.lpips(resized_pred_crops, resized_gt_crops)
        return lpips_loss.mean()

def interpolate_weights_batch(bools: torch.Tensor, max_weight=5.0) -> torch.Tensor:
    B, N = bools.shape
    indices = torch.arange(N, device=bools.device).unsqueeze(0).expand(B, N)
    weights = torch.full((B, N), max_weight, dtype=torch.float, device=bools.device)
    
    for b in range(B):
        true_idx = indices[b][bools[b]]
        if len(true_idx) > 0:
            dists = torch.stack([torch.abs(indices[b] - t) for t in true_idx]).min(dim=0).values
            dists[bools[b]] = 0
            weights[b] = dists / dists.max() * max_weight
        else:
            weights[b] = max_weight

    return weights