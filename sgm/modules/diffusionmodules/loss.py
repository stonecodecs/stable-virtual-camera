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
        face_perceptual_weight: float = 0.1,
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
        model_output: torch.Tensor,  # [B, T, C, H, W] latents
        target: torch.Tensor,  # [B, T, C, H, W] latents
        batch: Dict,
    ) -> torch.Tensor:
        """
        Compute perceptual loss on face-cropped regions in RGB space.
        1. Crops latents to face bboxes
        2. Decodes face crop latents to RGB using first_stage_model (VAE decoder)
        3. Computes LPIPS on RGB face crops
        
        Args:
            model_output: Predicted latents [B, T, C, H, W]
            target: Target latents [B, T, C, H, W]
            batch: Batch dict containing face_bbox [B, T, 4]
            
        Returns:
            Face crop perceptual loss scalar
        """
        # Check if decoder is available
        if self.first_stage_model is None or self.scale_factor is None:
            raise RuntimeError(
                "first_stage_model and scale_factor must be set before using face perceptual loss. "
                "Call loss_fn.first_stage_model = model.first_stage_model in your training setup."
            )
        
        B, T, C, H, W = model_output.shape
        
        # Get face bounding boxes [B, T, 4] in pixel space (x1, y1, x2, y2)
        face_bboxes = batch["face_bbox"]  # [B, T, 4]
        
        # Flatten batch and time dimensions
        model_flat = model_output.reshape(B * T, C, H, W)
        target_flat = target.reshape(B * T, C, H, W)
        face_bboxes_flat = face_bboxes.reshape(B * T, 4)
        
        # Collect cropped face patches (in latent space)
        cropped_pred_latents = []
        cropped_target_latents = []
        
        for i in range(B * T):
            x1, y1, x2, y2 = face_bboxes_flat[i]
            
            # Skip invalid bboxes (including -1 placeholder for no face)
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                continue
            
            # Convert pixel coords to latent coords (8x downsampling for VAE)
            x1_lat = int((x1 / 8.0).clamp(0, W - 1))
            y1_lat = int((y1 / 8.0).clamp(0, H - 1))
            x2_lat = int((x2 / 8.0).clamp(1, W))
            y2_lat = int((y2 / 8.0).clamp(1, H))
            
            # Skip if crop is too small
            if x2_lat <= x1_lat or y2_lat <= y1_lat:
                continue
            
            # Crop the face region in latent space
            pred_crop = model_flat[i:i+1, :, y1_lat:y2_lat, x1_lat:x2_lat]
            target_crop = target_flat[i:i+1, :, y1_lat:y2_lat, x1_lat:x2_lat]
            
            cropped_pred_latents.append(pred_crop)
            cropped_target_latents.append(target_crop)
        
        # If no valid face crops, return zero loss
        if len(cropped_pred_latents) == 0:
            return torch.tensor(0.0, device=model_output.device, dtype=model_output.dtype)
        
        # Stack all latent crops
        cropped_pred_latents = torch.cat(cropped_pred_latents, dim=0)  # [N, 4, H_crop, W_crop]
        cropped_target_latents = torch.cat(cropped_target_latents, dim=0)  # [N, 4, H_crop, W_crop]
        
        # Decode latent crops to RGB using first_stage_model
        with torch.no_grad():
            # Unscale latents before decoding
            cropped_pred_latents_unscaled = cropped_pred_latents / self.scale_factor
            cropped_target_latents_unscaled = cropped_target_latents / self.scale_factor
        
        # Decode to RGB (gradients flow through predictions, not targets)
        # We need gradients for pred but not for target
        cropped_pred_rgb = self.first_stage_model.decode(cropped_pred_latents_unscaled)  # [N, 3, H_rgb, W_rgb]
        with torch.no_grad():
            cropped_target_rgb = self.first_stage_model.decode(cropped_target_latents_unscaled)  # [N, 3, H_rgb, W_rgb]
        
        # Resize RGB crops to fixed size for LPIPS (expects consistent input)
        cropped_pred_rgb = F.interpolate(
            cropped_pred_rgb,
            size=(self.face_crop_size, self.face_crop_size),
            mode='bilinear',
            align_corners=False
        )
        cropped_target_rgb = F.interpolate(
            cropped_target_rgb,
            size=(self.face_crop_size, self.face_crop_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Compute LPIPS on RGB face crops (now in proper RGB space!)
        lpips_loss = self.lpips(cropped_pred_rgb, cropped_target_rgb)
        
        # Average over all valid face crops
        face_loss = lpips_loss.mean()
        
        return face_loss


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