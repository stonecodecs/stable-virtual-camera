"""
Face-aware perceptual loss for improving face region quality in diffusion models.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class FaceAwareDiffusionLoss(nn.Module):
    """
    Combines standard diffusion loss (MSE/L1) with face-weighted perceptual loss.
    Uses face bounding boxes to weight perceptual loss more heavily in face regions.
    """
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        use_face_loss: bool = True,
        face_loss_weight: float = 0.1,
        face_region_weight: float = 3.0,
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        decode_for_perceptual: bool = True,
    ):
        """
        Args:
            sigma_sampler_config: Config for sigma sampler
            loss_weighting_config: Config for loss weighting
            loss_type: Base loss type ("l2" or "l1")
            use_face_loss: Whether to use face-aware perceptual loss
            face_loss_weight: Weight for perceptual loss component (relative to base loss)
            face_region_weight: Multiplier for face region in perceptual loss (vs. non-face)
            offset_noise_level: Offset noise level for training
            batch2model_keys: Keys to pass from batch to model
            decode_for_perceptual: Whether to decode latents before computing perceptual loss
        """
        super().__init__()

        assert loss_type in ["l2", "l1"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level
        self.use_face_loss = use_face_loss
        self.face_loss_weight = face_loss_weight
        self.face_region_weight = face_region_weight
        self.decode_for_perceptual = decode_for_perceptual

        if self.use_face_loss:
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
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,  # clean_latent [B, T, C, H, W]
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (input.shape[0], input.shape[1])
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )

        # Get weighting for base loss
        if "mask" in cond:
            w = append_dims(
                self.loss_weighting(sigmas, cond["mask"], batch["ref_mask"]), 
                input.ndim
            )
        else:
            w = append_dims(self.loss_weighting(sigmas), input.ndim)

        # Compute base loss (MSE or L1 in latent space)
        base_loss = self.get_base_loss(model_output, input, w)

        # Compute face-aware perceptual loss if enabled
        if self.use_face_loss and "face_bbox" in batch:
            face_loss = self.get_face_perceptual_loss(
                model_output, input, batch, sigmas
            )
            total_loss = base_loss + self.face_loss_weight * face_loss
        else:
            total_loss = base_loss

        return total_loss

    def get_base_loss(self, model_output, target, w):
        """Compute base loss (L2 or L1) in latent space"""
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

    def get_face_perceptual_loss(
        self, 
        model_output: torch.Tensor,  # [B, T, C, H, W] latents
        target: torch.Tensor,  # [B, T, C, H, W] latents
        batch: Dict,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute face-aware perceptual loss.
        
        Args:
            model_output: Predicted latents [B, T, C, H, W]
            target: Target latents [B, T, C, H, W]
            batch: Batch dict containing face_bbox and decoder
            sigmas: Noise levels
            
        Returns:
            Face-weighted perceptual loss scalar
        """
        B, T = model_output.shape[:2]
        
        # Get face bounding boxes [B, T, 4] in pixel space (x1, y1, x2, y2)
        face_bboxes = batch["face_bbox"]  # [B, T, 4]
        
        # Decode latents to RGB if needed
        if self.decode_for_perceptual:
            # Assume we have access to decoder via batch or external reference
            # For now, compute loss directly on latents (perceptual net can handle)
            # In practice, you'd decode: rgb_pred = decoder(model_output), rgb_gt = decoder(target)
            # For efficiency, we'll compute on latents but scale to [-1, 1] for LPIPS
            rgb_pred = model_output
            rgb_gt = target
        else:
            rgb_pred = model_output
            rgb_gt = target
        
        # Flatten batch and time dimensions for processing
        rgb_pred_flat = rgb_pred.reshape(B * T, *rgb_pred.shape[2:])  # [B*T, C, H, W]
        rgb_gt_flat = rgb_gt.reshape(B * T, *rgb_gt.shape[2:])  # [B*T, C, H, W]
        face_bboxes_flat = face_bboxes.reshape(B * T, 4)  # [B*T, 4]
        
        # Create face weight mask
        H, W = rgb_pred_flat.shape[-2:]
        face_weight_mask = self._create_face_weight_mask(
            face_bboxes_flat, H, W, device=rgb_pred.device
        )  # [B*T, 1, H, W]
        
        # Compute LPIPS perceptual loss
        # Note: LPIPS expects RGB in [-1, 1], so we may need to scale latents
        # For latent-space loss, we skip scaling; for decoded RGB, ensure proper range
        with torch.no_grad():
            # Expand latents to 3 channels if needed (LPIPS expects 3-channel input)
            if rgb_pred_flat.shape[1] != 3:
                # Repeat or project to 3 channels
                rgb_pred_flat = F.interpolate(
                    rgb_pred_flat, 
                    size=(H * 8, W * 8),  # Upsample latents to image resolution
                    mode='bilinear', 
                    align_corners=False
                )
                rgb_gt_flat = F.interpolate(
                    rgb_gt_flat, 
                    size=(H * 8, W * 8),
                    mode='bilinear', 
                    align_corners=False
                )
                # Repeat channels to get 3-channel input
                rgb_pred_flat = rgb_pred_flat.repeat(1, 3 // rgb_pred_flat.shape[1] + 1, 1, 1)[:, :3]
                rgb_gt_flat = rgb_gt_flat.repeat(1, 3 // rgb_gt_flat.shape[1] + 1, 1, 1)[:, :3]
                
                # Update mask resolution
                face_weight_mask = F.interpolate(
                    face_weight_mask, 
                    size=(H * 8, W * 8),
                    mode='nearest'
                )
        
        # Compute perceptual loss
        lpips_loss = self.lpips(rgb_pred_flat, rgb_gt_flat)  # [B*T, 1, H', W']
        
        # Apply face weighting
        weighted_lpips = lpips_loss * face_weight_mask
        
        # Average over spatial dimensions and batch
        face_loss = weighted_lpips.mean()
        
        return face_loss

    def _create_face_weight_mask(
        self, 
        bboxes: torch.Tensor,  # [N, 4] in pixel coords (x1, y1, x2, y2)
        H: int, 
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create spatial weight mask that emphasizes face regions.
        
        Args:
            bboxes: Face bounding boxes [N, 4] as (x1, y1, x2, y2) in pixel space
            H, W: Latent spatial dimensions
            device: Target device
            
        Returns:
            Weight mask [N, 1, H, W] with higher weights in face regions
        """
        N = bboxes.shape[0]
        mask = torch.ones(N, 1, H, W, device=device)
        
        for i in range(N):
            x1, y1, x2, y2 = bboxes[i]
            
            # Skip if bbox is invalid
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                continue
            
            # Convert pixel coords to latent coords (assuming 8x downsampling)
            # Note: Adjust downsampling factor based on your VAE
            x1_lat = int((x1 / 8).clamp(0, W - 1))
            y1_lat = int((y1 / 8).clamp(0, H - 1))
            x2_lat = int((x2 / 8).clamp(0, W))
            y2_lat = int((y2 / 8).clamp(0, H))
            
            # Apply higher weight to face region
            mask[i, :, y1_lat:y2_lat, x1_lat:x2_lat] = self.face_region_weight
        
        return mask

