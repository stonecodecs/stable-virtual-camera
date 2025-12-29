from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser
 

def pad_to(latent, target_size, relative=False):
    """
    Center pads the latent to a target size.
    If relative, then we pad relative to the CURRENT size of 'latent'.
    Relative padding makes target_size 4D tensor for the padding values
    (from 2nd return value of this function!)
    (NOTE: if RGB image is 'latent' this will be used to pad the RGB image correspondingly.)
    """
    # latent is [B, C, H, W]
    B, C, H, W = latent.shape

    if relative:
        # then padding is actually 4D!
        return torch.nn.functional.pad(latent, target_size) \
               , (0, 0, 0, 0) # save padding values

    if isinstance(target_size, int):
        target_size = (target_size, target_size)
        
    H_target, W_target = target_size

    if H == H_target and W == W_target:
        return latent, (0, 0, 0, 0)
    
    # Calculate padding for height
    pad_h_total = max(0, H_target - H)
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    
    # Calculate padding for width
    pad_w_total = max(0, W_target - W)
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left
    
    # The padding format is (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(latent, (pad_left, pad_right, pad_top, pad_bottom)) \
           , (pad_left, pad_right, pad_top, pad_bottom) # save padding values

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        face_weighting: float = 0.0,
        arcface_loss_weight: float = 0.0,
        arcface_gt_key: str = "arcface_embedding",
        background_downweight: float = 0.0,
        depth_loss_weight: float = 0.0,
        seg_loss_weight: float = 0.0,
        **kwargs, # absorb unknown keys
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level
        self.face_weighting = face_weighting  # how much to weigh the face over the rest
        # 0.0 -> no extra face weighting, spatially uniform loss weighting
        self.arcface_loss_weight = arcface_loss_weight
        self.arcface_gt_key = arcface_gt_key
        self.background_downweight = background_downweight
        self.depth_loss_weight = depth_loss_weight
        self.seg_loss_weight = seg_loss_weight
        # Store reference to first_stage_model for RGB decoding (set externally)
        self.first_stage_model = None
        self.scale_factor = None

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()
        
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
        loss = self.get_loss(
            model_output,
            input,
            w,
            face_bbox=batch.get("face_bbox"),
            ref_mask=batch.get("ref_mask"),
            enable_face_weighting=self.face_weighting > 0.0,
            loss_mask=batch.get("frames_masks", None),
        )

        # Add auxiliary ArcFace identity loss if enabled
        if self.training and self.arcface_loss_weight > 0.0:
            arcface_loss = self.get_arcface_loss(network, batch)
            loss = loss + self.arcface_loss_weight * arcface_loss
            # clear for VRAM
            if hasattr(network, "diffusion_model") and hasattr(
                network.diffusion_model, "seva_model"
            ):
                network.diffusion_model.seva_model.predicted_arcface_embedding = None
            else:
                network.predicted_arcface_embedding = None

        # Add depth and segmentation losses if enabled
        if self.training and self.depth_loss_weight > 0.0:
            depth_loss = self.get_depth_loss(network, batch)
            loss = loss + self.depth_loss_weight * depth_loss
            if hasattr(network, "diffusion_model") and hasattr(
                network.diffusion_model, "seva_model"
            ):
                network.diffusion_model.seva_model.depth_pred = None
            else:
                network.depth_pred = None

        if self.training and self.seg_loss_weight > 0.0:
            seg_loss = self.get_seg_loss(network, batch)
            loss = loss + self.seg_loss_weight * seg_loss
            if hasattr(network, "diffusion_model") and hasattr(
                network.diffusion_model, "seva_model"
            ):
                network.diffusion_model.seva_model.seg_pred = None
            else:
                network.seg_pred = None

        return loss

    def get_arcface_loss(self, network: nn.Module, batch: Dict) -> torch.Tensor:
        """
        Computes the cosine similarity loss between predicted and ground-truth ArcFace embeddings.
        """
        # Get device and dtype from network parameters to ensure consistency
        device = next(network.parameters()).device
        dtype = next(network.parameters()).dtype
        
        # The path to the Seva model might vary depending on wrappers
        if hasattr(network, "diffusion_model") and hasattr(
            network.diffusion_model, "seva_model"
        ):
            predicted_embed = (
                network.diffusion_model.seva_model.predicted_arcface_embedding
            )
        else:
            predicted_embed = network.predicted_arcface_embedding

        if predicted_embed is None:
            return torch.tensor(0.0, device=device, dtype=dtype)

        gt_embed = batch.get(self.arcface_gt_key)
        face_mask = torch.any(gt_embed, dim=2) # zero tensors don't count

        if gt_embed is None or not face_mask.any():
            return torch.tensor(0.0, device=device, dtype=dtype)

        if predicted_embed.shape[0] != gt_embed.shape[0]:
            num_frames = predicted_embed.shape[0] // gt_embed.shape[0]
            # Move to device and dtype before repeat to avoid intermediate device transfers
            gt_embed_device = gt_embed.to(device=predicted_embed.device, dtype=predicted_embed.dtype)
            gt_embed = repeat(gt_embed_device, "b ... -> (b f) ...", f=num_frames)
            del gt_embed_device  # Clear intermediate tensor
        else:
            gt_embed = gt_embed.to(device=predicted_embed.device, dtype=predicted_embed.dtype)

        # get average embedding for gt_embed, compare with predicted_embed
        gt_embed_sum = gt_embed.sum(dim=1)
        gt_embed_count = face_mask.sum(dim=1)
        gt_embed_avg = gt_embed_sum / gt_embed_count.unsqueeze(-1)
        gt_embed = gt_embed_avg.unsqueeze(1)
        predicted_embed = predicted_embed * face_mask.unsqueeze(-1)

        # if nans, replace with 0
        gt_embed = torch.where(torch.isnan(gt_embed), torch.zeros_like(gt_embed), gt_embed)
        predicted_embed = torch.where(torch.isnan(predicted_embed), torch.zeros_like(predicted_embed), predicted_embed)
        
        # Normalize both embeddings before comparing
        gt_embed_norm = F.normalize(gt_embed, p=2, dim=2)
        predicted_embed_norm = F.normalize(predicted_embed, p=2, dim=2)

        loss = 1.0 - F.cosine_similarity(predicted_embed_norm, gt_embed_norm, dim=1)
        loss_mean = loss.mean(dim=1)

        del predicted_embed_norm, gt_embed_norm, loss
        return loss_mean

    def get_face_weighting_loss(self, face_bbox, spatial_loss, ref_mask):
        """
        Compute weighted loss that emphasizes face regions.
        
        Args:
            face_bbox: [B, T, 4] in pixel coords (x1, y1, x2, y2)
            spatial_loss: [B, T, C, H, W] spatial loss map
            ref_mask: [B, T] boolean tensor indicating reference frames (should be excluded)
            
        Returns:
            Face-weighted loss scalar per batch element [B]
        """
        B, T, C, H, W = spatial_loss.shape
        
        # Compute face loss per batch element directly without creating large boolean mask
        # This avoids creating a [B, T, C, H, W] boolean tensor which can be memory intensive
        face_loss = torch.zeros(B, device=spatial_loss.device, dtype=spatial_loss.dtype)
        
        for b in range(B):
            batch_face_losses = []
            for t in range(T):
                # Skip reference frames (ground truth)
                if ref_mask[b, t]:
                    continue
                    
                x1, y1, x2, y2 = face_bbox[b, t].long()
                
                # Skip invalid bboxes
                if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                    continue
                
                # Convert pixel coords to latent coords (8x downsampling)
                x1_lat = (x1 // 8).clamp(0, W - 1)
                y1_lat = (y1 // 8).clamp(0, H - 1)
                x2_lat = (x2 // 8).clamp(1, W)
                y2_lat = (y2 // 8).clamp(1, H)
                
                if x1_lat >= x2_lat or y1_lat >= y2_lat:
                    continue
                
                # Extract face region and compute mean loss directly
                face_region_loss = spatial_loss[b, t, :, y1_lat:y2_lat, x1_lat:x2_lat]
                batch_face_losses.append(face_region_loss.mean())
            
            if batch_face_losses:
                face_loss[b] = torch.stack(batch_face_losses).mean()
        
        return face_loss


    def get_loss(self, model_output, target, w, face_bbox=None, ref_mask=None, enable_face_weighting=False, loss_mask=None):
        # add face weighting if face_weighting > 0.0
        additional_loss = torch.tensor(0.0, device=model_output.device, dtype=model_output.dtype)
        if loss_mask is not None:
            assert loss_mask.shape[0] == model_output.shape[0], f"Loss mask batch size mismatch. Got {loss_mask.shape[0]} but expected {model_output.shape[0]}."
            # use F.interpolate to resize the loss mask to the latent spatial dimensions
            # HACK: hardcoded 8x downsampling
            loss_mask = F.interpolate(
                loss_mask.flatten(start_dim=0, end_dim=1), size=model_output.shape[-2:], mode='bilinear'
            ).unflatten(dim=0, sizes=model_output.shape[:2])
            loss_mask = torch.clamp(loss_mask, min=self.background_downweight, max=1.0).float() # downweight background by 100x (but not zero!)
 
        if self.loss_type == "l2":
            spatial_loss = w * (model_output - target) ** 2 * loss_mask# [B, T, C, H, W]
            loss = torch.mean(
                spatial_loss.reshape(target.shape[0], -1), 1
            )
            if enable_face_weighting and face_bbox is not None and len(face_bbox) > 0 and ref_mask is not None:
                additional_loss = self.face_weighting * self.get_face_weighting_loss(face_bbox, spatial_loss, ref_mask)
                loss = loss + additional_loss
            return loss
        elif self.loss_type == "l1":
            spatial_loss = w * (model_output - target).abs() * loss_mask
            loss = torch.mean(
                spatial_loss.reshape(target.shape[0], -1), 1
            )
            if enable_face_weighting and face_bbox is not None and len(face_bbox) > 0 and ref_mask is not None:
                additional_loss = self.face_weighting * self.get_face_weighting_loss(face_bbox, spatial_loss, ref_mask)
                loss = loss + additional_loss
            return loss
        elif self.loss_type == "lpips": # only really usable in RGB space
            loss = self.lpips(model_output, target).reshape(-1)
            if enable_face_weighting and face_bbox is not None and len(face_bbox) > 0 and ref_mask is not None:
                additional_loss = self.face_weighting * self.get_face_weighting_loss(face_bbox, loss, ref_mask)
                loss = loss + additional_loss
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")


    # ! DEPRECATED!
    def get_face_crop_perceptual_loss(
        self,
        model_output: torch.Tensor,  # Predicted latents [B, T, C, H, W]
        target: torch.Tensor,        # Target latents [B, T, C, H, W] (not used, gt_frames is used instead)
        batch: Dict,
    ) -> torch.Tensor:
        """
        Compute perceptual loss on face-cropped regions in RGB space.
        Only processes non-reference frames (where ref_mask is False).
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

        B, T, C, H, W = model_output.shape # H, W are latent dim 72x72 spatial dims

        # cropped model output (assuming that face output is in the same position)
        face_bboxes_flat = batch["face_bbox"].reshape(B * T, 4)
        ref_mask_flat = batch["ref_mask"].reshape(B * T)  # Flatten ref_mask to match
        model_output = model_output.reshape(B * T, C, H, W)
        rgb_gt = batch["frames"].reshape(B * T, 3, H * 8, W * 8) # these are 576^2
        
        # Get face bounding boxes and flatten them
        cropped_pred_latents = []
        cropped_gt_rgbs = []

        # 2. Loop through the batch to CROP the RGB images (skip reference frames)
        for i in range(B * T):
            # Skip reference frames (ground truth)
            if ref_mask_flat[i]:
                continue
                
            x1, y1, x2, y2 = face_bboxes_flat[i].long()
            
            if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                continue
            
            x1_lat, y1_lat = x1//8, y1//8
            x2_lat, y2_lat = x2//8, y2//8

            if x1_lat >=x2_lat or y1_lat >= y2_lat:
                continue
                
            # crop the "face region" @ the latent level
            # latents are spatially aligned well enough for this to be valid
            pred_crop = model_output[i:i+1, :, y1_lat:y2_lat, x1_lat:x2_lat]
            gt_crop = rgb_gt[i:i+1, :, y1:y2, x1:x2] # this is in RGB space
            
            cropped_pred_latents.append(pred_crop)
            cropped_gt_rgbs.append(gt_crop)

        # If no valid faces were found in the batch, return zero loss
        # NOTE: why not return 0 tensor?
        # >> A: need dummy run to keep the comp. graph static and avoid deadlocks during multi-gpu training
        # need to experiment more if this is "worth it" compared to overhead from dynamic comp. graph
        if not cropped_pred_latents:
            dummy_latent = torch.zeros(1, C, 1, 1, device=model_output.device, dtype=model_output.dtype)
            dummy_pred_rgb = self.first_stage_model.decode(dummy_latent)
            dummy_gt_rgb = torch.zeros_like(dummy_pred_rgb)
            dummy_loss = self.lpips(dummy_pred_rgb, dummy_gt_rgb)
            return dummy_loss.mean() * 0.0

        # get maximum spatial size of crops
        # will be used for stacking and corresponding RGB GT padding
        # to align with the padded latent stack
        # Why? -> takes around 20s/iter based on the list approach.
        # This takes [TBD].
        max_W = max(crop.shape[-1] for crop in cropped_pred_latents)
        max_H = max(crop.shape[-2] for crop in cropped_pred_latents)

        # If faces are found, pad latents to uniform size
        decoded_crop_latents = []
        rel_padding = []
        for crop in cropped_pred_latents:
            padded_crop, rel_pad = pad_to(crop, (max_H, max_W), relative=False)
            decoded_crop_latents.append(padded_crop)
            rel_padding.append(rel_pad)

        # stack and decode
        decoded_crop_latents = self.first_stage_model.decode(torch.cat(decoded_crop_latents, dim=0))

        # pad the RGB GTs accordingly with zero-pad
        # to "align" with the padded decoded latents
        # also, resize to target size
        padded_gt_rgbs = []
        for crop, rel_pad in zip(cropped_gt_rgbs, rel_padding):
            padded_crop, rel_pad = pad_to(crop, [pad * 8 for pad in rel_pad], relative=True)
            padded_gt_rgbs.append(F.interpolate(padded_crop, size=(self.face_crop_size, self.face_crop_size), mode='bilinear', align_corners=False))

        padded_gt_rgbs = torch.cat(padded_gt_rgbs, dim=0)

        # 3. Resize decoded latents to a fixed size (in RGB space)
        # matching the padded_gt_rgbs
        resized_pred_crops = torch.cat([
            F.interpolate(decoded_crop_latents, size=(self.face_crop_size, self.face_crop_size), mode='bilinear', align_corners=False)
        ], dim=0)
        
        # 4. Compute LPIPS loss on the batched, resized RGB crops
        # chunk_size = 4 -- use later if no space
        lpips_loss = self.lpips(resized_pred_crops, padded_gt_rgbs)
        return lpips_loss.mean()

    def get_depth_loss(self, network: nn.Module, batch: Dict) -> torch.Tensor:
        """
        Get L1 loss for network depth predictions against GT "sapiens" depth maps.
        """
        device = next(network.parameters()).device
        dtype = next(network.parameters()).dtype

         # The path to the Seva model might vary depending on wrappers
        if hasattr(network, "diffusion_model") and hasattr(
            network.diffusion_model, "seva_model"
        ):
            depth_pred = (
                network.diffusion_model.seva_model.depth_pred
            )
        else:
            depth_pred = network.depth_pred

        if depth_pred is None:
            return torch.tensor(0.0, device=device, dtype=dtype)

        B, T = batch['mask'].shape[:2]
        depth_pred = depth_pred.reshape(B, T, *depth_pred.shape[-3:])

        sapiens_conditioning = batch.get("sapiens_conditioning")
        if sapiens_conditioning is None or "depth" not in sapiens_conditioning:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        gt_depth = sapiens_conditioning["depth"]
        # resize gt_depth to depth_pred spatially
        gt_depth_resized = F.interpolate(
            gt_depth.view(B * T, *gt_depth.shape[2:]),
            size=depth_pred.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).view(B, T, 1, *depth_pred.shape[-2:])

        return F.l1_loss(depth_pred, gt_depth_resized, reduction='none').mean(dim=(2, 3, 4)).mean(dim=1)
        
    def get_seg_loss(self, network: nn.Module, batch: Dict) -> torch.Tensor:
        """
        Get L1 loss for network segmentation predictions against GT "sapiens" segmentation maps.
        """
        device = next(network.parameters()).device
        dtype = next(network.parameters()).dtype

        # The path to the Seva model might vary depending on wrappers
        if hasattr(network, "diffusion_model") and hasattr(
            network.diffusion_model, "seva_model"
        ):
            seg_pred = (
                network.diffusion_model.seva_model.seg_pred
            )
        else:
            seg_pred = network.seg_pred
        
        sapiens_conditioning = batch.get("sapiens_conditioning")
        if seg_pred is None or sapiens_conditioning is None or "seg_masks" not in sapiens_conditioning:
            return torch.tensor(0.0, device=device, dtype=dtype)

        gt_seg = sapiens_conditioning["seg_masks"]
        # Convert one-hot to class indices for cross-entropy
        # gt_seg is one-hot: [B, T, C, H, W] -> class indices: [B, T, H, W]
        if gt_seg.shape[2] > 1:  # One-hot encoded
            gt_seg_indices = gt_seg.argmax(dim=2)  # [B, T, H, W]
        else:
            gt_seg_indices = gt_seg.squeeze(2)  # [B, T, H, W]
    
        B, T = batch['mask'].shape[:2]
        seg_pred = seg_pred.reshape(B, T, *seg_pred.shape[-3:])  # [B, T, C, H, W]
        # seg_pred should remain as logits [B, T, C, H, W] for cross-entropy loss
        
        # Reshape gt_seg_indices to [B*T, 1, H, W] for interpolation
        gt_seg_indices_4d = gt_seg_indices.unsqueeze(2).float()  # [B, T, 1, H, W]
        gt_seg_indices_4d = gt_seg_indices_4d.view(B * T, 1, *gt_seg_indices.shape[2:])  # [B*T, 1, H, W]
        
        # Resize to match seg_pred spatial dimensions
        gt_seg_resized = F.interpolate(
            gt_seg_indices_4d,
            size=seg_pred.shape[-2:],  # [H, W] from [B, T, C, H, W]
            mode='nearest',
        ).squeeze(1).long()  # [B*T, H, W]

        # Reshape seg_pred to [B*T, C, H, W] for cross-entropy
        seg_pred_flat = seg_pred.view(B * T, *seg_pred.shape[2:])  # [B*T, C, H, W]
        
        seg_loss = F.cross_entropy(seg_pred_flat, gt_seg_resized, reduction='none')  # [B*T, H, W]
        seg_loss = seg_loss.mean(dim=(1, 2))  # [B*T]
        seg_loss = seg_loss.view(B, T).mean(dim=1)  # [B,]
        return seg_loss

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