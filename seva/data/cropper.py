import torch
from typing import Callable, Tuple, Optional, List
from einops import repeat

from seva.data.preprocessing import update_intrinsics, get_bbox_center_and_size

# NOTE: this should be applied to the OUTPUT 576x576 image shape AFTER initial cropping!
class RandomBBoxCropper(object):
    def __init__(self, random_crop=True, random_crop_prob=1.0, crop_size_bounds=None, padding=[0,0,0,0], face_crop_prob=0.5, face_bias_strength=0.7):
        """
        Random (Gaussian) crop transform centered around a 2D bounding box.
        NOTE: images are NOT resized to (576, 576) here!
        - padding: [left, top, right, bottom] (in pixels) only for deterministic crop!
        - face_crop_prob: probability of biasing crop towards face region when face bbox is available
        - face_bias_strength: strength of bias towards face (0.0 = no bias, 1.0 = fully centered on face)
        """
        self.crop_size_bounds = crop_size_bounds # (min_crop_size, max_crop_size)
        self.random_crop = random_crop # if maximal_crop only, then should be False
        self.random_crop_prob = random_crop_prob
        self.face_crop_prob = face_crop_prob
        self.face_bias_strength = face_bias_strength
        if not self.random_crop:
            self.random_crop_prob = 0.0

        if isinstance(padding, int) or isinstance(padding, float):
            # ! for now, should always be an int for uniform padding!
            self.padding = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            self.padding = padding
        else:
            raise ValueError(f"Invalid padding type: {type(padding)}")
        # if random_crop:
        #     self.padding = [0,0,0,0]

    def _get_crop_params(
        self, 
        bbox: torch.Tensor, 
        K: torch.Tensor,
        options: dict,
        face_bboxes: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Calculate crop parameters based on bbox and intrinsics.
        BBox is the initial "crop" onto the image, before random cropping (done here).

        NOTE: `pre_scale` will affect bbox parameters here.
        
        Args:
            bbox: Tensor of shape (B,4) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (B, 3, 3)
            options: Dictionary containing the following keys:
                - "W": Width of the image (used to clamp samples)
                - "H": Height of the image (used to clamp samples)
                - "center_mean": 2D list of mean values (x,y)_mean
                - "center_std": 2D list of std values (x,y)_std
                - "crop_size_mean": 1D list of mean values (crop_size)_mean
                - "crop_size_std": 1D list of std values (crop_size)_std
            face_bboxes: Tensor of shape (B, 4) with [x1, y1, x2, y2] for face regions
                - [-1, -1, -1, -1] indicates no face detected
            NOTE: mean & std can be in normalized (0-1) or absolute (pixel) values.
            
        Returns:
            Dictionary containing:
                - "bbox": Crop coordinates (B, 4)
                - "K": Updated intrinsics matrix (B, 3, 3)
                - "relative_bbox": Relative bbox coordinates to inconsistent images
        """
        W = options["W"]
        H = options["H"]
        B = bbox.shape[0]

        center, size = get_bbox_center_and_size(bbox)
        centers = torch.stack(center, dim=1)
        sizes = torch.stack(size, dim=1)
        center_x, center_y = centers.T
        bbox_W, bbox_H = sizes.T

        # Define canonical (body-centered) boundaries first
        bbox_max_dim_can = torch.maximum(bbox_W, bbox_H) # (B,) max dimension from raw bbox
        total_size_can = (bbox_max_dim_can + self.padding[0] + self.padding[2]).int() # canoncial size (synthetic square)
        x1_can = torch.floor(center_x - (bbox_max_dim_can // 2) - self.padding[0]).int()
        y1_can = torch.ceil(center_y - (bbox_max_dim_can // 2) - self.padding[1]).int()
        x2_can = x1_can + total_size_can
        y2_can = y1_can + total_size_can

        # Check if we should bias towards face regions
        if face_bboxes is not None:
            use_face_bias = torch.rand(B) < self.face_crop_prob
        else:
            use_face_bias = torch.zeros(B, dtype=torch.bool)
        
        # If face bboxes are provided and valid, bias the base centers towards them
        bbox_max_dim = bbox_max_dim_can.clone()
        if face_bboxes is not None:
            # Identify valid face bboxes (false for no-face indicator: [-1, -1, -1, -1])
            valid_faces = face_bboxes[:, 0] != -1
            apply_face_bias = use_face_bias & valid_faces
            
            if apply_face_bias.any():
                # Calculate face centers for valid faces
                face_center_x = (face_bboxes[:, 0] + face_bboxes[:, 2]) / 2.0
                face_center_y = (face_bboxes[:, 1] + face_bboxes[:, 3]) / 2.0
                
                # Blend between body center and face center based on bias strength
                center_x[apply_face_bias] = (
                    (1 - self.face_bias_strength) * center_x[apply_face_bias] + 
                    self.face_bias_strength * face_center_x[apply_face_bias]
                )
                center_y[apply_face_bias] = (
                    (1 - self.face_bias_strength) * center_y[apply_face_bias] + 
                    self.face_bias_strength * face_center_y[apply_face_bias]
                )

                # Zoom in more if we are gravitating towards a face (hardcoded a empirically good value)
                bbox_max_dim[apply_face_bias] = bbox_max_dim[apply_face_bias] * (0.30 + (torch.rand(B, device=bbox.device)[apply_face_bias] * 2 - 1) * 0.20)
                # Ensure zoomed bbox is not larger than canonical
                bbox_max_dim[apply_face_bias] = torch.clamp(bbox_max_dim[apply_face_bias], max=total_size_can[apply_face_bias])

        # 'Current' total size (potentially face-biased)
        total_size = (bbox_max_dim + self.padding[0] + self.padding[2]).int()

        # Calculate coordinates and clamp them to stay within the canonical square
        x1 = torch.floor(center_x - (bbox_max_dim // 2) - self.padding[0]).int()
        y1 = torch.ceil(center_y - (bbox_max_dim // 2) - self.padding[1]).int()
        
        x1 = torch.clamp(x1, min=x1_can, max=x2_can - total_size)
        y1 = torch.clamp(y1, min=y1_can, max=y2_can - total_size)
        
        x2 = x1 + total_size
        y2 = y1 + total_size
        
        # Update centers for random crop logic to be centered on the deterministic crop
        centers = torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dim=1)
        
        # rel_bbox will store the total delta from canonical crop to final crop
        rel_bbox = torch.zeros(B, 4)

        # NOTE: when face crops are applied, we do NOT apply the random crop!
        to_random_crop = options.get("to_crop", False)
        if self.random_crop and to_random_crop:
            # Identify indices that are NOT face-biased
            random_indices = ~use_face_bias # if no face_bbox, all True
            
            if random_indices.any():
                center_mean    = options.get("center_mean", centers)
                center_std     = options.get("center_std", torch.stack([(W - bbox_W) / 6, (H - bbox_H) / 6], dim=1))
                crop_size_mean = options.get("crop_size_mean", (bbox_W + bbox_H) * 3 / 4)
                crop_size_std  = options.get("crop_size_std", (bbox_W + bbox_H) / 2)
                min_crop_size  = options.get("min_crop_size", (bbox_max_dim * 3) // 4)

                center_mean    = percent_to_absolute(center_mean, torch.tensor([H, W]))
                center_std     = torch.as_tensor(center_std)
                crop_size_mean = percent_to_absolute(crop_size_mean, torch.tensor([min(H, W)]))
                crop_size_std  = torch.as_tensor(crop_size_std)

                size_sample = torch.clamp(torch.randn(B) * crop_size_std + crop_size_mean, min=min_crop_size, max=bbox_max_dim)

                if self.crop_size_bounds is not None:
                    size_sample = torch.clamp(
                        size_sample,
                        min=percent_to_absolute(self.crop_size_bounds[0], torch.tensor([min(H, W)])),
                        max=percent_to_absolute(self.crop_size_bounds[1], torch.tensor([min(H, W)]))
                    )

                size_sample_int = size_sample.int()

                x_offset = torch.clamp(
                    torch.randn(B,1) * center_std[:,0].view(-1,1) + center_mean[:,0].view(-1,1),
                    min=(x1 + size_sample_int // 2).view(-1, 1),
                    max=(x2 - size_sample_int // 2).view(-1, 1)
                )
                y_offset = torch.clamp(
                    torch.randn(B,1) * center_std[:,1].view(-1,1) + center_mean[:,1].view(-1,1),
                    min=(y1 + size_sample_int // 2).view(-1, 1),
                    max=(y2 - size_sample_int // 2).view(-1, 1)
                )

                # random crop coordinates
                x1_new = torch.floor(x_offset - (size_sample_int // 2).view(-1, 1)).int().view(-1)
                y1_new = torch.floor(y_offset - (size_sample_int // 2).view(-1, 1)).int().view(-1)
                x2_new = x1_new + size_sample_int.view(-1)
                y2_new = y1_new + size_sample_int.view(-1)

                # ONLY apply the new coordinates to non-face-biased indices
                x1[random_indices] = x1_new[random_indices]
                y1[random_indices] = y1_new[random_indices]
                x2[random_indices] = x2_new[random_indices]
                y2[random_indices] = y2_new[random_indices]

            # Store deltas relative to canonical boundaries [dx1, dy1, dx2, dy2]
            # dx1, dy1: shift from top-left (positive)
            # dx2, dy2: shift from bottom-right (negative)
            rel_bbox[:, 0] = x1 - x1_can
            rel_bbox[:, 1] = y1 - y1_can
            rel_bbox[:, 2] = x2 - x2_can
            rel_bbox[:, 3] = y2 - y2_can
        else:
            # Absolute coordinates relative to canonical square
            rel_bbox[:, 0] = x1 - x1_can
            rel_bbox[:, 1] = y1 - y1_can
            rel_bbox[:, 2] = x2 - x2_can
            rel_bbox[:, 3] = y2 - y2_can

        if len(K.shape) == 2:
            K_ = repeat(K, 'd1 d2 -> n d1 d2', n=B).detach().clone()
        else:
            K_ = K.detach().clone()

        K_new = update_intrinsics(
            torch.as_tensor(K_), 
            crop_x=x1,
            crop_y=y1,
            scale=1,
            crop_first=False,
            padding_mode=True
        )

        # Scale rel_bbox relative to the canonical size (mapped to 576)
        scale = 576.0 / total_size_can # accounts for padding
        rel_bbox = (rel_bbox * scale.view(-1, 1)).int()

        return {
            "bbox": torch.stack([x1, y1, x2, y2], dim=1),
            "K": K_new,
            "relative_bbox": rel_bbox
        }


    def _possibly_pad_img(self, images, x1, y1, x2, y2):
        """
        Pad the image if the crop parameters extend beyond the image.
        """
        # handle padding if needed
        H, W = images.shape[-2:]
        pad_left = torch.maximum(torch.zeros_like(x1), -x1)
        pad_top = torch.maximum(torch.zeros_like(y1), -y1)
        pad_right = torch.maximum(torch.zeros_like(x2), x2 - W)
        pad_bottom = torch.maximum(torch.zeros_like(y2), y2 - H)
        
        # if the new crop parameters extend beyond the image, pad the image
        if torch.any(pad_left > 0) or torch.any(pad_top > 0) or torch.any(pad_right > 0) or torch.any(pad_bottom > 0):
            # print("WARNING: Crop parameters extend beyond the image!")
            image_list = []
            padding = torch.stack([pad_left.int(), pad_right.int(), pad_top.int(), pad_bottom.int()], dim=1)

            # images is a list of different sized images, so need to iterate separately
            for i, image in enumerate(images):
                image = torch.nn.functional.pad(image, padding[i].tolist(), mode="constant", value=0)
                image_list.append(image) # keep list since padding is different for each image

            # ! already done in get_crop_params
            # K_new = update_intrinsics(
            #     K,
            #     crop_x=-pad_left,
            #     crop_y=-pad_top,
            #     scale=1,
            #     crop_first=False,
            #     padding_mode=True
            # )
            new_bbox = torch.stack([x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top], dim=1)
            return image_list, new_bbox
        else:
            return images, torch.stack([x1, y1, x2, y2], dim=1)

    def crop_images(self, images, x1, y1, x2, y2):
        """
        Crop images based on bounding box.
        """
        cropped_images = []
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        for i in range(len(images)):
            if len(images[i].shape) == 2:
                cropped_img = images[i][int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            else:
                cropped_img = images[i][:, int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            cropped_images.append(cropped_img)
        return cropped_images

    def __call__(
        self, 
        images: torch.Tensor, 
        bbox: torch.Tensor, 
        K: torch.Tensor,
        face_bboxes: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[list, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, C, H, W)
            bbox: Tensor of shape (B, 4) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (B, 3, 3)
            face_bboxes: Tensor of shape (B, 4) with [x1, y1, x2, y2]
                - [-1, -1, -1, -1] used to indicate no face detected
        Returns:
            Cropped image and updated intrinsics matrix
        """

        # kwargs will always include image size metadata
        # if other parameters (mean, std) are NOT provided, then, we use the center as the mean,
        # and the crop shape
        options = {
            "H": images.shape[-2],
            "W": images.shape[-1],
            "to_crop": True if torch.rand(1) < self.random_crop_prob else False
        }
        options.update(kwargs) # bias towards face based on "annots" face box!

        # get new crop parameters (pass face_bboxes for potential face-biased cropping)
        crop_params = self._get_crop_params(bbox, K, options, face_bboxes=face_bboxes) # * GOOD
        bbox = crop_params["bbox"]
        K_new = crop_params["K"]
        rel_bbox = crop_params["relative_bbox"] # for cropping ic images (scaled to 576^2)

        # get new crop coordinates
        x1, y1, x2, y2 = bbox.T
        prev_pad_bbox = bbox.clone()
        # if negative coordinates, need to pad the image (K already previously updated)
        images, bbox = self._possibly_pad_img(images, x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox.T

        face_bboxes_new = None
        if face_bboxes is not None:
            # reposition face wrt new corner point
            # scale this to target shape INTERNALLY (here)!
            face_bboxes_new = face_bboxes.to(torch.float32)  # Convert to float for arithmetic operations
            no_face_mask = face_bboxes[:,0] != -1
            face_bboxes_new[no_face_mask, 0] = face_bboxes_new[no_face_mask, 0] - x1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask, 1] = face_bboxes_new[no_face_mask, 1] - y1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask, 2] = face_bboxes_new[no_face_mask, 2] - x1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask, 3] = face_bboxes_new[no_face_mask, 3] - y1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask] = face_bboxes_new[no_face_mask] * (576.0 / torch.maximum((x2 - x1)[no_face_mask].to(torch.float32), (y2 - y1)[no_face_mask].to(torch.float32)).unsqueeze(-1)) # ! HARDCODED to 576
            face_bboxes_new[~no_face_mask] = -1 # just to ensure
            # if any become out-of-bounds post random crop, then set to -1 as well
            oob_mask = (face_bboxes_new < 0).any(dim=-1) | (face_bboxes_new > 576.0).any(dim=1)
            face_bboxes_new[oob_mask] = -1
            
            face_bboxes_new = face_bboxes_new.to(torch.int32)  # Convert back to int32 for indexing

        # perform the actual crop
        cropped_images = self.crop_images(images, x1, y1, x2, y2)
        return cropped_images, K_new, rel_bbox, face_bboxes_new if face_bboxes is not None else None, bbox.int(), prev_pad_bbox.int()


def percent_to_absolute(arr, abs_arr):
    _arr = torch.as_tensor(arr)
    orig_shape = _arr.shape
    _arr = _arr.reshape(-1)
    decimal_mask = (torch.where((_arr <= 1) & (_arr >= 0))[0]).to(torch.int32)
    if len(decimal_mask) == 0:
        return _arr.reshape(orig_shape).to(torch.float32)
    _arr[decimal_mask] = _arr[decimal_mask] * abs_arr # convert to pixel coords
    return _arr.reshape(orig_shape).to(torch.float32)


# use for later; we'll need this to convert the bbox from crop_params.npz to centered square
# NOTE: bbox values can be negative, in which case, we'll need to pad the image to fit (during runtime)
# REMEMBER TO UPDATE INTRINSICS!
# - for our inconsistent dataset, we'll use this to explicitly crop images to square, then reshape to 576x576 to put into pipeline
# - for our MVHN dataloader, we only explicitly crop the image (and update intrinsics) when we need it (during runtime)
# ! - DATALOADER CURRENTLY HAS K NORMALIZED INTRINSICS! Be sure to do the cropping inside the dataloader!
def convert_to_square_crop(bbox):
    """
    Convert a bbox to a square crop.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    crop_size = max(w, h)
    center = (x1 + (w // 2), y1 + (h // 2))
    x1 = center[0] - (crop_size // 2)
    y1 = center[1] - (crop_size // 2)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return (x1, y1, x2, y2)