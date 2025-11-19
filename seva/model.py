from dataclasses import dataclass, field

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from seva.modules.layers import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from seva.modules.transformer import MultiviewTransformer
from typing import Union


from safetensors.torch import load_file

class ArcFaceHead(nn.Module):
    """
    Projects features from the middle block of the U-Net to the ArcFace embedding space.
    """

    def __init__(self, in_channels: int, out_channels: int = 512, num_images: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(in_channels)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        x = x.flatten(start_dim=1, end_dim=3)
        x = self.norm(x)
        x = self.proj(x)
        x = x.reshape(-1, num_images, self.out_channels) # ! 8 hardcoded for now
        return x


@dataclass
class SevaParams(object):
    in_channels: int = 11
    model_channels: int = 320
    out_channels: int = 4
    num_frames: int = 21
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_head_channels: int = 64
    transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    context_dim: int = 1024
    dense_in_channels: int = 6
    dropout: float = 0.0
    unflatten_names: list[str] = field(
        default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
    )
    ckpt_path: str | None = None
    use_ip_adapter: bool = False
    face_context_dim: int = 512
    use_id_head: bool = True

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.transformer_depth)


class Seva(nn.Module):
    def __init__(self, params: SevaParams, freeze_layers:bool=False, load_pretrained:bool=True) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                            use_ip_adapter=params.use_ip_adapter,
                            face_context_dim=params.face_context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
                use_ip_adapter=params.use_ip_adapter,
                face_context_dim=params.face_context_dim,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch
        if params.use_id_head:
            self.arcface_head = ArcFaceHead(in_channels=ch, num_images=params.num_frames)
        else:
            self.arcface_head = None

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                            use_ip_adapter=params.use_ip_adapter,
                            face_context_dim=params.face_context_dim,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )
        self.predicted_arcface_embedding = None

        if load_pretrained:
            from seva.utils import print_load_warning
            state_dict = load_seva_state_dict(params)
            missing, unexpected = self.load_state_dict(state_dict, assign=True)
            print_load_warning(missing, unexpected)
        
        if params.ckpt_path is not None:
            from seva.utils import print_load_warning
            state_dict = load_file(params.ckpt_path)
            missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)
            print_load_warning(missing, unexpected)

        if freeze_layers:
            self.freeze()
    
    def freeze(self, layers: list[str] = ["middle", "output"]):
        if "input" in layers:
            input_param_gen = self.input_blocks.named_parameters()
            for (name, param) in input_param_gen:
                if name.startswith("0.0"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if "middle" in layers:
            for param in self.middle_block.parameters():
                param.requires_grad = False
        if "output" in layers:
            for param in self.output_blocks.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
        face_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                face_context=face_context,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            face_context=face_context,
        )

        if self.training and face_context is not None and self.arcface_head is not None:
            self.predicted_arcface_embedding = self.arcface_head(h)
        else:
            self.predicted_arcface_embedding = None

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                face_context=face_context,
            )
        h = h.type(x.dtype)
        return self.out(h) # [B*num_images, C=4, H=72, W=72]


def load_seva_state_dict(
    params: SevaParams = SevaParams(),
    pretrained_model_name_or_path: str = "stabilityai/stable-virtual-camera",
    weight_name: str = "model.safetensors",
    device: str | torch.device = "cuda",
):
    from seva.utils import download_pretrained_checkpoint
    state_dict = download_pretrained_checkpoint(pretrained_model_name_or_path, weight_name, device)
    input_block_c1 = state_dict["input_blocks.0.0.weight"] # reshape this based on params.in_channels
    input_block_b1 = state_dict["input_blocks.0.0.bias"]
    new_input_block = nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
    new_input_block.weight.data[:, :input_block_c1.shape[1], :, :] = input_block_c1
    new_input_block.weight.data[:, input_block_c1.shape[1]:, :, :] = 0 # zeros for the rest
    new_input_block.bias.data[:input_block_b1.shape[0]] = input_block_b1
    new_input_block.bias.data[input_block_b1.shape[0]:] = 0 # zeros for the rest
    new_state_dict = {
        "input_blocks.0.0.weight": new_input_block.weight,
        "input_blocks.0.0.bias": new_input_block.bias,
    }
    for k, v in state_dict.items():
        if not k.startswith("input_blocks.0.0"):
            new_state_dict[k] = v

    return new_state_dict


# for compatibility with SGM
class SGMWrapper(nn.Module):
    def __init__(self, module: Seva): # or SevaLoRAWrappers
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        # c: crossattn, concat, dense_vector, face_cond
        # kwargs 'num_frames'
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        # 16,11,72,72 (concat is 7, latent x is 4)
        return self.module(
            x,
            t=t,
            y=c["crossattn"],
            dense_y=c["dense_vector"],
            face_context=c.get("face_cond"),
            **kwargs,
        )
