import torch
import torch.nn as nn
from typing import Optional, Dict, Any, cast, List, Union
from .lora import LoRALinear, LoRAAttentionWrapper, LoRAFeedForwardWrapper
from ..model import Seva, SevaParams
from sgm.util import instantiate_from_config


def skip_module_if_excluded(module_path: str, excluded_modules: list[str]) -> bool:
    path_parts = module_path.split('.')
    for i in range(len(path_parts)):
        parent_path = '.'.join(path_parts[:i+1])
        if parent_path in excluded_modules:
            return True
    return False

# In seva/modules/lora_wrapper.py

class SevaLoRAWrapper(nn.Module):
    def __init__(
        self,
        seva_model_config: Dict[str, Any],
        self_attn_rank: int = 4,
        cross_attn_rank: int = 8,
        ff_rank: int = 8,
        alpha: Union[float, List] = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        keys_to_lora: list[str] = ["q", "k", "v"],
        excluded_modules: list[str] = [],
        lora_for_face_attn_only: bool = False,
        freeze_lora: bool = False,
    ):
        super().__init__()
        self.seva_model: nn.Module = cast(nn.Module, instantiate_from_config(seva_model_config))
        self.self_attn_rank = self_attn_rank
        self.cross_attn_rank = cross_attn_rank
        self.ff_rank = ff_rank
        # Support per-component alpha: [self_attn_alpha, cross_attn_alpha, ff_alpha]
        if isinstance(alpha, (list, tuple)):
            assert len(alpha) == 3, "alpha must be a float or a list/tuple of length 3"
            self.alpha = [float(alpha[0]), float(alpha[1]), float(alpha[2])]
        else:
            self.alpha = [float(alpha)] * 3
        self.dropout = dropout
        self.excluded_modules = excluded_modules
        self.keys_to_lora = keys_to_lora

        if isinstance(self.seva_model, Seva):
            # When only training face attention LoRA, freeze everything else.
            if lora_for_face_attn_only:
                self.seva_model.requires_grad_(False)
                print("SevaLoRAWrapper: Froze all parameters in Seva model.")
            else:
                self.seva_model.freeze(["input", "middle", "output"])

        # This single call will recursively find and wrap all qualifying modules.
        self._apply_lora_to_model(
            self.seva_model,
            self_attn_rank,
            cross_attn_rank,
            ff_rank,
            self.alpha,
            dropout,
            self.keys_to_lora,
            self.excluded_modules
        )

        if freeze_lora:
            # freeze all parameters, even LoRA weights
            self.freeze_all()
            print("[SevaLoRAWrapper] LoRA is frozen.")

        if lora_for_face_attn_only:
            # unfreeze only the face attention blocks.
            for name, module in self.seva_model.named_modules():
                if 'attn_face' in name:
                    module.requires_grad_(True)
                    print(f"[SevaLoRAWrapper] Unfroze all parameters for {name}")

    def _apply_lora_to_model(self, module, self_attn_rank, cross_attn_rank, ff_rank, alphas, dropout, keys_to_lora, excluded_modules, prefix=""):
        """
        Applies LoRA to modules that look like transformer blocks. (Updated from complicated recursive version that would break state_dict)
        """
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name

            # First, recurse to the deepest children
            self._apply_lora_to_model(child, self_attn_rank, cross_attn_rank, ff_rank, alphas, dropout, keys_to_lora, excluded_modules, path)

        # After recursion, check if the current module should be wrapped.
        # This is a "duck-typing" approach.
        is_transformer_block = hasattr(module, "attn1") and hasattr(module, "attn2") and hasattr(module, "ff")
        
        if is_transformer_block:
            if skip_module_if_excluded(prefix, excluded_modules):
                return
                
            alpha_self, alpha_cross, alpha_ff = alphas
            
            if hasattr(module, "attn1") and not isinstance(module.attn1, LoRAAttentionWrapper):
                module.attn1 = LoRAAttentionWrapper(module.attn1, self_attn_rank, alpha_self, dropout, keys_to_lora)
            
            if hasattr(module, "attn2") and not isinstance(module.attn2, LoRAAttentionWrapper):
                module.attn2 = LoRAAttentionWrapper(module.attn2, cross_attn_rank, alpha_cross, dropout, keys_to_lora)
            
            # if hasattr(module, "attn_face") and not isinstance(module.attn_face, LoRAAttentionWrapper):
            #     module.attn_face = LoRAAttentionWrapper(module.attn_face, cross_attn_rank, alpha_cross, dropout, keys_to_lora)

            if hasattr(module, "ff") and isinstance(module.ff, nn.Module) and not isinstance(module.ff, LoRAFeedForwardWrapper):
                try:
                    module.ff = LoRAFeedForwardWrapper(module.ff, ff_rank, alpha_ff, dropout)
                except Exception: pass
            
            if hasattr(module, "ff_in") and isinstance(module.ff_in, nn.Module) and not isinstance(module.ff_in, LoRAFeedForwardWrapper):
                try:
                    module.ff_in = LoRAFeedForwardWrapper(module.ff_in, ff_rank, alpha_ff, dropout)
                except Exception: pass
                
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, dense_y: torch.Tensor, num_frames: Optional[int] = None, face_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.seva_model(x, t, y, dense_y, num_frames, face_context)

    def save_lora_weights(self, path: str):
        """Save only the LoRA weights."""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, (LoRAAttentionWrapper, LoRAFeedForwardWrapper)):
                for param_name, param in module.named_parameters():
                    if "lora_" in param_name:
                        lora_state_dict[f"{name}.{param_name}"] = param
        torch.save(lora_state_dict, path)

    def load_lora_weights(self, path: str):
        """Load only the LoRA weights."""
        lora_state_dict = torch.load(path)
        self.load_state_dict(lora_state_dict, strict=False) 

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        for module in self.modules():
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = True