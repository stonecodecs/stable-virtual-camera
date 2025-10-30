from seva.data.mvh_dataloader import MVHumanNetDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from sgm.models.diffusion import DiffusionEngine
from eval_utils import normalize_tensor
from eval_utils import show_tensor_batch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import torch 
import os

def expand_only_include(only_include):
    """
    Expands a string range into a list of zero-padded strings.
    Example input: "100001-102000,102020-104000"
    Example output: ['100001', '100002', ..., '102000', '102020', ..., '104000']
    """
    if isinstance(only_include, str): # in the format ex: "100001-102000,102020-104000"
        only_include = only_include.split(",")
        expanded_includes = []
        for subrange in only_include:
            start, end = [int(num) for num in subrange.split("-")]
            print(start, end)
            expanded_includes.extend([str(i).zfill(6) for i in range(start, end + 1)])
        return expanded_includes
    else:
        return only_include

def all_items_to_gpu(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to("cuda")
    return batch

def cfg_test(batch, engine, scales=[0.0, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0, 8.0]):
    for scale in scales:
        batch_copy = batch.copy()
        samples = run_step(batch_copy, engine, scale=scale)
        decoded_samples = normalize_tensor(engine.decode_first_stage(samples)).to("cpu")
        print(f"scale: {scale}")
        _ = show_tensor_batch(decoded_samples, masks=(batch_copy["ref_mask"].to("cpu")).squeeze(0))

def run_step(batch, engine, scale=1.0):
    batch = all_items_to_gpu(batch)
    engine = engine.to("cuda")
    samples = engine.infer(batch, scale=scale).squeeze(0)
    return samples

# Create dataset and dataloader
objects = expand_only_include("100010-100010")
dataset = MVHumanNetDataset(
    root_dir="/workspace/datasetvol/mvhuman_data/mv_captures",
    num_images=8,
    step_size=60,
    only_include=objects,
    preload_path="/workspace/datasetvol/mvhuman_data/mv_captures/preloaded_filepaths.json",
    iclight_dataset_path="/workspace/datasetvol/mvhuman_data/relit_images",
    infu_dataset_path=None,
    random_crop=False,
    maximal_crop=True,
    use_inconsistent=True,
)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# INPUT: logdir that contains the model checkpoint + config
# required inputs
log_path = "/workspace/stonevol2/logs/2025-09-20T22-34-57_example_training-gen-seva-mlp-lora"
# log_path = "/workspace/stonevol/logs/2025-08-27T12-30-30_example_training-seva-phase1"
checkpoint_name = "old_epoch=000190.ckpt"
# checkpoint_name = "last.ckpt"

# get config
filename_base = os.path.basename(log_path).split("_")[0]
run_config = OmegaConf.load(f"{log_path}/configs/{filename_base}-project.yaml")
ckpt_path = f"{log_path}/checkpoints/{checkpoint_name}"

model_config = run_config.model
# parameters to change (from config)
model_config.params.sampler_config.params.guider_config.params.cfg_min = 1.0

engine: DiffusionEngine = instantiate_from_config(run_config.model)
engine.init_from_ckpt(ckpt_path)
engine.eval()

os.makedirs("output_samples", exist_ok=True)

predicted_samples = []
target_images = []

for batch in loader:
    batch_ = batch.copy()
    samples = run_step(batch_, engine, scale=1.0)
    torch.save(batch['c2w'], "output_samples/c2w.pt")
    torch.save(batch['K'], "output_samples/K.pt")
    torch.save(batch['ref_mask'], "output_samples/ref_mask.pt")
    decoded_samples = normalize_tensor(engine.decode_first_stage(samples)).to("cpu")

    # Don't show images during evaluation
    # _ = show_tensor_batch(decoded_samples, masks=(batch["ref_mask"].to("cpu")).squeeze(0))
    # _ = show_tensor_batch(normalize_tensor(batch["frames"].squeeze(0)), masks=(batch["ref_mask"].to("cpu")).squeeze(0))
    # This is decoded images
    # decoded_samples
    # This is the ground truth images
    # batch["frames"]
    # normalize_tensor(batch["frames"].squeeze(0))
    # save_image_batch(batch, decoded_samples)
    ground_truth = normalize_tensor(batch["frames"].squeeze(0))
    predicted_samples.append(decoded_samples.detach())
    target_images.append(ground_truth.detach())

    # Deelte batch to free GPU memory
    del batch
    del batch_
    torch.cuda.empty_cache()

del loader
del dataset
torch.cuda.empty_cache()
predicted_samples = torch.cat(predicted_samples, dim=0)
target_images = torch.cat(target_images, dim=0)

# Compute pnsr, ssim, lpips here
pnsr_metric = PeakSignalNoiseRatio(data_range=1.0)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

pnsr_value = pnsr_metric(predicted_samples, target_images).item()
ssim_value = ssim_metric(predicted_samples, target_images).item()

print(f"PSNR: {pnsr_value}, SSIM: {ssim_value}")
