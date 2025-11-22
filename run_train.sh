export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=lo
python main.py \
--base configs/example_training/seva-all-scratch-iclight.yaml \
--projectname seva-on-mvhsamples \
--no-test \
--resume_from_checkpoint /workspace/stonevol2/logs/old/old_only_faceweight/checkpoints/epoch\=000117.ckpt \
--no-strict-loading \
--override_ngpu 0,