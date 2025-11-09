export NCCL_SOCKET_IFNAME=lo
python main.py \
--base configs/example_training/gen-seva-infu-identity.yaml \
--projectname seva-on-mvhsamples \
--no-test \
--resume /workspace/stonevol2/logs/old_only_faceweight/checkpoints/epoch\=000117.ckpt \
--no-strict-loading \
--override_ngpu 0,