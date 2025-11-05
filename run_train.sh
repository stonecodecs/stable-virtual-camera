export NCCL_SOCKET_IFNAME=lo
python main.py \
--base configs/example_training/gen-seva-infu-identity.yaml \
--projectname seva-on-mvhsamples \
--no-test \
--override_ngpu 0,