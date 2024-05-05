#python save_feats.py --method_name byol --ckpt_path ckpt/byol-cifar10-32brzx9a-ep=999.ckpt
#python save_feats.py --method_name simclr --ckpt_path ckpt/simclr-cifar10-b30xch14-ep=999.ckpt
#python save_feats.py --method_name mocov2 --ckpt_path ckpt/mocov2plus-cifar10-1nhrg2pm-ep=999.ckpt
#python save_feats.py --method_name swav --ckpt_path ckpt/swav-2rwotcpy-ep=999.ckpt
#python save_feats.py --method_name barlow --ckpt_path ckpt/barlow-cifar10-otu5cw89-ep=999.ckpt



python save_feats_cifar100_branchtuning.py --method_name byol-bt-task4 --ckpt_path ckpt/byol-bt-task4.ckpt
python save_feats_cifar100_branchtuning.py --method_name byol-bt-task0 --ckpt_path ckpt/byol-bt-task0.ckpt
python save_feats_cifar100_branchtuning.py --method_name byol-ft-task4 --ckpt_path ckpt/byol-ft-task4.ckpt

