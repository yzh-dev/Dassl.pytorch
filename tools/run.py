import os

# DG任务
cmd1 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd2 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd3 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd4 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd5 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"


os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)
os.system(cmd5)
