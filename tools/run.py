import os

# DG任务
# cmd1 = "python train.py --resume 0 --trainer DomainMix --root  D:/ML/Dataset  --output-dir ../output/DomainMix_exp/ --config-file ../configs/trainers/dg/ddg/OfficeHome.yaml  --dataset-config-file  ../configs/datasets/dg/ddg_with_mixstyle_OfficeHome_resnet50.yaml  --source-domains art product real_world --target-domains clipart"
cmd1 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains clipart product real_world --target-domains art"
cmd2 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains clipart product real_world --target-domains art"
cmd3 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains clipart product real_world --target-domains art"
cmd4 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains clipart product real_world --target-domains art"
cmd5 = "python train.py --resume 0 --trainer DDAIG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/ddaig/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains clipart product real_world --target-domains art"


os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)
os.system(cmd5)
