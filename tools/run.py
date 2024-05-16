import os

# DG任务
# cmd1 = "python train.py --resume 0 --trainer DomainMix --root  D:/ML/Dataset  --output-dir ../output/DomainMix_exp/ --config-file ../configs/trainers/dg/ddg/OfficeHome.yaml  --dataset-config-file  ../configs/datasets/dg/ddg_with_mixstyle_OfficeHome_resnet50.yaml  --source-domains art product real_world --target-domains clipart"
cmd1 = "python train.py --resume 0 --trainer DAELDG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/daeldg/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd2 = "python train.py --resume 0 --trainer DAELDG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/daeldg/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd3 = "python train.py --resume 0 --trainer DAELDG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/daeldg/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd4 = "python train.py --resume 0 --trainer DAELDG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/daeldg/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"
cmd5 = "python train.py --resume 0 --trainer DAELDG --root  D:/ML/Dataset  --output-dir ../output/DAELDG_exp/ --config-file ../configs/trainers/dg/daeldg/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/dg/office_home_dg.yaml  --source-domains art product real_world --target-domains clipart"


os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)
os.system(cmd5)
