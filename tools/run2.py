import os

cmd1 = "python train.py --resume 0 --trainer DRT_DG_Mixup --pe_type CI --root  D:/ML/Dataset  --output-dir ../output/PE_DomainMix_exp/ --config-file ../configs/trainers/drt_dg_r/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2.yaml  --source-domains art product real_world --target-domains clipart --seed 2022"
cmd2 = "python train.py --resume 0 --trainer DRT_DG_Mixup --pe_type CK --root  D:/ML/Dataset  --output-dir ../output/PE_DomainMix_exp/ --config-file ../configs/trainers/drt_dg_r/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v3_strong_transform_2.yaml  --source-domains art product real_world --target-domains clipart --seed 2022"
cmd3 = "python train.py --resume 0 --trainer DRT_DG_Mixup --pe_type CI --root  D:/ML/Dataset  --output-dir ../output/PE_DomainMix_exp/ --config-file ../configs/trainers/drt_dg_r/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v4_strong_transform_2.yaml  --source-domains art product real_world --target-domains clipart --seed 2022"
cmd4 = "python train.py --resume 0 --trainer DRT_DG_Mixup --pe_type CK --root  D:/ML/Dataset  --output-dir ../output/PE_DomainMix_exp/ --config-file ../configs/trainers/drt_dg_r/office_home_dg.yaml  --dataset-config-file  ../configs/datasets/drt_dg_r/office_home_dg_resnet50_draac_v4_strong_transform_2.yaml  --source-domains art product real_world --target-domains clipart --seed 2022"
# cmd5 = "python train.py --resume 0 --trainer DRT_DG_Mixup --pe_type CK --root  D:/ML/Dataset  --output-dir ../output/PE_DomainMix_exp/ --config-file ../configs/trainers/drt_dg_r/office_home_dg_odconv.yaml  --dataset-config-file  ../configs/datasets/drt_dg_r/office_home_dg_odconv4x_resnet50_strong_transform_2.yaml  --source-domains art product real_world --target-domains clipart --seed 2022"


os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)
# os.system(cmd5)
