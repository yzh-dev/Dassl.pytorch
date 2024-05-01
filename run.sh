#Below we provide an example for training a source-only baseline on the popular domain adaptation dataset, Office-31,

#CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#--root "D:\ML\Dataset" \
#--trainer SourceOnly \
#--source-domains amazon \
#--target-domains webcam \
#--dataset-config-file configs/datasets/da/office31.yaml \
#--config-file configs/trainers/da/source_only/office31.yaml \
#--output-dir output/source_only_office31

