import os

# DG任务
cmd1 = "python tools/train.py " \
       "--root D:\ML\Dataset " \
       "-- output-dir ../output/pacs_exp" \
       "--trainer DDAIG " \
       "--config-file ./configs/trainers/dg/ddaig/pacs.yaml  " \
       "--dataset-config-file  ./configs/datasets/dg/pacs.yaml  " \
       "--source-domains art_painting cartoon photo " \
       "--target-domains sketch"

os.system(cmd1)
