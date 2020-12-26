CUDA_VISIBLE_DEVICES=4 \
python3 eval.py \
--project global_wheat \
--compound_coef 5 \
--weights logs/global-wheat-detection/efficientdet-d5_23_16000.pth \
--split_ids logs/global-wheat-detection/split_ids.txt \
--force_resolution 896