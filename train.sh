# train from d6 pretrained weight provided by zylo117
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 train.py \
# --compound_coef 6 \
# --batch_size 4 \
# --load_weights models/efficientdet-d6.pth \
# --debug True

# Disable visualizations to train faster
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 train.py \
# --compound_coef 6 \
# --batch_size 4 \
# --load_weights last \

# train from d5 pretrained weight provided by zylo117
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 train.py \
# --compound_coef 5 \
# --batch_size 4 \
# --lr 1e-3 \
# --load_weights models/efficientdet-d5.pth

# train from d5 pretrained weight provided by zylo117
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 train.py \
# --compound_coef 5 \
# --batch_size 4 \
# --lr 1e-4 \
# --load_weights last

# S1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
# python3 train.py \
# --compound_coef 7 \
# --load_weights models/efficientdet-d7.pth \
# --batch_size 10 \
# --force_input_size 512 \
# --aug_prob 0 \
# --lr 5e-4 \
# --num_epoch 10

# S2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
# python3 train.py \
# --compound_coef 7 \
# --load_weights logs/global-wheat-detection/efficientdet-d7_10_2959.pth \
# --batch_size 10 \
# --force_input_size 512 \
# --aug_prob 0.5 \
# --lr 3e-4 \
# --num_epoch 50

# S1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
# python3 train.py \
# --compound_coef 7 \
# --load_weights models/efficientdet-d7.pth \
# --batch_size 10 \
# --force_input_size 512 \
# --aug_prob 0 \
# --lr 3e-4 \
# --num_epoch 20 \

# S2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
# python3 train.py \
# --compound_coef 7 \
# --load_weights last \
# --batch_size 10 \
# --force_input_size 512 \
# --aug_prob 0.1 \
# --lr 3e-4 \
# --num_epoch 80 \

# S3
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
python3 train.py \
--train_split 1 \
--compound_coef 7 \
--load_weights last \
--batch_size 10 \
--force_input_size 512 \
--aug_prob 0.1 \
--lr 3e-4 \
--num_epoch 100 \