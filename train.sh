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
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train.py \
--compound_coef 5 \
--batch_size 4 \
--lr 1e-3 \
--load_weights models/efficientdet-d5.pth