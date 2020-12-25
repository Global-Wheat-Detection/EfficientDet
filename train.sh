# train from d6 pretrained weight provided by zylo117
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train.py \
--compound_coef 6 \
--batch_size 4 \
--lr 1e-4 \
--optim adamw \
--num_epochs 500 \
--load_weights models/efficientdet-d6.pth \
--debug True
