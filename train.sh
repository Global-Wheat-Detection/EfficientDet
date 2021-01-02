# S1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
python3 train.py \
--compound_coef 7 \
--load_weights models/efficientdet-d7.pth \
--batch_size 10 \
--force_input_size 512 \
--aug_prob 0 \
--lr 3e-4 \
--num_epoch 20 \

# S2
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
python3 train.py \
--compound_coef 7 \
--load_weights last \
--batch_size 10 \
--force_input_size 512 \
--aug_prob 0.1 \
--lr 3e-4 \
--num_epoch 80 \ # 20 + 60

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
--num_epoch 100 \ # 20 + 60 + 20