## Reproducing Processes

1. Install requirements
```
pip install -r requirements.txt
```
2. Download data from [Kaggle](https://www.kaggle.com/c/global-wheat-detection/data), and uncompress the files into **datasets/global-wheat-detection/**
3. Run **./wheat2coco.py** to get COCO-format annotations
```
python3 wheat2coco.py
```
4. Download pretrained weight [D5](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth), and place it in **models/**
5. Run script file and start to train
```
sh train.sh
```