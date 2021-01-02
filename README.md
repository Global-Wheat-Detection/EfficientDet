## Reproducing Processes

1. Install requirements
```
pip install -r requirements.txt
```

2. Download data from [here (1.68GB)](https://drive.google.com/drive/folders/1NZid0yEcbdecosLEpsItL17u2VLCEeet?usp=sharing), and uncompress the files into **datasets/**

(**wheat2coco.py** is our previous version of code which transforms original dataset into COCO-format, you might find it useful if you are working with COCO-format-friendly implementations)

3. Download pretrained weight [D7](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth), and place it in **models/**

4. Run script file and start to train
```
sh train.sh
```


## Data Description

Our data are collected from kaggle, including [Jigsaw](https://www.kaggle.com/alexanderliao/wheatfullsize), [SPIKE](https://www.kaggle.com/alexanderliao/wheatspike), and [original data](https://www.kaggle.com/c/global-wheat-detection/data). Note that the annotations have been tranformed into COCO-format. (with **construct_dataset.py**)