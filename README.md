# Introduction
This is the source code of the cross-modal information balance-aware reasoning network.
## Requirements and Installation
The following dependencies are recommended.

* Python==3.7.0
* pytorch==1.7.0
* torchvision==0.8.0
* torchaudio==0.7.0
* pytorch-pretrained-bert==0.6.2
  
## Pretrained model
If you don't want to train from scratch, you can download the pre-trained CIBRN model  from [here](https://drive.google.com/drive/folders/1eddbVAGbjHvofX96FuY4Sq7YcJTmuMhV?usp=drive_link)(for Flickr30K)
```bash
i2t: 488.0
Image to text: 73.8  93.4  96.6
Text to image: 54.7  81.3  88.2
t2i: 502.8
Image to text: 77.4  94.3  97.7
Text to image: 59.2  84.1  90.1
```
## Download Data 
We utilize the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). Some related text data can be found in the 'data' folder of the project (for Flickr30K).

## Training 
```bash
python train.py 
```
## Testing
```bash
python test.py
```
