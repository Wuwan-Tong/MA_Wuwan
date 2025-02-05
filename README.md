### This is the code of my Master Thesis: System Design and Performance Evaluation for Semantic Communication with Large Language Model  
#### Environments  
You can install the environment by requirements.txt  
We use CUDA 12.1, pytorch 2.3.0, and python 3.8  

#### datasets
You can download MS COCO at https://cocodataset.org/#download, we use coco2014 version, and download Flickr30k at https://rwth-aachen.sciebo.de/s/Gq2RdccXQysBuUN  

#### Checkpoints  
checkpoints of fine-tuning can be downloaded from:  
MS COCO: https://rwth-aachen.sciebo.de/s/eACUUDOzBFtVm74  
Flickr30k: https://rwth-aachen.sciebo.de/s/2OHVBKvgB1WXpox  
you can put the checkpoints at root/coco_ckpt and root/flickr_ckpt  
#### Run  
you can run the code at the files in root/run and switch datasets by importing various dataclasses in root/configs/data  
finetuning is mainly based on the code from OpenCLIP (root/training), you can run root/training/ft_main to start finetuning  


 
