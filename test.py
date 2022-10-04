''' Testing '''

import evaluation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_path = "./runs/runX/checkpoint/Position_relation/p=4_model_best.pth.tar"
#model_path = "./runs/runX/checkpoint/NTN_relation/p=4_model_best.pth.tar"
data_path = "./data/"
evaluation.evalrank(model_path, data_path=data_path, split="test")
