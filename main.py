import module as mod
import pandas as pd
import numpy as np
import heaan_sdk as heaan
import random
import os
import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import math

logN=16
os.environ["OMP_NUM_THREADS"] = "32"  # set the number of CPU threads to use for parallel regions : 32
# set key_dir_path
key_file_path = Path('./keys_FGb')
# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
context = heaan.Context(
    params,
    key_dir_path=key_file_path,
    load_keys="all",
    generate_keys=False,
)

num_slot = context.num_slots # 32768
log_num_slot = context.log_slots

a = sys.argv[2]
a = a.split('/')[-1].split('.')[0]

df = pd.read_csv(sys.argv[1])
train = pd.read_csv(sys.argv[2])
test = pd.read_csv(sys.argv[3])
train_ori = pd.read_csv(sys.argv[4])
test_ori = pd.read_csv(sys.argv[5])

qqq = int(sys.argv[7])

json_path = './JSON'
mod.save_metadata_json_eval(df, train, json_path)


with open(json_path + 'Metadata.json') as f:
    meta = json.load(f)

# 확인하기
for i in ['ndata', 'train_ndata', 'test_ndata', 'n', 'd', 't']:
    print(f"{i} : ", meta[i])

n = meta['n']
d = meta['d']
t = meta['t']
train_ndata = meta['train_ndata'] 
test_ndata = meta['test_ndata']

attribute_value_pair = []
for i in range(d):
    for j in range(n):
        attribute_value_pair.append('X'+ str(i+1) + '_' + str(j+1))


model_path = sys.argv[6] + '/' 

print(" *** *** *** Training *** *** ***")

start = time.time()
if train_ndata < num_slot:
    mod.Rule_generation(model_path, train, train_ndata, n,d,t,logN,context,a)
else:
    mod.Rule_generation_mult(model_path, train, train_ndata, n,d,t,logN,context,a)
end = time.time()
print()
print('!!!!!!! Rule_generation time !!!!!!! ',f"{end - start:.8f} sec")
print()

print()
print(" *** *** *** inference *** *** ***")
start = time.time()
cy_hat_list = mod.inference(test, model_path,n,d,t,logN,context,a)
end = time.time()
print()
print('!!!!!!! inference time !!!!!!! ',f"{end - start:.8f} sec")
print()
test_res = mod.accurate(test_ori,cy_hat_list)
###
print('test accuracy: ', round(test_res,8))



