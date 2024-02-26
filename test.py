import warnings
warnings.filterwarnings('ignore')
import time
import math
import os
import sys


def main(data):
    data_name = data.split('_')[0]

    data_file_path = './fold/'+data_name+'/'+data+'.csv'

    train = []
    test = []
    train_ori = []
    test_ori = []

    a = data_file_path.split('/')
    c = a[-1].split('.')

    for j in range(3): # fold 0,1,2
        b1 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_train.csv'
        b2 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_test_one.csv'
        b3 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_train_no.csv'
        b4 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_test.csv'
        
        train.append(b1)
        test.append(b2)
        train_ori.append(b3)
        test_ori.append(b4)

    path = './log/0226_1/' # log folder path
    mkdir = []
    
    a = data_file_path.split('/')
    c = a[-1].split('.')
    b = path + c[0]
    mkdir.append(b)
    os.makedirs(b, mode=0o775, exist_ok=True)
        
        
    save_txt = []
    for i in range(len(train)):
        a = train[i].split('/')
        b = a[-1].split('train')
        c = b[0][:-1] + '_train.txt'
        save_txt.append(c)

    start = time.time()
    for i in range(len(train)):
        x = math.floor(i/3)
        for j in range(1):
            print(f'{j},  {data_file_path} {train[i]} {test[i]} {train_ori[i]} {test_ori[i]} {mkdir[x]} {j+1}')
            print(f'{mkdir[x]}/{j+1}_{save_txt[i]}')
            os.system(f'python3 main.py {data_file_path} {train[i]} {test[i]} {train_ori[i]} {test_ori[i]} {mkdir[x]} {j+1} >> {mkdir[x]}/{j+1}_{save_txt[i]}')
    end = time.time()
    print('♨♨♨♨♨♨ time ♨♨♨♨♨♨ ',f"{end - start:.5f} sec")

if __name__ == "__main__":
    
    if len(sys.argv) > 0:
        main(sys.argv[1])