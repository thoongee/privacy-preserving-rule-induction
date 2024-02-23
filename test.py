import warnings
warnings.filterwarnings('ignore')
import time
import math
import os



dataset = [
           './fold/iris/iris_doane.csv'
        #    './fold/iris/iris_fd.csv',
        #    './fold/iris/iris_scott.csv',
        #    './fold/iris/iris_sturges.csv',
           
        #    './fold/wine/wine_doane.csv',
        #    './fold/wine/wine_fd.csv',
        #    './fold/wine/wine_scott.csv',
        #    './fold/wine/wine_sturges.csv',
           
        #    './fold/cancer/cancer_doane.csv',
        #    './fold/cancer/cancer_fd.csv',
        #    './fold/cancer/cancer_scott.csv',
        #    './fold/cancer/cancer_sturges.csv',
           
        #    './fold/BreastCancer/BreastCancer.csv',
        #    './fold/Soybean/Soybean.csv'
           ]


train = []
test = []
train_ori = []
test_ori = []
for i in range(len(dataset)):
    a = dataset[i].split('/')
    c = a[-1].split('.')
    # for j in range(3):
    for j in range(1):
        b1 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_train.csv'
        b2 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_test_one.csv'
        b3 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_train_no.csv'
        b4 = './fold/' + a[2] + '/' + c[0] + f'_f{j}_test.csv'
        
        train.append(b1)
        test.append(b2)
        train_ori.append(b3)
        test_ori.append(b4)


path = './test/0222_7/' # log file path
mkdir = []
for i in range(len(dataset)):
    a = dataset[i].split('/')
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
        print(f'{j},  {dataset[x]} {train[i]} {test[i]} {train_ori[i]} {test_ori[i]} {mkdir[x]} {j+1}')
        print(f'{mkdir[x]}/{j+1}_{save_txt[i]}')
        os.system(f'python3 main.py {dataset[x]} {train[i]} {test[i]} {train_ori[i]} {test_ori[i]} {mkdir[x]} {j+1} >> {mkdir[x]}/{j+1}_{save_txt[i]}')
end = time.time()
print('♨♨♨♨♨♨ time ♨♨♨♨♨♨ ',f"{end - start:.5f} sec")

