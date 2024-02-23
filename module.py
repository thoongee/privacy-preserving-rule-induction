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
global num_slot
num_slot = context.num_slots # 32768
log_num_slot = context.log_slots

# ============================================
# ============== Basic Operation ==============
# ============================================

def print_ctxt(c,size):
    m = c.decrypt(inplace=False)
    for i in range(size):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
    m = None

def print_ctxt_1(c,size):
    m = c.decrypt(inplace=False)
    for i in range(size):
        if np.abs(m[i].real) < 0.001:
            pass
        else:
            print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
    m = None

def check_boot(x):
    if x.level==3:
        print('####bootstrapping####')
        x.bootstrap()
    return x

def mult(a,b):
    c = a*b
    check_boot(c)
    return c

def save_metadata_json(df1, df2, json_path):
    # df1 = pd.read_csv(csv1)
    # df2 = pd.read_csv(csv2)
    # meta_data함수로 ndata,n,d,t 파악하기
    ndata, d, n, t = meta_data(df1, df2)
    pow2_n = get_smallest_pow_2(n)
    N = get_smallest_pow_2(ndata)
    d_in_ctxt = num_slot//(N*pow2_n)
    if d_in_ctxt >= d:
        d_in_ctxt = d
    else:
        pass
        
    if d_in_ctxt == 0:
        numc = "cannot put in one ciphertext!"
        
    else:
        numc = int(np.ceil(d/d_in_ctxt))
    Metadata = {'ndata':ndata,
                'n':n,
                'd':d,
                't':t,
                # 'depth':depth,
                'd_in_ctxt':d_in_ctxt,
                'N':N,
                'pow2_n':pow2_n,
                'numc':numc
                }
    
    # Metadata라는 이름으로 json 파일 저장(경로 확인!)
    with open(json_path + "Metadata.json", "w") as json_file:
        json.dump(Metadata, json_file, indent=2)

def get_smallest_pow_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def left_rotate_reduce(context,data,gs,interval):
    # print('--------left_rotate_reduce 입력암호문 레벨------ ;',data.level)
    m0 = heaan.Block(context,encrypted = False, data = [0]*context.num_slots)
    res = m0.encrypt()
    
    empty_msg= heaan.Block(context,encrypted = False)
    rot = empty_msg.encrypt(inplace=False)
    
    binary_list = []
    while gs > 1:
        if gs%2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs//2
    binary_list.append(gs)

    # print("1")
    # print_ctxt1(data,context.num_slots)
    i = len(binary_list)-1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data
            # print("0")
            # print_ctxt1(tmp,context.num_slots)
            while ind < i:
                
                rot = tmp.__lshift__(s)
                # print("1")
                # print_ctxt1(rot,context.num_slots)
                # check_boot()
                tmp = tmp + rot
                # print("2")
                # print_ctxt1(tmp,context.num_slots)
                s = s*2
                ind = ind+1
            if sdind > 0:
                tmp = tmp.__lshift__(sdind)
            # print("3")
            # print_ctxt1(tmp,context.num_slots)
            res = res + tmp
            # print("4")
            # print_ctxt1(res,context.num_slots)
            sdind = sdind + s
        i = i - 1            

    del  rot, tmp
    return res

def right_rotate_reduce(context, data, gs, interval):

    m0 = heaan.Block(context, encrypted=False, data=[0]*context.num_slots)
    res = m0.encrypt()
    
    empty_msg = heaan.Block(context, encrypted=False)
    rot = empty_msg.encrypt(inplace=False)
    
    binary_list = []
    while gs > 1:
        if gs % 2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs // 2
    binary_list.append(gs)

    i = len(binary_list) - 1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data

            while ind < i:
                
                rot = tmp.__rshift__(s)  # 변경된 부분: 왼쪽 시프트 대신 오른쪽 시프트 연산 사용

                tmp = tmp + rot
 
                s = s*2
                ind = ind + 1
            if sdind > 0:
                tmp = tmp.__rshift__(sdind)  # 변경된 부분: 왼쪽 시프트 대신 오른쪽 시프트 연산 사용

            res = res + tmp

            sdind = sdind + s
        i = i - 1            

    del rot, tmp
    
    return res

# ============================================
# ============== Meta Data ==============
# ============================================

def find_max_cat_X(df):
    col = df.columns
    max_values = []
    for i in col.drop('label'):
        max_values.append(max(df[i]))
    max_x = int(max(max_values))
    return max_x

def save_metadata_json_eval(df1,df2, json_path):
    # meta_data함수로 ndata,n,d,t 파악하기
    ndata, d, n, t, train_ndata, test_ndata = meta_data_eval(df1, df2)
    Metadata = {'ndata':ndata,
                'n':n,
                'd':d,
                't':t,
                'train_ndata': train_ndata,
                'test_ndata' : test_ndata
                }
    
    # Metadata라는 이름으로 json 파일 저장(경로 확인!)
    with open(json_path + "Metadata.json", "w") as json_file:
        json.dump(Metadata, json_file, indent=2)

def meta_data_eval(df1, df2):
    col = df1.columns

    for cname in col:
        if cname != 'label':  # Assuming 'label' should not be converted to categorical in this way
            # Convert to category and then map each category to a new code
            df1[cname] = pd.Categorical(df1[cname])
            df1[cname] = df1[cname].cat.codes + 1  # Shift codes to start from 1

    ndata = df1.shape[0]
    train_ndata = df2.shape[0]
    test_ndata = ndata - train_ndata
    d = len(col) - 1
    n = find_max_cat_X(df1)
    t = len(df1['label'].unique())

    return ndata, d, n, t, train_ndata, test_ndata

# ============================================
# ================ Training ==================
# ============================================
def Rule_generation(model_path, train, train_ndata, n,d,t,logN,context,qqq):
    
    # if (data instance < 32768) && (n × d < 32768)
    print('Training Start!')
    
    global attribute_value_pair
    attribute_value_pair = []
    for i in range(d):
        for j in range(n):
            attribute_value_pair.append('X'+ str(i+1) + '_' + str(j+1))
    

    start = time.time()
    train_ctxt, label_ctxt = input_training(train,train_ndata, n,d,t,logN,context)
    end = time.time()

    print('!!!!!!! input_training time !!!!!!! ',f"{end - start:.8f} sec")
    print()
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*context.num_slots)
    Rule = m0.encrypt(inplace=False)
    
    i=0
    # for i in range((n*d)):
    for i in range((n*d)-round(n*d*1/3)):
    # for i in range(2):
        
        print('feature > >> > ',i)
        print()
        start = time.time()
        frequency, fre_label = measure_frequency(train_ndata, train_ctxt, label_ctxt, n,d,t,logN,context) # ok
        end = time.time()
        print('========measure_frequency result ========')
        # print('==frequency ctxt==')
        # print_ctxt(frequency,n*d)
        print('==fre label ctxt==')
        print_ctxt(fre_label,n*d)
        print()
        print('≫≫≫≫≫≫ measure_frequency time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        g, g_list = calculate_Gini(train_ndata, frequency, fre_label, n,d,t,logN,context)
        #g : gini 값 가장 작은 slot만 1 나머지 0
        #g_list : g_list[0] = g의 0번째 슬롯 값을 전체에 복사해 둔 암호문
         
        end = time.time()
        print('========calculate_Gini result ========')
        print_ctxt(g,n*d)
        print()
        print('≫≫≫≫≫≫ calculate_Gini time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        # for i in range(n*d):
            # print(i)
            # print_ctxt_1(g_list[i],dec,sk,logN,n*d)
        print()
        
        start = time.time()
        y_cy, cy, c_cy, c_sum = find_label(train_ndata, g_list, train_ctxt, label_ctxt,n,d,t,logN,context)
        end = time.time()
        print('========find_label result ========')
        print('==y_cy==')
        print_ctxt(y_cy,n*d)
        print('==cy==')
        print_ctxt(cy,t+3)
        print('==c_cy==')
        print_ctxt(c_cy,n*d)
        print('==c_sum==')
        print_ctxt(c_sum,n*d)

        print()
        print('≫≫≫≫≫≫ find_label time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        start = time.time()
        ca = isReal_1(y_cy, cy,context)
        end = time.time()
        print('========isreal result ========')
        print_ctxt(ca,n*d)
        print()
        print('≫≫≫≫≫≫ isReal time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        one_rule = create_rule(g, c_cy,n,d,logN,context)
        end = time.time()
        print('========create_rule result ========')
        print_ctxt(one_rule,n*d)
        print()
        print('≫≫≫≫≫≫ create_rule time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        train_ctxt, label_ctxt = data_update_5(g_list, c_sum, train_ctxt,label_ctxt,n,d,t,logN,context)
        end = time.time()
        # print('========data_update result ========')
        # print('==train ctxt==')
        # print_ctxt(train_ctxt,n*d)
        # print('==label ctxt==')
        # print_ctxt(label_ctxt,n*d)
        print()
        print('≫≫≫≫≫≫ data_update time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        encoding_rule = change_rule(g, g_list, cy, n,d,t,logN,context)
        end = time.time()
        print('========change_rule result ========')
        print_ctxt(encoding_rule,n*d)
        print()
        print('≫≫≫≫≫≫ change_rule time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        # rule 가공 해야 함
        # ca(by)가 1 이면 룰은 살아남고 0이면 없에야 됨
        # 단순 곱
        start = time.time()
        # mult(encoding_rule, ca, one_rule, eval)
        # eval.add(one_rule, Rule, Rule)
        one_rule  = encoding_rule * ca
        check_boot(one_rule)
        Rule = Rule + one_rule
        end = time.time()
        print('=====rule ! ! !======')
        print_ctxt_1(one_rule,n*d)
        # print()
        print('≫≫≫≫≫≫ rule_add time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
    
    start = time.time()
    Rule.save(model_path + f'{qqq}_Rule.ctxt')
    end = time.time()
    print()
    print('total rule ! ! !')
    print_ctxt_1(Rule,n*d)
    print()
    print()
    print('≫≫≫≫≫≫ rule_save time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
    print()
    print()
    print()

#1
def input_training(train,train_ndata, n,d,t,logN,context):
    print(' --- input_training --- ')
    # training data encryption
    # 암호문은 각 instance 별이 아닌 feature별로 암호화
    
    # if (data instance < 32768) && (n × d < 32768)
    
    # X1_1 X1_2 X2_1 X2_2 X3_1 X3_2 label_1 label_2
    #   1    0    1    0    0    1      0       1
    #   0    1    0    1    0    1      1       0
    #   1    0    0    1    1    0      1       0
    #   0    1    1    0    1    0      1       0
    #   1    0    0    1    0    1      0       1
    
    # X1_1 : 1 0 1 0 1
    # X1_2 : 0 1 0 1 0

    train_ctxt = []
    for i in range(n*d):
        x = train[attribute_value_pair[i]].values.tolist() + [0]*(num_slot-train_ndata)
        # len(x)
        mess = heaan.Block(context, data = x, encrypted=False)
        x_tmp = mess.encrypt(inplace=False)
        train_ctxt.append(x_tmp)
        
    label_ctxt = []
    for i in range(t):
        x = train['label_' + str(i+1)].values.tolist() + [0]*(num_slot-train_ndata)
        # len(x)
        mess = heaan.Block(context, data = x, encrypted=False)
        x_tmp = mess.encrypt(inplace=False)
        train_ctxt.append(x_tmp)
        
        if t > 3:
            x_tmp = x_tmp * (1/t)
            check_boot(x_tmp)
        
        label_ctxt.append(x_tmp)
    
    return train_ctxt, label_ctxt
#2
def measure_frequency(train_ndata, train_ctxt, label_ctxt, n,d,t,logN,context):
    print(' --- measure_frequency --- ')
    real_time = 0
    # 지니 지수 계산을 위한 빈도수 계산
    
    ''' 나름의 상세 설명
    # Yes인 암호문 하나
    # No인 암호문 하나
    # Total 암호문 하나
    # 총 암호문 세 개로 사용

    # X1_1 X1_2 X2_1 X2_2 X3_1 X3_2 label_1 label_2
    #   1    0    1    0    0    1      0       1
    #   0    1    0    1    0    1      1       0
    #   1    0    0    1    1    0      1       0
    #   0    1    1    0    1    0      1       0
    #   1    0    0    1    0    1      0       1


    # X1_1 : 1 0 1 0 1
    # X1_2 : 0 1 0 1 0
    # label_1 : 0 1 1 1 0
    # label_2 : 1 0 0 0 1

    # Yes 암호문
        # X1_1 10101 × 01110 = 00100 ⇒ 1
        # X1_2 01010 × 01110 = 01010 ⇒ 2
    # No 암호문
        # X1_1 10101 × 10001 = 10001 ⇒ 2
        # X1_2 01010 × 10001 = 00000 ⇒ 0
    # Total 암호문
        # X1_1 10101 ⇒ sum(10101) ⇒ 3
        # X1_1 01010 ⇒ sum(01010) ⇒ 2

    # One-hot encoding 된 데이터에서 한 column당 세 개의 암호문 생성
    # 예시 데이터에서는 18개 암호문 생성
        # ⇒ n × d × 3
    
    # Yes 암호문
    # 1 2 1 2 2 1

    # No 암호문
    # 2 0 1 1 0 2

    # Total 암호문
    # 3 2 2 3 2 3

    # 암호문 하나에 밀어 넣고 3개의 암호문 생성'''
    # X1_1 X1_2 X2_1 X2_2 X3_1 X3_2 label_1 label_2
    #   1    0    1    0    0    1      0       1
    #   0    1    0    1    0    1      1       0
    #   1    0    0    1    1    0      1       0
    #   0    1    1    0    1    0      1       0
    #   1    0    0    1    0    1      0       1

    # label_1과의 frequency 먼저 계산
    # X1_1 × label_1 = 0 0 1 0 0
    # left_rotate_reduce ⇒ 1
    # right_rotate(j → j=(1,...,n×d))
    # label_1에 대해 한번 iteration 하면 한 암호문 나옴
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*context.num_slots)
    # m0 = heaan.Message(logN-1,0)
    
    # fre_tmp = heaan.Ciphertext(context)
    # fre_tmp.to_device()

    # m100 = heaan.Message(logN-1,0)
    # m100_ = [1] + [0]*(num_slot-1)
    # m100.set_data(m100_)
    # m100.to_device()
    m100 = heaan.Block(context,encrypted = False, data = [1] + [0]*(context.num_slots-1))

    frequency = []
    
    etc_time = 0
    start = time.time()
    for i in range(t):
        start_1 = time.time()
        # fre_label = heaan.Ciphertext(context)
        # enc.encrypt(m0, pk, fre_label)
        # fre_label.to_device()
        fre_label = m0.encrypt(inplace=False)
        end_1 = time.time()
        etc_time += end_1 - start_1
        for j in range(n*d):
            # X1_1 × label_1 = 0 0 1 0 0
            # mult(label_ctxt[i], train_ctxt[j], fre_tmp, eval)
            # mod.print_ctxt(fre_tmp,dec,sk,logN,5)
            fre_tmp = label_ctxt[i] * train_ctxt[j]
            check_boot(fre_tmp)
            
            # left_rotate_reduce ⇒ 1
            # eval.left_rotate_reduce(fre_tmp, 1, train_ndata, fre_tmp)
            fre_tmp = left_rotate_reduce(context,fre_tmp,train_ndata,1)
            check_boot(fre_tmp)
            # 마스킹
            fre_tmp = fre_tmp * m100
            check_boot(fre_tmp)
            # mult(fre_tmp, m100, fre_tmp, eval)
            
            # right_rotate(j → j=(0,...,n×d-1))
            # right_rotate(fre_tmp, j, fre_tmp, eval)
            fre_tmp = fre_tmp.__rshift__(j)
            # mod.print_ctxt(fre_tmp,dec,sk,logN,5)
            
            # eval.add(fre_label, fre_tmp, fre_label)
            fre_label = fre_label + fre_tmp
            # label_1에 대해 한번 iteration 하면 한 암호문 나옴
        frequency.append(fre_label)
    end = time.time()
    real_time += (end-start) - etc_time

    # total frequency 세야 함
    # frequency 더하면 됨
    ## 진아 ; 위에 fre_label이라는 변수를 썼는데 여기서 또 똑같은 이름으로 선언해서 쓴다고? total_label로 이름 바꿔야지..
    
    # fre_label = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, fre_label)
    # fre_label.to_device()
    total_fre = m0.encrypt(inplace=False)
    
    start = time.time()
    for i in range(t):    
        # eval.add(frequency[i], fre_label, fre_label)
        total_fre = total_fre + frequency[i]
    end = time.time()
    real_time += end-start
    print('measure_frequency real time ',f"{real_time:.8f} sec")
    # print('*** label ')
    # print_ctxt(fre_label,dec,sk,logN,n*d)
    # print()
    
    return frequency, total_fre
#3
def calculate_Gini(train_ndata, frequency, fre_label, n,d,t,logN,context):
    print(' --- calculate_Gini --- ')
    real_time = 0 
    m100_ = [1] + [0]*(num_slot-1)
    m100 = heaan.Block(context,encrypted = False, data = m100_)
    
    # 측정한 빈도수로 지니 지수 계산
    # -------------------------------------
    # 지니지수 계산해야 함
    # 나눗셈 연산은 동형암호에 적합하지 않음
    # 본래의 지니지수
    #   G(S|A)=1-1/|S|∑_t∑_c|S_t,c|^2/|S_t|^2
    # Modified Gini
    #   MG(S|A)=1/|S|^2∑_t(|S_t|^2-∑_c|S_t,c|^2)

    # 직접 계산 해보니 (total)^2-(label_1^2 + label_2^2)
    # 위 식 계산해 최소값 찾으면 됨

    # (total)^2-(label_1^2 + label_2^2 + ...)

    # 계산된 gini에서 제일 작은걸 찾아야 함

    # ººº ººº ººº ººº ººº ººº ººº ººº ººº 
    # total과 label에 train_ndata 나눠줌
    # ººº ººº ººº ººº ººº ººº ººº ººº ººº 
    
    # m0 = heaan.Message(logN-1,0)
    m0 = heaan.Block(context,encrypted = False, data = [0]*context.num_slots)
    
    # tmp = heaan.Ciphertext(context)
    # tmp.to_device()

    # square_total = heaan.Ciphertext(context)
    # square_total.to_device()
    start = time.time()
    tmp = fre_label * (1/train_ndata)
    check_boot(tmp)
    # mult(fre_label, 1/train_ndata, tmp, eval)
    square_total = tmp * tmp
    check_boot(square_total)
    # mult(tmp, tmp, square_total, eval)
    end = time.time()
    real_time += end-start
    # print_ctxt(tmp,dec,sk,logN,5)
    # print_ctxt(fre_label,dec,sk,logN,5)
    # print_ctxt(square_total,dec,sk,logN,5)
    # print_ctxt(frequency[0],dec,sk,logN,5)

    # gini = heaan.Ciphertext(context)
    # gini.to_device()

    # square_label = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, square_label)
    # square_label.to_device()
    square_label = m0.encrypt(inplace=False)
    start = time.time()
    for i in range(t):
        # mult(frequency[i], 1/train_ndata, tmp, eval)
        tmp = frequency[i] * (1/train_ndata)
        check_boot(tmp)
        # mult(tmp, tmp, gini, eval)
        gini = tmp*tmp
        check_boot(gini)
        # eval.add(gini, square_label, square_label)
        square_label = square_label + gini
    # print_ctxt(square_label,dec,sk,logN,5)

    # eval.sub(square_total, square_label, gini)
    gini = square_total - square_label
    # print('### gini')
    # print_ctxt(gini,dec,sk,logN,n*d)
    # print()

    # min_gini = heaan.Ciphertext(gini)
    # min_gini.to_device()
    min_gini = gini
    
    g = findMinPos(min_gini, context,logN,d,n,n*d)
    end = time.time()
    real_time += end-start
    
    # 이게 n × d 안에서는 1이 한 개지만 범위 넘어가면 1 여러개 나옴
    # 쓰레기 값 처리
    # n_d_mask_ = [1]*n*d + [0]*(num_slot-n*d)
    # n_d_mask = heaan.Message(logN-1,0)
    # n_d_mask.set_data(n_d_mask_)
    # n_d_mask.to_device()
    
    n_d_mask = heaan.Block(context,encrypted = False, data = [1]*n*d + [0]*(num_slot-n*d))

    start = time.time()
    # mult(g, n_d_mask, g, eval)
    g = g * n_d_mask
    check_boot(g)
    end = time.time()
    real_time += end-start
    
    # print(' gini level : ',g.level)
    start = time.time()
    # eval.bootstrap(g,g)
    end = time.time()
    real_time += end-start
    
    g_list = []
    etc_time = 0
    start = time.time()
    for i in range(n*d):
        start_1 = time.time()
        pick_ = [0]*i + [1] + [0]*(num_slot-(i+1))
        # pick = heaan.Message(logN-1,0)
        # pick.set_data(pick_)
        # pick.to_device()
        pick = heaan.Block(context,encrypted = False, data = pick_)
        
        # tmp = heaan.Ciphertext(context)
        # tmp.to_device()
        ## 진아 ; tmp를 항상 다시 생성해야하나?
        end_1 = time.time()
        etc_time += end_1 - start_1
        tmp = g * pick
        check_boot(tmp)
        # mult(g, pick, tmp, eval)
        # eval.left_rotate_reduce(tmp, 1, num_slot, tmp)
        tmp = left_rotate_reduce(context,tmp,num_slot,1)
        
        # tmp = tmp * m100 ## 진아 추가
        g_list.append(tmp)
    end = time.time()
    real_time += (end-start)-etc_time
    print('calculate_Gini real time ',f"{real_time:.8f} sec")
    return g, g_list

# in 3
def findMinPos(c, context,logN,d,n,n_comp):

    print(' ======findMinPos 입력암호문==== ======')
    print_ctxt(c,n*d)
    cmin = findMin4(c,context,logN,d,n,n_comp)
    print(' ====== findMin4 결과암호문 ======')
    print_ctxt(cmin,n*d)
    # os.system('nvidia-smi -q -d memory') 
    # print(' ====== ====== ======')
    # check_boot(cmin, he)
    # print("findMinPos cmin:",cmin.level)

    # m100 = heaan.Message(logN-1,0)
    # m100_ = [1] + [0]*(num_slot-1)
    # m100.set_data(m100_)
    # m100.to_device()
    m100 = heaan.Block(context,encrypted = False, data = [1] + [0]*(context.num_slots-1))
    # mult(cmin, m100, cmin, he)
    cmin = cmin * m100
    check_boot(cmin)
    
    # he.right_rotate_reduce(cmin,1,num_slot,cmin)
    cmin = right_rotate_reduce(context,cmin,num_slot,1)
    
    # he.sub(c,cmin,c)
    c = c - cmin

    # he.sub(c,1/2**15,c)
    c = c - 1/2**15

    # ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    # ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    
    # c_red = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, c_red)
    # heaan.math.approx.sign(he, c, c_red)
    c_red = c.sign(inplace = False, log_range=0)
    check_boot(c_red)
    # he.negate(c_red, c_red)
    c_red = c_red.__neg__()
    check_boot(c_red)
    # he.add(c_red, 1, c_red)
    c_red = c_red + 1
    # mult(c_red, 0.5, c_red, he)
    c_red = c_red * 0.5
    check_boot(c_red)

    # md1_ = [1]*n_comp + [0]*(context.num_slots-n_comp)
    # md1 = heaan.Message(logN-1)
    # md1.set_data(md1_)
    # md1.to_device()
    # md1 = heaan.Block(context,encrypted = False, data = md1_)

    ### Need to generate a rotate sequence to choose 1 in a random position   
    # c_out = heaan.Ciphertext(context)
    # heaan.math.approx.discrete_equal_zero(he, c_red, c_out)
    c_out = selectRandomOnePos(c_red,context,n_comp) # 같은 min값 잇으면 하나만 랜덤으로 고르기
    print('==========selectRandomOnePos 결과암호문 ===================')
    print_ctxt(c_out,d*n)

    # cmin = None
    # cmin.to_host()
    # del cmin, c_red
    # print(' ====== ====== ======')
    # os.system('nvidia-smi -q -d memory') 
    # print(' ====== ====== ======')
    return c_out

# in 3
def findMin4(c, context, logN, d, n, n_comp):
    print('n comp : ',n_comp)
    
    if (n_comp==1): 
        print("findMin4 ends..")
        print_ctxt(c,1)
        return c
    check_boot(c)
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    ##print("cinn level:",c.level)

    ## divide ctxt by four chunks
    if (n_comp % 4 !=0):
        i=n_comp
        # msg = heaan.Message(logN-1, 0)
        m=[0] * (1 << (logN-1)) 
        while (i % 4 !=0):
            m[i]=1.0000
            i+=1
        n_comp=i
        
        msg = heaan.Block(context,encrypted = False, data = m)
        c= c+msg
    msg = None

    ## Divide c by 4 chunk
    m = [0]*(1 << (logN-1))
    for i in range(n_comp//4):
        m[i]=1
    
    # msg1.set_data(data=m)
    # msg1.to_device()
    msg1 = heaan.Block(context,encrypted = False, data = m)

    # mult(c,msg1,ca, he)
    ca = c * msg1 
    check_boot(ca)
    
    # left_rotate(c,n_comp//4,ctmp1, he)
    ctmp1 = c.__lshift__(n_comp//4)
    # mult(ctmp1,msg1,cb, he)
    cb = ctmp1 * msg1
    check_boot(cb)

    # left_rotate(c,n_comp//2,ctmp1, he)
    ctmp1 = c.__lshift__(n_comp//2)
    # mult(ctmp1,msg1,cc, he)
    cc = ctmp1 * msg1
    check_boot(cc)

    # left_rotate(c,n_comp*3//4,ctmp1,he)
    ctmp1 = c.__lshift__(n_comp*3//4)
    # mult(ctmp1,msg1,cd, he)
    cd = ctmp1 * msg1
    check_boot(cd)

    c1 = ca - cb

    c2 = cb - cc

    c3 = cc - cd

    c4 = cd - ca

    c5 = ca - cc

    c6 = cb-cd    
    
    ctmp1 = c2.__rshift__(n_comp//4)

    ctmp1 = ctmp1 + c1

   
    ctmp2 = c3.__rshift__(n_comp//2)
    
    ctmp1 = ctmp1 + ctmp2
    
  
    ctmp2 = c4.__rshift__(n_comp*3//4)
 
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c5.__rshift__(n_comp)
    
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c6.__rshift__(5*n_comp//4)
 
    ctmp1 = ctmp1 + ctmp2
    
    c0 = ctmp1.sign(inplace = False, log_range=0) ## input ctxt range : -1 ~ 0 (log value)
    check_boot(c0)


    ## Making equality ciphertext 
    # ceq = heaan.Ciphertext(context)
 
    c0_c= c0
    
    mkmsg = [1.0]*num_slot
    
    mkall = heaan.Block(context,encrypted = False, data=mkmsg)

    c0 = c0 + mkall

    c0 = c0 * 0.5
    check_boot(c0)

    ceq = c0_c * c0_c
    check_boot(ceq)

    ceq = ceq.__neg__()

    ceq = ceq + mkall
    
    # print("step 6")
    ## Step 6..
    mk1=msg1

    # print("step 6-1")
    m = [0]*(1 << (logN-1))
    for i in range(n_comp//4,n_comp//2):
        m[i]=1
    
    # mk2.set_data(data=m)
    mk2 = heaan.Block(context,encrypted = False, data=m)
    
    
    m = [0]*(1 << (logN-1))
    for i in range(n_comp//2,(3*n_comp)//4):
        m[i]=1
    # mk3.set_data(data=m)
    mk3 = heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range((3*n_comp)//4,n_comp):
        m[i]=1
    # mk4.set_data(data=m)
    mk4 = heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range(n_comp,(5*n_comp)//4):
        m[i]=1
    # mk5.set_data(data=m)
    mk5 = heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range((5*n_comp)//4,(3*n_comp)//2):
        m[i]=1
    # mk6.set_data(data=m)
    mk6 = heaan.Block(context,encrypted = False, data=m)

    ## Step 7
    # print("step 7")
    # c_neg = heaan.Ciphertext(c0)
    c_neg = c0
    # he.negate(c0,c_neg)
    c_neg = c0.__neg__()

    # he.add(c_neg,mkall,c_neg) ## c_neg = 1-c0 
    c_neg = c_neg + mkall 
    
    ### When min=a    
    # c0n = heaan.Ciphertext(c0)
    c0n = c0
    # c0n.to_device()
    # mult(c_neg,mk1,ctmp1,he)
    ctmp1 = c_neg * mk1
    check_boot(ctmp1)
    ctxt=c0n
    # c0=heaan.Ciphertext(ctxt)
    c0 = ctxt
    # c0.to_device()
    # mult(c0,mk4,ctmp2, he)
    ctmp2 = c0 * mk4
    check_boot(ctmp2)
    # left_rotate(ctmp2,(3*n_comp)//4,ctmp2, he)
    ctmp2 = ctmp2.__lshift__((3*n_comp)//4)
    
    
    # cda = heaan.Ciphertext(ctmp2) ## (d>a)
    # cda.to_device()
    cda = ctmp2
    # mult(ctmp1,ctmp2,ctmp1, he)
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)
    # mult(c_neg,mk5,ctmp2,he)
    ctmp2 = c_neg * mk5
    check_boot(ctmp2)
    # left_rotate(ctmp2,n_comp,ctmp2,he)
    ctmp2 = ctmp2.__lshift__(n_comp)
    
    ## cca
    # cca = heaan.Ciphertext(context)
    # cca.to_device()
    # mult(ctmp1,ctmp2,cca,he)
    cca = ctmp1 * ctmp2
    check_boot(cca)

    ## Min=b
    # mult(c0,mk1,ctmp1, he)
    # mult(c_neg,mk2,ctmp2,he)
    # left_rotate(ctmp2,n_comp//4,ctmp2,he)
    # mult(ctmp1,ctmp2,ctmp1,he)
    ctmp1 = c0 * mk1
    check_boot(ctmp1)
    ctmp2 = c_neg * mk2
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(n_comp//4)
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    # mult(c_neg,mk6,ctmp2,he)
    ctmp2 = c_neg * mk6
    check_boot(ctmp2)
    # left_rotate(ctmp2,n_comp*5//4,ctmp2,he)
    ctmp2 = ctmp2.__lshift__(n_comp*5//4)
    # ccb = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, ccb)
    # ccb.to_device()
    ccb = ctmp1 * ctmp2
    check_boot(ccb)
    # he.mult(ctmp1,ctmp2,ccb)
    # mult(ctmp1,ctmp2,ccb,he)

    ## Min=c
    # mult(c0,mk2,ctmp1,he)
    ctmp1 = c0 * mk2
    check_boot(ctmp1)
    
    # left_rotate(ctmp1,n_comp//4,ctmp1,he)
    ctmp1 = ctmp1.__lshift__(n_comp//4)
    # cbc = heaan.Ciphertext(ctmp1) ## (b>c)
    # cbc.to_device()
    cbc = ctmp1

    # mult(c_neg,mk3,ctmp2,he)
    ctmp2 = c_neg * mk3
    check_boot(ctmp2)

    # left_rotate(ctmp2,n_comp//2,ctmp2, he)
    ctmp2 = ctmp2.__lshift__(n_comp//2)

    # mult(ctmp1,ctmp2,ctmp1, he)
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)
    # mult(c0,mk5,ctmp2,he)
    ctmp2 = c0 * mk5
    check_boot(ctmp2)
    # left_rotate(ctmp2,n_comp,ctmp2,he)
    ctmp2 = ctmp2.__lshift__(n_comp)
    # ccc = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, ccc)
    # ccc.to_device()
    # he.mult(ctmp1,ctmp2,ccc)
    # mult(ctmp1,ctmp2,ccc,he)
    ccc = ctmp1 * ctmp2
    check_boot(ccc)


    ## Min=d
    # mult(c0,mk3,ctmp1,he)
    ctmp1 = c0 * mk3
    check_boot(ctmp1)
    # left_rotate(ctmp1,n_comp//2,ctmp1,he)
    ctmp1 = ctmp1.__lshift__(n_comp//2)

    # mult(c_neg,mk4,ctmp2,he)
    ctmp2 = c_neg * mk4
    check_boot(ctmp2)
 
    # left_rotate(ctmp2,3*n_comp//4,ctmp2,he)
    ctmp2 = ctmp2.__lshift__(n_comp*3//4)
    
    # mult(ctmp1,ctmp2,ctmp1,he)
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    # mult(c0,mk6,ctmp2,he)
    ctmp2 = c0 * mk6
    check_boot(ctmp2)
  
    # left_rotate(ctmp2,5*n_comp//4,ctmp2,he)
    ctmp2 = ctmp2.__lshift__(n_comp*5//4)

    # ccd = heaan.Ciphertext(context)
    # ccd.to_device()
    # mult(ctmp1,ctmp2,ccd,he)
    ccd = ctmp1 * ctmp2
    check_boot(ccd)
    # print("step 8")


    # mult(cca,ca,cca,he)
  
    # mult(ccb,cb,ccb,he)

    # mult(ccc,cc,ccc,he)
 
    # mult(ccd,cd,ccd,he)
    cca = cca * ca
    check_boot(cca)
    ccb = ccb * cb
    check_boot(ccb)
    ccc = ccc * cc
    check_boot(ccc)
    ccd = ccd * cd
    check_boot(ccd)
    # cout = heaan.Ciphertext(cca)
    # cout.to_device()
    # 진아 ; NB에 있는 코드는 여기가 mult인데 왜 여긴 add ?
    # he.add(cout,ccb,cout)
    # he.add(cout,ccc,cout)
    # he.add(cout,ccd,cout)
    # he.bootstrap(cout, cout)
    cout = cca
 
    cout = cout + ccb
 
    cout = cout + ccc
 
    cout = cout + ccd

    ### Going to check equality
    # print("step 9")
    # ceq_ab = heaan.Ciphertext(context)
  
    # ceq_ab.to_device()
  
    # mult(ceq,mk1,ceq_ab,he)
    ceq_ab = ceq * mk1
    check_boot(ceq_ab)
    # ceq_bc = heaan.Ciphertext(context)
 
    # ceq_bc.to_device()

    # left_rotate(ceq,(n_comp)//4,ceq_bc,he)
    ceq_bc = ceq.__lshift__(n_comp//4)
    # mult(ceq_bc,mk1,ceq_bc,he)
    ceq_bc = ceq_bc * mk1
    check_boot(ceq_bc)

    # ceq_cd = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, ceq_cd)
    # ceq_cd.to_device()
    ceq_cd = m0.encrypt(inplace=False)
    # left_rotate(ceq,(n_comp)//2,ceq_cd,he) 
    ceq_cd = ceq.__lshift__(n_comp//2)
    # mult(ceq_cd,mk1,ceq_cd,he)
    ceq_cd = ceq_cd * mk1
    check_boot(ceq_cd)

    # ceq_da = heaan.Ciphertext(context)
    # ceq_da.to_device()
  
    # left_rotate(ceq,(n_comp)*3//4,ceq_da,he)
    ceq_da = ceq.__lshift__((n_comp)*3//4)
    # mult(ceq_da,mk1,ceq_da,he)
    ceq_da = ceq_da * mk1
    check_boot(ceq_da)

    ## Checking remaining depth
    # ncda = heaan.Ciphertext(cda)
    # ncda.to_device()
    # he.negate(ncda,ncda)
    # he.add(ncda,mk1,ncda)
    ncda = cda
    ncda = ncda.__neg__()
    ncda = ncda + mk1

    # ncbc = heaan.Ciphertext(cbc)
    # ncbc.to_device()
    # he.negate(ncbc,ncbc)
    # he.add(ncbc,mk1,ncbc)
    # print("d=a")
    ncbc = cbc
    ncbc = ncbc.__neg__()
    ncbc = ncbc + mk1
  
    ## (a=b)(b=c)(d>a)
 
    # mult(ceq_ab,ceq_bc,ctmp2,he)
    # mult(ctmp2,cda,ctmp1,he)
    # he.bootstrap(ctmp1, ctmp1)
    # c_cond3 = heaan.Ciphertext(ctmp1)
    # c_cond3.to_device()
    ctmp2 = ceq_ab * ceq_bc
    check_boot(ctmp2)
    c_cond3 = ctmp1
    ## (b=c)(c=d)(1-(d>a))

    # mult(ceq_bc,ceq_cd,ctmp1,he)
    # mult(ctmp1,ncda,ctmp1,he)
    # he.add(c_cond3,ctmp1,c_cond3)
    ctmp1 = ceq_bc * ceq_cd
    check_boot(ctmp1)
    ctmp1 = ctmp1 * ncda
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1

    ## (c=d)(d=a)(b>c)
  
    # mult(ceq_cd,ceq_da,ctmp1,he)
    # mult(ctmp1,cbc,ctmp1,he)
    # he.add(c_cond3,ctmp1,c_cond3)
    ctmp1 = ceq_cd * ceq_da
    check_boot(ctmp1)
    ctmp1 = ctmp1 * cbc
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1

    ## (d=a)(a=b)(1-(b>c))
  
    # mult(ceq_ab,ceq_da,ctmp1,he)
    # mult(ctmp1,ncbc,ctmp1,he)
    # he.add(c_cond3,ctmp1,c_cond3)
    # c_cond4 = heaan.Ciphertext(context)
    # c_cond4.to_device()
    # mult(ctmp2,ceq_cd,c_cond4,he)
    ctmp1 = ceq_ab * ceq_da
    check_boot(ctmp1)
    ctmp1 = ctmp1 * ncbc
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1
    c_cond4 = ctmp2 * ceq_cd
    check_boot(c_cond4)
 
    # print("step 10")

    # c_tba = heaan.Ciphertext(context)
    # c_tba.to_device()
    # mult(c_cond3,0.333333333,c_tba,he)
    # he.add(c_tba,mkall,c_tba)
    c_tba = c_cond3 * 0.333333333
    check_boot(c_tba)
    c_tba = c_tba + mkall
    # mult(c_cond4,0.333333333,c_cond4,he)
    # he.add(c_cond4,c_tba,c_tba)
    c_cond4 = c_cond4 * 0.333333333
    c_tba = c_cond4 + c_tba
    # mult(cout,c_tba,cout,he)
    cout = cout * c_tba
    check_boot(cout)

    del m, msg1, ca, cb, cc, cd, c1, c2, c3, c4, c5, c6, ctmp1, ctmp2, c0, c0_c
    del mkall, ceq, mk1, mk2, mk3, mk4, mk5, mk6, c_neg, c0n, ctxt, cda, cca, ccb, cbc
    del ccc, ccd, ceq_ab, ceq_bc, ceq_cd, ceq_da, ncda, ncbc, c_cond3, c_cond4
    return findMin4(cout, context, logN, d, n, n_comp//4)

# in 3
def selectRandomOnePos(c_red,context,ndata):
    print('=======selectRandomOnePOs 입력암호문 ==========')
    print_ctxt(c_red,ndata)
    # eval_.bootstrap(c_red, c_red)
    # m0 = heaan.Message(logN-1,0)
    # c_sel = heaan.Ciphertext(context)
    # enc.encrypt(m0,kpack,c_sel)
    # c_sel.to_device()
    check_boot(c_red)
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    c_sel = m0.encrypt(inplace=False)
    
    rando = np.random.permutation(ndata)
    # ctmp1 = heaan.Ciphertext(c_red)
    # ctmp1.to_device()
    # ctmp2 = heaan.Ciphertext(context)
    # ctmp2.to_device()
    ctmp1 = c_red
    empty_msg= heaan.Block(context,encrypted = False)
    ctmp2 = empty_msg.encrypt(inplace=False)

    # m0_ = [1] + [0]*(num_slot-1)
    # m0.set_data(m0_)
    m0_ = [1] + [0]*(num_slot-1)
    m0 = heaan.Block(context,encrypted = False, data = m0_)
    
    for l in rando:
        if (l>0):
            # left_rotate(ctmp1,l,ctmp1,eval_)
            # mult(c_sel,ctmp1,ctmp2,eval_)
            # eval_.sub(ctmp1,ctmp2,ctmp1)
            ctmp1 = ctmp1.__lshift__(l)
            ctmp2 = c_sel * ctmp1
            check_boot(ctmp2)
            ctmp1 = ctmp1 - ctmp2
            
            # m0.to_device() 
            # mult(ctmp1,m0,ctmp2,eval_)
            # right_rotate(ctmp1,l,ctmp1,eval_)
            # eval_.add(c_sel,ctmp2,c_sel)
            ctmp2 = ctmp1 * m0
            check_boot(ctmp2)
            ctmp1 = ctmp1.__rshift__(l)
            c_sel = c_sel + ctmp2
        else:
            # mult(c_sel,ctmp1,ctmp2,eval_)
            # eval_.sub(ctmp1,ctmp2,ctmp1)
            # m0.to_device() 
            # mult(ctmp1,m0,ctmp2,eval_)
            # eval_.add(c_sel,ctmp2,c_sel)
            ctmp2 = c_sel * ctmp1
            check_boot(ctmp2)
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * m0
            check_boot(ctmp2)
            c_sel = c_sel + ctmp2
            
    ctmp2 = None
    c_sel = None
    m0 = None
    return ctmp1

#4
def find_label(train_ndata, g_list, train_ctxt, label_ctxt,n,d,t,logN,context):
    print(' --- find_label --- ')
    real_time = 0 
    
    # Rule prediction을 위한 label 값 찾음
    
    # 지니지수가 최소 찾았어
    # 그러면 label 값을 봐야대
    # R1(y=?), X0=0 까지 나왔으니 ? 찾아야 됨

    # MinPos 결과 : 0 0 0 0 1 0 ⇒ X3=1 선택
    # 첫번째 0 가져와서 전체 복사
    # 0 0 0 0 0 0 0 0 ...
    # 첫번째 train data와 곱함
    # 두 번째 0 가져와서 전체 복사
    # ...
    # 다섯 번째 1 가져와서 전체 복사
    # 1 1 1 1 1 1 1 1 ...
    # 다섯 번째 train data와 곱함
    # 1 1 1 1 1 1 × 0 0 1 1 0 = 0 0 1 1 0

    # 이렇게 구한 결과 값 싹 더함
    # 그럼 0 0 1 1 0 나오겠지 == (진아) X3=1 인 train data 개수? == c_sum

    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m100 = heaan.Block(context,encrypted = False, data = [1] + [0]*(num_slot-1))
    
    # tmp = heaan.Ciphertext(context)
    # tmp.to_device()
    empty_msg= heaan.Block(context,encrypted = False)
    tmp = empty_msg.encrypt(inplace=False) 

    c_sum = m0.encrypt(inplace=False)
    y_cy = m0.encrypt(inplace=False)
        
    start = time.time()
    for i in range(n*d):
        
        # mult(g_list[i], train_ctxt[i], tmp, eval)
        # eval.add(tmp, c_sum, c_sum)
        tmp = g_list[i] * train_ctxt[i]
        check_boot(tmp)
        c_sum = tmp + c_sum
        
    # i=0
    for i in range(t):
        # mult(c_sum, label_ctxt[i], tmp, eval)
        tmp = c_sum * label_ctxt[i]
        check_boot(tmp)
        # 다 더해서 몇개 있는지 구함
        # eval.left_rotate_reduce(tmp, 1, train_ndata, tmp)
        # mult(tmp, m100, tmp, eval)
        # right_rotate(tmp, i, tmp, eval)
        print('-------before left rotate reduce---------')
        print_ctxt(tmp,train_ndata)
        ## 진아 변경 : tmp -> tmp_after_rot
        tmp_after_rot = left_rotate_reduce(context,tmp,train_ndata,1) # train_ndata : train data row수
        tmp_after_rot = tmp_after_rot * m100
        check_boot(tmp_after_rot)
        print('-------after left rotate reduce---------')
        print_ctxt(tmp_after_rot,10)
        tmp_after_rot = tmp_after_rot.__rshift__(i)
        y_cy = tmp_after_rot + y_cy
    # y_cy : 0번째 슬롯 : X3=1 이면서 label=1 인 데이터 갯수, 1번째 슬롯 : X3=1이면서 label=2인 데이터 갯수 ... t번째 슬롯까지 값 존재

    target = y_cy
    print(' =======y_cy before scaling===========')
    print_ctxt(y_cy,t) # 진아; t개가 다 음수값이 나오는데 맞아..?
    
    ## scaling
    # mult(target, 1/(train_ndata), target, eval)
    target = target * (1/train_ndata) 
    print(' =======y_cy after scaling===========')
    print_ctxt(target,t)
    # print(' *** target')
    # print_ctxt(target,dec,sk,logN,t)
    # print()

    # slot 개수 줄 때 10보다 작으면 결과 슬롯에 1 없음
    # 두 번째 슬롯의 값이 제일 크면 두 번째 슬롯만 1이여야되는데
    # 0.2정도 나옴
    # 최소 10 slot 줘야 제대로 나옴

    cy = findMaxPos(target,context,logN,d,n,t) # 어떤 라벨 값이 제일 많은지 그 위치를 담고 있음 (앞에서 t개의 슬롯 중 하나만 1)
    check_boot(cy)
    # eval.bootstrap(cy, cy)
    end = time.time()
    real_time += end-start
    # print()
    # print(' *** target')
    # print_ctxt(target,dec,sk,logN,t)
    # print()
    print(' ========after findMaxPos =========')
    print_ctxt(cy,n*d)
    # print()

    # cy 0 1 0 0 ... 이면 2 0 0 0 ...으로 바꾸기 = c_cy (원핫인코딩 -> 값)

    start = time.time()
    # eval.left_rotate_reduce(cy,1,t,c_cy)
    # eval.left_rotate_reduce(c_cy,1,t,c_cy)
    c_cy = left_rotate_reduce(context,cy,t,1)
    
    c_cy = left_rotate_reduce(context,c_cy,t,1)

    c_cy = c_cy * m100
    check_boot(c_cy)
    
    end = time.time()
    real_time += end-start
    print('find_label real time ',f"{real_time:.8f} sec")
    print(' *** c_cy')
    print_ctxt(c_cy,t)
    print()
    print(' *** y_cy')
    print_ctxt(y_cy,t)
    print()
    
    return y_cy, cy, c_cy, c_sum

# in 4
def findMaxPos(c,context,logN,d,n,ndata):
    # c : 0~1 사이의 값을 담고 있음
    print("=================findMax4 input :")
    print_ctxt(c,ndata)
    cmax = findMax4(c,context,logN,d,n,ndata)
    
    print("=================findMax4 output :")
    print_ctxt(cmax,ndata)
  
    m100_ = [1] + [0]*(num_slot-1)
    m100 = heaan.Block(context,encrypted = False, data = m100_)
    cmax = cmax * m100
    check_boot(cmax)
    # he.right_rotate_reduce(cmax,1,num_slot,cmax)
    print('------before right_rotate_reduce----------')
    print_ctxt(cmax,n*d)
    
    cmax = right_rotate_reduce(context,cmax,num_slot,1)
    
    print('------after right_rotate_reduce----------')
    print_ctxt(cmax,10)
    # print("findmax4 결과를 가져와서 모든 슬롯에 넣은 결과 : ")
    # print_ctxt(cmax,dec,sk,logN,d*n)

    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)

    c = c - cmax
    # print("c - cmax : ")
    # print_ctxt(c,dec,sk,logN,d)

    # he.add(c,0.0001,c)
    c = c + 0.0001
    # print("c에서 조금 더 더하기 : ")
    # print_ctxt(c,dec,sk,logN,d)

    # check_boot(c, he)
    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    # c_red = heaan.Ciphertext(context)
    # enc.encrypt(m0, kpack, c_red)
    # heaan.math.approx.sign(he, c, c_red,  numiter_g=9, numiter_f=4)
    
    c_red = c.sign(inplace = False, log_range=0)
    # he.bootstrap(c_red, c_red)
    check_boot(c_red)
    
    # he.add(c_red, 1, c_red)
    # mult(c_red, 0.5, c_red, he)
    c_red = c_red + 1
    c_red = c_red * 0.5
    check_boot(c_red)
    
    ## 진아; NB에서는 sign, add, mult 이 과정 대신 c_red = c.greater_than_zero() 이렇게함

    ### Need to generate a rotate sequence to choose 1 in a random position
    c_out=selectRandomOnePos(c_red,context,ndata)
    # print("selRandOne cmin:",c_out.level)
    print("=============findMaxPos result :")
    print_ctxt(c_out,ndata)

    return c_out

# in 4
def findMax4(c, context, logN, d,n,ndata):
    
    check_boot(c)
    # enc = heaan.Encryptor(context)
    if (ndata==1): return c

    ## divide ctxt by four chunks
    if (ndata % 4 !=0):
        i=ndata
        m = [0] * (1 << (logN-1)) 
        while (i % 4 !=0):
            m[i]=0.00000
            i+=1
        ndata=i
        msg = heaan.Block(context,encrypted = False, data = m)
        c = c+msg

    ## Divide c by 4 chunk

    m = [0] * (1 << (logN-1)) 
    for i in range(ndata//4):
        m[i]=1
 
    msg1 = heaan.Block(context,encrypted = False, data = m)

    ca = c * msg1
    check_boot(ca)
    ctmp1 = c.__lshift__(ndata//4)
    cb = ctmp1 * msg1
    check_boot(cb)
    ctmp1 = c.__lshift__(ndata//2)
    cc = ctmp1 * msg1
    check_boot(cc)
    ctmp1 = c.__lshift__(ndata*3//4)
    cd = ctmp1 * msg1    
    check_boot(cd)
    

    c1= ca-cb

    c2 = cb - cc

    c3 = cc - cd

    c4 = cd - ca

    c5 = ca - cc

    c6 = cb-cd 
  
    
    ctmp1 = c2.__rshift__(ndata//4)
    ctmp1 = ctmp1 + c1

    ctmp2 = c3.__rshift__(ndata//2)
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c4.__rshift__(ndata*3//4)
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c5.__rshift__(ndata)
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c6.__rshift__(5*ndata//4)
    ctmp1 = ctmp1 + ctmp2
    
    ## 
    ##print("approxSign input")
    ##print_ctxt(ctmp1,dec,sk,17,d*n)
  
    c0 = ctmp1.sign(inplace = False, log_range=0) ## input ctxt range : -1 ~ 0 (log value)
    check_boot(c0)
    c0_c = c0

    mkall = heaan.Block(context,encrypted = False, data=[1]*num_slot)

    c0 = c0 + mkall

    c0 = c0 * 0.5
    check_boot(c0)

    ## Making equality ciphertext 

    ceq = c0_c * c0_c 
    check_boot(ceq)
    
    ceq = ceq.__neg__()
    ceq = ceq + mkall
    
    ## Step 6..
    # print("step 6")
    ## Step 6..
    mk1=msg1

    # print("step 6-1")
    m = [0]*(1 << (logN-1))
    for i in range(ndata//4,ndata//2):
        m[i]=1
    
    mk2 = heaan.Block(context,encrypted = False, data=m)
    
    m = [0]*(1 << (logN-1))
    for i in range(ndata//2,(3*ndata)//4):
        m[i]=1
    mk3= heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range((3*ndata)//4,ndata):
        m[i]=1
    mk4= heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range(ndata,(5*ndata)//4):
        m[i]=1
    mk5= heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range((5*ndata)//4,(3*ndata)//2):
        m[i]=1
    mk6= heaan.Block(context,encrypted = False, data=m)
    
    ## Step 7
    # print("step 7")

    c_neg = c0.__neg__()
    c_neg = c_neg+mkall

    ### When max=a
    ## ctmp1 = a>b
 
    c0n = c0
    ctmp1 = c0n * mk1
    check_boot(ctmp1)
    ctxt=c0n
    c_ab = ctmp1
    
    ## ctmp2 = a>d

    c0 = ctxt
    ctmp2 = c_neg * mk4   
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__((ndata*3)//4)
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)
    
    ## ctmp2 = a>c

    ctmp2 = c0 * mk5
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata)
    
    ## cca

    cca = ctmp1 * ctmp2
    check_boot(cca)
    
    ## Min=b
    ctmp1 = c_neg * mk1
    check_boot(ctmp1)

    ctmp2 = c0 * mk2
    check_boot(ctmp1)

    ctmp2 = ctmp2.__lshift__(ndata//4)

    c_bc = ctmp2

    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)


    ctmp2 = c0*mk6
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata*5//4)
    
    ccb = ctmp1 * ctmp2
    check_boot(ccb)

    ## Min=c

    ctmp1 = c_neg *mk2
    check_boot(ctmp1)
    ctmp1 = ctmp1.__lshift__(ndata//4)
    
    ctmp2 = c0 * mk3
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata//2)

    c_cd = ctmp2
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)
    ctmp2 = c_neg * mk5
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata)
    ccc = ctmp1 * ctmp2
    check_boot(ccc)
    ## Min=d
 
    ctmp1 = c_neg * mk3
    check_boot(ctmp1)
    ctmp1 = ctmp1.__lshift__(ndata//2)

    ctmp2 = c0 * mk4
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata*3//4)
    
    cda = ctmp2
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    ctmp2 = c_neg * mk6
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata*5//4)
 
    ccd= ctmp1 * ctmp2
    check_boot(ccd)
    cca = cca * ca
    check_boot(cca)
    ccb = ccb * cb
    check_boot(ccb)
    ccc = ccc * cc
    check_boot(ccc)
    ccd = ccd * cd
    
    check_boot(ccd)
   
    cout = cca
 
    cout = cout + ccb
 
    cout = cout + ccc
 
    cout = cout + ccd

    ### Going to check equality
    # print("step 9")

    cneq = ceq.__neg__()
 
    cneq = cneq + mkall
   
    cneq_da = cneq.__lshift__(ndata*3//4)
   
    cneq_da = cneq_da * mk1
    check_boot(cneq_da)
    
    cneq_bc = cneq.__lshift__(ndata//4)
    cneq_bc =cneq_bc*mk1
    check_boot(cneq_bc)


    ceq_ab = ceq * mk1
    check_boot(ceq_ab)
    ceq_bc = ceq.__lshift__(ndata//4)
    ceq_bc = ceq_bc * mk1

    ceq_cd = ceq.__lshift__(ndata//2)
    ceq_cd = ceq_cd * mk1

    ceq_da = cneq_da.__neg__()
    ceq_da = ceq_da + mk1
    check_boot(ceq_bc)
    check_boot(ceq_cd)
    check_boot(ceq_da)

    ## Need to check (a=b)(b=c)(c>d)

    ctmp2 = ceq_ab * ceq_bc
    check_boot(ctmp2)
    ctmp1 = ctmp2 * c_cd
    check_boot(ctmp1)
    c_cond3 = ctmp1

    ## (b=c)(c=d)(d>a)

    ctmp1 = ceq_bc * ceq_cd
    check_boot(ctmp1)
    ctmp1 = ctmp1 * cda
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1
    
    ## (c=d)(d=a)(a>b)

    ctmp1 = ceq_cd * ceq_da
    check_boot(ctmp1)
    ctmp1 = ctmp1 * c_ab
    check_boot(ctmp1)
    c_cond3 - c_cond3 + ctmp1
    
    ## (a=b)(d=a)(b>c)

    ctmp1 = ceq_ab + ceq_da
    ctmp1 = ctmp1 * c_bc
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1

    c_cond4 = ceq_cd* ctmp2
    check_boot(c_cond4)
   
    c_tba = c_cond3 * 0.333333333
    check_boot(c_tba)
    c_tba = c_tba + mkall
    ctmp1 = c_cond4 + mkall
    c_tba = c_tba * ctmp1
    check_boot(c_tba)
    cout = cout * c_tba
    check_boot(cout)

    ca = None
    cb = None
    cc = None
    cd = None
    c1 = None
    c2 = None
    c3 = None
    c4 = None
    c5 = None
    c6 = None
    ctmp1 = None
    ctmp2 = None

    return findMax4(cout, context, logN, d,n,ndata//4)

#5
def isReal_1(y_cy, cy,context):
    print(' --- isReal --- ')
    real_time = 0 
    
    # ca(by) 만들기
    # 진짜 값이 있는지 확인
    
    # 가장 큰 값이 있는 위치 0 1 0 0 ... #진아 ; cy (ex. X3=1 일때 어떤 라벨값이 가장 많은지 그 위치)
    # 실제 y 값의 개수 1 3 0 0 ... # 진아; y_cy
    # 두 개를 곱하면 0 3 0 0 ... # 진아; ca

    # 슬롯 전체 복사 3 3 3 3 ...
    # inverse 수행 0.33 0.33 0.33 ...... # 진아; inv

    # 0 3 0 0 × 0.33 0.33 0.33 ...
    # = 0 1 0 0 ...
    # 슬롯 전체 복사
    # 1 1 1 1 1 1 

  
    start = time.time()
 
    ca = cy * y_cy
    check_boot(ca)
    print('----ca-------')
    print_ctxt(ca,10)


    ca = left_rotate_reduce(context,ca,num_slot,1)

    print('----ca after left rotate reduce-------')
    print_ctxt(ca,10)
   
    inv = ca.inverse(greater_than_one=True) # 진아; y 개수니까 True 맞겠지?
    check_boot(inv)
    print('----inverse ca-------')
    print_ctxt(inv,10)
 
    ca = ca * inv
    check_boot(ca)


    ca = ca * ca
    check_boot(ca)

    end = time.time()
    real_time += end-start
    print('isReal real time ',f"{real_time:.8f} sec")
    return ca

#6
def create_rule(g, c_cy,n,d,logN,context):
    print(' --- create_rule --- ')
    real_time = 0
    print('---create rule input 1----')
    print_ctxt(g,n*d) # cv
    print('---create rule input 2----')
    print_ctxt(c_cy,n*d) # cy
    # cy: 가장 큰 값이 있는 위치 0 1 0 0 ... (ex. X3=1 일때 어떤 라벨값이 가장 많은지 그 위치)
    # cy 0 1 0 0 ... 이면 2 0 0 0 ...으로 바꾸기 = c_cy (원핫인코딩 -> 값)
    # g : gini 값 가장 작은 slot만 1 나머지 0 (밑에선 cv라고 표현)
    
    # Rule 하나 생성
    
    # cv랑 cy랑 곱해야 원하는 위치에 있을 수 있음
    # 여기서의 cv는 곧 MinPos
    # cy : 2 0 0 0 0 0 ....
    # MinPos : 0 0 0 0 1 0

    # cy를 n × d만큼 복사 후 MinPos와 곱함
    # 2 2 2 2 2 2
    # 0 0 0 0 1 0
    # 0 0 0 0 2 0
    # 진아 ; cy값을 MinPos 위치로 이동시킨다고 생각하면 되려나?

    # rule 한 개 완성
    
    # one_rule = heaan.Ciphertext(context)
    # one_rule.to_device()

    start = time.time()
    # eval.right_rotate_reduce(c_cy, 1, n*d, c_cy)
    c_cy = right_rotate_reduce(context,c_cy,n*d,1)
    
    # print_ctxt_1(tmp,dec,sk,logN,n*d)
    # print_ctxt_1(one_rule,dec,sk,logN,n*d)

    # mult(g, c_cy, one_rule, eval)
    one_rule = g * c_cy
    check_boot(one_rule)
    # print('one rule !! !! ')
    # print_ctxt_1(one_rule,dec,sk,logN,n*d)
    # print()
    end = time.time()
    real_time += end-start
    print('create_rule real time ',f"{real_time:.8f} sec")
    return one_rule
 
#7
def data_update_5(g_list, c_sum, train_ctxt, label_ctxt,n,d,t,logN,context):
    print(' --- data_update --- ')
    real_time = 0
    num_slot = context.num_slots
    # 데이터 업데이트 해야 함
    # 같은 attribute value 쌍 선택 방지
    
    # cv로 데이터 업데이트

    # m1 = heaan.Message(logN-1, 1)
    # ctxt_one = heaan.Ciphertext(context)
    # enc.encrypt(m1, pk, ctxt_one)
    # ctxt_one.to_device()
    m1 = heaan.Block(context,encrypted = False, data=[1]*num_slot)
    ctxt_one = m1.encrypt(inplace=False)
    
    # tmp_sub = heaan.Ciphertext(context)
    # tmp_sub.to_device()
    # tmp_rot = heaan.Ciphertext(context)
    # tmp_rot.to_device()

    # 지니지수를 다 더해버려
        
    start = time.time()
    # eval.sub(ctxt_one, c_sum, tmp_sub)
    tmp_sub = ctxt_one - c_sum

    for k in range(t):
        # mult(label_ctxt[k], tmp_sub, label_ctxt[k], eval)
        label_ctxt[k] = label_ctxt[k] * tmp_sub
        check_boot(label_ctxt[k])

    for j in range(n*d):
        # mult(train_ctxt[j], tmp_sub, train_ctxt[j], eval)
        train_ctxt[j] = train_ctxt[j] * tmp_sub
        check_boot(train_ctxt[j])
    
    for i in range(n*d):
        # eval.sub(ctxt_one, train_ctxt[i], tmp_sub)
        # mult(tmp_sub, g_list[i], tmp_rot, eval)
        tmp_sub = ctxt_one - train_ctxt[i]
        tmp_rot = tmp_sub * g_list[i]
        check_boot(tmp_rot)
        
        # eval.add(train_ctxt[i], tmp_rot, train_ctxt[i])
        train_ctxt[i] = train_ctxt[i] + tmp_rot
    end = time.time()
    real_time += end-start

    print('data_update real time ',f"{real_time:.8f} sec")
    
    return train_ctxt, label_ctxt

#8
def change_rule(g, g_list, cy, n,d,t,logN,context):
    print(' --- change_rule --- ')
    real_time = 0
    
    # m0 = heaan.Message(logN-1, 0)
    # encoding_rule = heaan.Ciphertext(context)
    # enc.encrypt(m0, pk, encoding_rule)
    # encoding_rule.to_device()
    m0 = heaan.Block(context,encrypted = False, data=[0]*context.num_slots)
    encoding_rule = m0.encrypt(inplace=False)
    # tmp_rot = heaan.Ciphertext(context)
    # tmp = heaan.Ciphertext(context)
    # print_ctxt_1(cy,dec,sk,logN,t)
    # print_ctxt_1(g,dec,sk,logN,n*d)
    
    # print_ctxt_1(g_list[i],dec,sk,logN,t)
    # print_ctxt_1(tmp_rot,dec,sk,logN,t*n*d)
    # print_ctxt_1(tmp,dec,sk,logN,t*n*d)
  
    start = time.time()
    for i in range(n*d):
        # right_rotate(cy, t*i, tmp_rot,eval)
        # mult(g_list[i], tmp_rot, tmp, eval)
        tmp_rot = cy.__lshift__(t*i)
        tmp = g_list[i] * tmp_rot
        check_boot(tmp)
        
        # eval.add(encoding_rule, tmp, encoding_rule)
        encoding_rule = encoding_rule + tmp
    end = time.time()
    real_time += end-start
    # print_ctxt_1(encoding_rule,dec,sk,logN,t*n*d)


    print('change_rule real time ',f"{real_time:.8f} sec")
    return encoding_rule


# ============================================
# ================ Inference =================
# ============================================
def inference(test, model_path,d,n,t,logN,context,qqq):
    # evaluation 수행!
    
    # test_ = test.copy()
    # for i in range(1, t+1):
    #     test_ = test_.drop(f'label_{i}', axis=1)
        
    # start = time.time()
    # cy = load_cy(model_path,logN,context,pk,enc)
    # end = time.time()
    # print()
    # print('!!!!!!! load_cy time !!!!!!! ',f"{end - start:.8f} sec")
    # print()
    start = time.time()
    cy = load_rule(model_path,context,qqq)
    end = time.time()
    print()
    print('!!!!!!! load_rule time !!!!!!! ',f"{end - start:.8f} sec")
    print()
    
    print()
    print('∂∇∂ … input_ctxt … ∂∇∂')
    print()
    
    test_ = test.copy()
    for i in range(t):
        test_ = test_.drop(f'label_{i+1}', axis=1)
    
  
    cnt_y = []
    for i in range(t):
        tmp = [0]*i + [1] + [0]*(t-i-1)
        tmp = tmp * (n*d)
        # print(cnt_mess_)
        tmp = tmp + [0]*(num_slot-len(tmp))
        
        # cnt_mess = heaan.Message(logN-1)
        # cnt_mess.set_data(tmp)
        # cnt_mess.to_device()
        cnt_mess = heaan.Block(context,encrypted = False, data=tmp)
        cnt_y.append(cnt_mess)
        
    total_eval = 0
    start_1 = time.time()
    cy_hat_list=[]
    i=0
    for i in range(test_.shape[0]):
        print('┏━━━━━━━━━━━━━━━━━━━━━┓ ')
        print('  ∵∴∵∴∵∴ row : ',i+1)
        
        x = test_.iloc[i].values.tolist()
        
        start = time.time()
        x_tmp = make_input(x,n,d,t, logN, context)
        end = time.time()
        print()
        print('≫≫≫≫≫≫ make_input time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        # print_ctxt_1(x_tmp,dec,sk,logN,18)
        
        
        start_1row = time.time()
        start = time.time()
        ccur = make_ccur(x_tmp, cy, context)
        end = time.time()
        # print('     ccur ')
        # print_ctxt_1(ccur,dec,sk,logN,n*d)
        # print_ctxt_1(ccur,dec,sk,logN,n*d)
        print()
        print('≫≫≫≫≫≫ make_ccur time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        
        start = time.time()
        # target = find_y_value(ccur,cnt_y,d,n,t,logN,context,pk,sk,eval,enc,dec)
        target = find_y_value_1(ccur,cnt_y,d,n,t,logN,context)
        end = time.time()
        # print('     target ')
        # print_ctxt_1(target,dec,sk,logN,n*d)
        print()
        print('≫≫≫≫≫≫ find_y_value time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        end_1row = time.time()
        
        # return_y = heaan.Message(logN-1)
        # target.to_host()
        # dec.decrypt(target, sk, return_y)
        return_y = target.decrypt()
        findmax = list(np.round(return_y).real)[:t]
        
        start = time.time()
        y = find_max_y_plain(findmax,d,n,t)
        end = time.time()
        print()
        print('≫≫≫≫≫≫ find_max_y_plain time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        
        
        # cy_hat_list = []
        for i in range(t):
            if round(y[i].real) == 0:
                pass
            else:
                cy_hat_list.append(i+1)
        
        total_eval += round(end_1row - start_1row, 8)
        
        print('1 row eval time  ',f"{end_1row - start_1row:.8f} sec")

    print('cy_hat_list')
    print(cy_hat_list)
    print()
    print('¤¤¤¤¤¤¤¤¤¤¤ 딱아주그냥eval만 time one ¤¤¤¤¤¤¤¤¤¤¤ ', total_eval,'sec')
    end_1 = time.time()
    print('!!!!!!! Eval time one !!!!!!! ',f"{end_1 - start_1:.8f} sec")
    
    return cy_hat_list

#1
def load_rule(model_path,context,qqq):
    print('  load_cy  ')
    
    # cy = heaan.Ciphertext(context)
    # cy.load(model_path + f'{qqq}_Rule.ctxt')
    # eval.bootstrap(cy, cy)
    # cy.to_device()
    empty_msg= heaan.Block(context,encrypted = False)
    cy = empty_msg.encrypt(inplace=False) 
    cy = cy.load(model_path + f'{qqq}_Rule.ctxt')
    check_boot(cy)
    return cy

#2
def make_input(x,n,d,t, logN, context):
    print('  make_input  ')
    
    # x = x + [0]*(num_slot-len(x))
    
    tmp = [0]*t
    tmp = tmp * (n*d)
    # len(tmp)
    cnt = 0
    for i in range(n*d):
        for j in range(t):
            cnt += 1
            # print(cnt)
            tmp[cnt-1] = x[i]
    
    x = tmp + [0]*(num_slot-len(tmp))

    # mess = heaan.Message(logN-1,0)
    # mess.set_data(x)
    mess = heaan.Block(context,encrypted = False, data=x)

    # x_tmp = heaan.Ciphertext(context)
    # enc.encrypt(mess, pk, x_tmp)
    # x_tmp.to_device()
    x_tmp = mess.encrypt(inplace=False)
    
    return x_tmp

#3
def make_ccur(x_tmp, cy, context):
    print('  make_ccur  ')
    
    # ccur = heaan.Ciphertext(context)
    # ccur.to_device()

    start = time.time()
    # input과 rule 곱해 줌
    # mult(x_tmp, cy, ccur, eval)
    ccur = x_tmp * cy
    check_boot(ccur)

    end = time.time()
    # real_time += (end-start)
    print('make_ccur real time ',f"{end - start:.8f} sec")
    return ccur

#4
def find_y_value_1(ccur,cnt_y,d,n,t,logN,context):
    # target 값이 각각 몇 개인지
    print('  find_y_value_1  ')

    m1_ = [1] + [0]*(num_slot-1)
    # m1 = heaan.Message(logN-1, 0) # 진아; 여기도 m1이고 밑에 m1.set_data(m1_)에서 m1을 또쓰네 코드가 왜이래.. 내가 걍 m1 m0로 바꿈
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    # target = heaan.Ciphertext(context)
    # enc.encrypt(m1, pk, target)
    # target.to_device()
    target = m0.encrypt(inplace=False)

    # m1.set_data(m1_)
    # m1.to_device()
    m1 = heaan.Block(context,encrypted = False, data=m1_)

    # tmp = heaan.Ciphertext(context)
    # tmp.to_device()
    empty_msg= heaan.Block(context,encrypted = False)
    tmp = empty_msg.encrypt(inplace=False) 

    i=0
    start = time.time()
    for i in range(t):
        # mult(ccur, cnt_y[i], tmp, eval)
        # eval.left_rotate_reduce(tmp, 1, n*d*t, tmp)
        # mult(tmp, m1, tmp, eval)
        tmp = ccur * cnt_y[i]
        check_boot(tmp)
        tmp = left_rotate_reduce(context,tmp,n*d*t,1)
        tmp = tmp * m1 
        check_boot(tmp)
        # right_rotate(tmp, i, tmp, eval)
        # eval.add(tmp, target, target)
        tmp = tmp.__rshift__(i)
        target = target + tmp
    end = time.time()

    # print_ctxt_1(tmp,dec,sk,logN,n*d*t)
    # print_ctxt_1(tmp,dec,sk,logN,t)
    # print_ctxt_1(target,dec,sk,logN,t)
  
    print('find_y_value real time ',f"{end-start:.8f} sec")
    
    return target

#5
def find_max_y_plain(findmax,d,n,t):
    # 빈도수가 최대인 값 찾아 줌
    print('  find_max_y  ')
    
    
    # mult(target, (1/d), target, eval)

    # target 값 find max pos 돌려서 최대값 찾아야 함
    # y = findMaxPos(target,context,pk,logN,d,n, t,dec,sk,enc,eval)
    
    # Find Max
    target_max = np.max(findmax)
    
    # 최대 값이 중복일 경우 랜덤 선택
    select_ = []
    for i in range(t):
        if findmax[i] == target_max:
            select_.append(i)
    try:
        one = random.choice(select_)
    except IndexError:
        one = random.choice(range(n*d))
        # pass
    
    # Find Max Pos
    cv = np.array([0]*t)
    for i in range(t):
        if i == one:
            cv[i] = 1
    return cv 


def accurate(df, cy_hat_list):
    
    warnings.filterwarnings('ignore') # 경고메시지 무시

    ### csv불러오기 
    # df = pd.read_csv(csv)
    label_list = df.label

    print('len(cy_hat_list): ',len(cy_hat_list))
    print('len(label_list): ',len(label_list))
    sum = 0
    # for i in range(len(label_list)):
        
    #     # cy_hat_list[i] = cy_hat_list[i]+1
    #     if label_list[i] == cy_hat_list[i]:
    #         sum += 1
    
    try:
        for i in range(len(label_list)):
            
            # cy_hat_list[i] = cy_hat_list[i]+1
            if label_list[i] == cy_hat_list[i]:
                sum += 1
    except IndexError:
        pass

    accuracy = sum/len(label_list)
    return accuracy

