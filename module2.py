import pandas as pd
import numpy as np
import heaan_sdk as heaan
import random
import os
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import math

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
        if np.abs(m[i].real) < 0.0001:
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

def get_smallest_pow_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def left_rotate_reduce(context,data,gs,interval):
    num_slot = context.num_slots
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
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

    i = len(binary_list)-1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data

            while ind < i:
                
                rot = tmp.__lshift__(s)

                tmp = tmp + rot

                s = s*2
                ind = ind+1
            if sdind > 0:
                tmp = tmp.__lshift__(sdind)
 
            res = res + tmp

            sdind = sdind + s
        i = i - 1            

    del  rot, tmp
    return res

def right_rotate_reduce(context, data, gs, interval):
    num_slot= context.num_slots
    m0 = heaan.Block(context, encrypted=False, data=[0]*num_slot)
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
                
                rot = tmp.__rshift__(s) 

                tmp = tmp + rot
 
                s = s*2
                ind = ind + 1
            if sdind > 0:
                tmp = tmp.__rshift__(sdind)  

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
    
    ndata, d, n, t, train_ndata, test_ndata = meta_data_eval(df1, df2)
    Metadata = {'ndata':ndata,
                'n':n,
                'd':d,
                't':t,
                'train_ndata': train_ndata,
                'test_ndata' : test_ndata
                }
    
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
    
    num_slot = context.num_slots
    
    global attribute_value_pair
    attribute_value_pair = []
    for i in range(d):
        for j in range(n):
            attribute_value_pair.append('X'+ str(i+1) + '_' + str(j+1))
    

    start = time.time()
    train_ctxt, label_ctxt = input_training(train,train_ndata, n,d,t,num_slot,context)
    end = time.time()

    print('≫≫≫ input_training time :',f"{end - start:.8f} sec")
    print()
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    Rule = m0.encrypt(inplace=False)
    
    # for i in range((n*d)):
    for i in range(10):
        
        print('≫≫≫ feature: ',i)
        print()
        start = time.time()
        frequency, fre_label = measure_frequency(train_ndata, train_ctxt, label_ctxt, n,d,t,num_slot,context) # ok
        end = time.time()
      
        print('≫≫≫ measure_frequency time : ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        g_list = calculate_Gini(train_ndata, frequency, fre_label, n,d,t,logN,num_slot,context)
        end = time.time()

        print('≫≫≫ calculate_Gini time : ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        y_cy, cy, c_sum = find_label(train_ndata, g_list, train_ctxt, label_ctxt,n,d,t,logN,num_slot,context)
        end = time.time()
     
        print('≫≫≫ find_label time : ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        ca = isReal(y_cy, cy,context) 
        end = time.time()

        print()
        print('≫≫≫ isReal time : ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        train_ctxt, label_ctxt = data_update(g_list, c_sum, train_ctxt,label_ctxt,n,d,t,context)
        end = time.time()
 
        print()
        print('≫≫≫ data_update time : ',f"{end - start:.8f} sec")
        print()
        
        start = time.time()
        encoding_rule = change_rule(g_list,cy,n,d,t,context)
        end = time.time()
 
        print()
        print('≫≫≫≫≫≫ change_rule time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        print()
        
        
        start = time.time()
        # If data corresponding to the rule exists, keep the rule
        one_rule  = encoding_rule * ca
        check_boot(one_rule)
        Rule = Rule + one_rule
        end = time.time()
  
        print('≫≫≫ rule_add time : ',f"{end - start:.8f} sec")
        print()
    
    start = time.time()
    Rule.save(model_path + f'{qqq}_Rule.ctxt')
    end = time.time()
    print()
    print('========== Total rule ==========')
    print_ctxt_1(Rule,n*d)

    print()
    print('≫≫≫ rule_save time : ',f"{end - start:.8f} sec")
    print()

#1
def input_training(train,train_ndata, n,d,t,num_slot,context):
    print(' --- input_training --- ')
    # training data encryption
    # Encrypt the ciphertext for each feature
    
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

        mess = heaan.Block(context, data = x, encrypted=False)
        x_tmp = mess.encrypt(inplace=False)
        train_ctxt.append(x_tmp)
        
    label_ctxt = []
    for i in range(t):
        x = train['label_' + str(i+1)].values.tolist() + [0]*(num_slot-train_ndata)

        mess = heaan.Block(context, data = x, encrypted=False)
        x_tmp = mess.encrypt(inplace=False)
        
        if t > 3:
            x_tmp = x_tmp * (1/t)
            check_boot(x_tmp)
        
        label_ctxt.append(x_tmp)
    
    return train_ctxt, label_ctxt
#2
def measure_frequency(train_ndata, train_ctxt, label_ctxt, n,d,t,num_slot,context):
    
    # Calculating frequency for computing the Gini index
    
    real_time = 0
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
   
    m100 = heaan.Block(context,encrypted = False, data = [1] + [0]*(num_slot-1))

    frequency = []
    
    etc_time = 0
    start = time.time()
    for i in range(t):
        start_1 = time.time()
 
        fre_label = m0.encrypt(inplace=False)
        end_1 = time.time()
        etc_time += end_1 - start_1
        
        for j in range(n*d):
  
            fre_tmp = label_ctxt[i] * train_ctxt[j]
            check_boot(fre_tmp)

            fre_tmp = left_rotate_reduce(context,fre_tmp,train_ndata,1)
            check_boot(fre_tmp)
            
            # masking
            fre_tmp = fre_tmp * m100
            check_boot(fre_tmp)
      
            fre_tmp = fre_tmp.__rshift__(j)
      
            fre_label = fre_label + fre_tmp
     
        frequency.append(fre_label)
    end = time.time()
    real_time += (end-start) - etc_time

    # Count the total frequency
    total_fre = m0.encrypt(inplace=False)
    
    start = time.time()
    for i in range(t):    
 
        total_fre = total_fre + frequency[i]
    end = time.time()
    real_time += end-start
    print('≫≫≫ measure_frequency real time ',f"{real_time:.8f} sec")
    
    return frequency, total_fre
#3
def calculate_Gini(train_ndata, frequency, fre_label, n,d,t,logN,num_slot,context):

    # Calculate the Gini index using the measured frequencies
    
    real_time = 0 
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    
    start = time.time()
    tmp = fre_label * (1/train_ndata)
    check_boot(tmp)

    square_total = tmp * tmp
    check_boot(square_total)

    end = time.time()
    real_time += end-start

    square_label = m0.encrypt(inplace=False)
    start = time.time()
    for i in range(t):

        tmp = frequency[i] * (1/train_ndata)
        check_boot(tmp)
 
        gini = tmp*tmp
        check_boot(gini)

        square_label = square_label + gini

    gini = square_total - square_label


    min_gini = gini
    
    g = findMinPos(min_gini, context,logN,d,n,n*d,num_slot)
    end = time.time()
    real_time += end-start
    
    n_d_mask = heaan.Block(context,encrypted = False, data = [1]*n*d + [0]*(num_slot-n*d))

    start = time.time()
    
    g = g * n_d_mask # masking
    check_boot(g)
    end = time.time()
    real_time += end-start
 
    
    g_list = []
    etc_time = 0
    start = time.time()
    for i in range(n*d):
        start_1 = time.time()
        pick_ = [0]*i + [1] + [0]*(num_slot-(i+1))

        pick = heaan.Block(context,encrypted = False, data = pick_)
        
        end_1 = time.time()
        etc_time += end_1 - start_1
        tmp = g * pick
        check_boot(tmp)

        tmp = left_rotate_reduce(context,tmp,num_slot,1)
        
        g_list.append(tmp)
    end = time.time()
    real_time += (end-start)-etc_time
    print('≫≫≫ calculate_Gini real time ',f"{real_time:.8f} sec")
    return g_list

# in 3
def findMinPos(c, context,logN,d,n,n_comp,num_slot):


    cmin = findMin4(c,context,logN,d,n,n_comp,num_slot)
  
    m100 = heaan.Block(context,encrypted = False, data = [1] + [0]*(num_slot-1))

    cmin = cmin * m100 # masking
    check_boot(cmin)

    cmin = right_rotate_reduce(context,cmin,num_slot,1)
    
    c = c - cmin

    c = c - 1/2**15

    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    
    c_red = c.sign(inplace = False, log_range=0)
    check_boot(c_red)

    c_red = c_red.__neg__()
    check_boot(c_red)

    c_red = c_red + 1
 
    c_red = c_red * 0.5
    check_boot(c_red)


    ## Need to generate a rotate sequence to choose 1 in a random position   

    c_out = selectRandomOnePos(c_red,context,n_comp)
    print('==========selectRandomOnePos 결과암호문 ===================')
    print_ctxt_1(c_out,d*n)

    return c_out

# in 3
def findMin4(c, context, logN, d, n, n_comp,num_slot):
    
    if (n_comp==1): 
        print("findMin4 ends..")
        print_ctxt(c,1)
        return c
    check_boot(c)
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)


    ## divide ctxt by four chunks
    if (n_comp % 4 !=0):
        i=n_comp

        m=[0] * (1 << (logN-1)) 
        while (i % 4 !=0):
            m[i]=1.0000
            i+=1
        n_comp=i
        
        msg = heaan.Block(context,encrypted = False, data = m)
        c= c+msg
    msg = None


    m = [0]*(1 << (logN-1))
    for i in range(n_comp//4):
        m[i]=1
    
    msg1 = heaan.Block(context,encrypted = False, data = m)


    ca = c * msg1 
    check_boot(ca)
    
    ctmp1 = c.__lshift__(n_comp//4)
    cb = ctmp1 * msg1
    check_boot(cb)


    ctmp1 = c.__lshift__(n_comp//2)

    cc = ctmp1 * msg1
    check_boot(cc)


    ctmp1 = c.__lshift__(n_comp*3//4)
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
    
    ## Step 6..
    mk1=msg1

    # step 6-1
    m = [0]*(1 << (logN-1))
    for i in range(n_comp//4,n_comp//2):
        m[i]=1
    

    mk2 = heaan.Block(context,encrypted = False, data=m)
    
    m = [0]*(1 << (logN-1))
    for i in range(n_comp//2,(3*n_comp)//4):
        m[i]=1

    mk3 = heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range((3*n_comp)//4,n_comp):
        m[i]=1
 
    mk4 = heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range(n_comp,(5*n_comp)//4):
        m[i]=1

    mk5 = heaan.Block(context,encrypted = False, data=m)

    m = [0]*(1 << (logN-1))
    for i in range((5*n_comp)//4,(3*n_comp)//2):
        m[i]=1

    mk6 = heaan.Block(context,encrypted = False, data=m)

    ## Step 7

    c_neg = c0

    c_neg = c0.__neg__()

    c_neg = c_neg + mkall 
    
    ## When min=a    
    c0n = c0

    ctmp1 = c_neg * mk1
    check_boot(ctmp1)
    ctxt=c0n

    c0 = ctxt

    ctmp2 = c0 * mk4
    check_boot(ctmp2)

    ctmp2 = ctmp2.__lshift__((3*n_comp)//4)
    
    cda = ctmp2

    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    ctmp2 = c_neg * mk5
    check_boot(ctmp2)

    ctmp2 = ctmp2.__lshift__(n_comp)
    
    cca = ctmp1 * ctmp2
    check_boot(cca)

    ## Min=b

    ctmp1 = c0 * mk1
    check_boot(ctmp1)
    ctmp2 = c_neg * mk2
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(n_comp//4)
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    ctmp2 = c_neg * mk6
    check_boot(ctmp2)

    ctmp2 = ctmp2.__lshift__(n_comp*5//4)

    ccb = ctmp1 * ctmp2
    check_boot(ccb)


    ## Min=c

    ctmp1 = c0 * mk2
    check_boot(ctmp1)
    

    ctmp1 = ctmp1.__lshift__(n_comp//4)

    cbc = ctmp1


    ctmp2 = c_neg * mk3
    check_boot(ctmp2)


    ctmp2 = ctmp2.__lshift__(n_comp//2)


    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    ctmp2 = c0 * mk5
    check_boot(ctmp2)

    ctmp2 = ctmp2.__lshift__(n_comp)

    ccc = ctmp1 * ctmp2
    check_boot(ccc)


    ## Min=d

    ctmp1 = c0 * mk3
    check_boot(ctmp1)

    ctmp1 = ctmp1.__lshift__(n_comp//2)


    ctmp2 = c_neg * mk4
    check_boot(ctmp2)
 
    ctmp2 = ctmp2.__lshift__(n_comp*3//4)
    
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    ctmp2 = c0 * mk6
    check_boot(ctmp2)
  

    ctmp2 = ctmp2.__lshift__(n_comp*5//4)

    ccd = ctmp1 * ctmp2
    check_boot(ccd)
    
    ## Step 8

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

    ## Going to check equality
    ## Step 9
 
    ceq_ab = ceq * mk1
    check_boot(ceq_ab)
  
    ceq_bc = ceq.__lshift__(n_comp//4)
    ceq_bc = ceq_bc * mk1
    check_boot(ceq_bc)

    ceq_cd = m0.encrypt(inplace=False)
    ceq_cd = ceq.__lshift__(n_comp//2)
    ceq_cd = ceq_cd * mk1
    check_boot(ceq_cd)

    ceq_da = ceq.__lshift__((n_comp)*3//4)
    ceq_da = ceq_da * mk1
    check_boot(ceq_da)

    ## Checking remaining depth

    ncda = cda
    ncda = ncda.__neg__()
    ncda = ncda + mk1

    ncbc = cbc
    ncbc = ncbc.__neg__()
    ncbc = ncbc + mk1
  
    ## (a=b)(b=c)(d>a)

    ctmp2 = ceq_ab * ceq_bc
    check_boot(ctmp2)
    c_cond3 = ctmp1
    
    ## (b=c)(c=d)(1-(d>a))

    ctmp1 = ceq_bc * ceq_cd
    check_boot(ctmp1)
    ctmp1 = ctmp1 * ncda
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1

    ## (c=d)(d=a)(b>c)
  
    ctmp1 = ceq_cd * ceq_da
    check_boot(ctmp1)
    ctmp1 = ctmp1 * cbc
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1

    ## (d=a)(a=b)(1-(b>c))
  
    ctmp1 = ceq_ab * ceq_da
    check_boot(ctmp1)
    ctmp1 = ctmp1 * ncbc
    check_boot(ctmp1)
    c_cond3 = c_cond3 + ctmp1
    c_cond4 = ctmp2 * ceq_cd
    check_boot(c_cond4)
 
    ## Step 10

    c_tba = c_cond3 * 0.333333333
    check_boot(c_tba)
    c_tba = c_tba + mkall

    c_cond4 = c_cond4 * 0.333333333
    c_tba = c_cond4 + c_tba
    cout = cout * c_tba
    check_boot(cout)

    del m, msg1, ca, cb, cc, cd, c1, c2, c3, c4, c5, c6, ctmp1, ctmp2, c0, c0_c
    del mkall, ceq, mk1, mk2, mk3, mk4, mk5, mk6, c_neg, c0n, ctxt, cda, cca, ccb, cbc
    del ccc, ccd, ceq_ab, ceq_bc, ceq_cd, ceq_da, ncda, ncbc, c_cond3, c_cond4
    return findMin4(cout, context, logN, d, n, n_comp//4, num_slot)

# in 3
def selectRandomOnePos(c_red,context,ndata):
    print('=======selectRandomOnePOs 입력암호문 ==========')
    print_ctxt_1(c_red,ndata)

    check_boot(c_red)
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    c_sel = m0.encrypt(inplace=False)
    
    rando = np.random.permutation(ndata)

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
def find_label(train_ndata, g_list, train_ctxt, label_ctxt,n,d,t,logN,num_slot,context):
 
    real_time = 0 
 
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m100 = heaan.Block(context,encrypted = False, data = [1] + [0]*(num_slot-1))
    
    empty_msg= heaan.Block(context,encrypted = False)
    tmp = empty_msg.encrypt(inplace=False) 

    c_sum = m0.encrypt(inplace=False)
    y_cy = m0.encrypt(inplace=False)
        
    start = time.time()
    for i in range(n*d):
        
        tmp = g_list[i] * train_ctxt[i]
        check_boot(tmp)
        c_sum = tmp + c_sum

    for i in range(t):

        tmp = c_sum * label_ctxt[i]
        check_boot(tmp)

        tmp_after_rot = left_rotate_reduce(context,tmp,train_ndata,1) # train_ndata : the number of train data row
        tmp_after_rot = tmp_after_rot * m100
        check_boot(tmp_after_rot)

        tmp_after_rot = tmp_after_rot.__rshift__(i)
        y_cy = tmp_after_rot + y_cy

    target = y_cy
    
    target = target * (1/train_ndata) 
    check_boot(target)


    cy = findMaxPos(target,context,logN,d,n,t) # position of the label value that occurs most frequently among the slots (where only one out of t slots is '1')
    check_boot(cy)

    end = time.time()
    real_time += end-start

    print('≫≫≫ find_label time: ',f"{real_time:.8f} sec")
    print()
 
    return y_cy, cy, c_sum

# in 4
def findMaxPos(c,context,logN,d,n,ndata):
    # c : contain the values between 0 to 1
    
    num_slot = context.num_slots
   
    cmax = findMax4(c,context,logN,d,n,ndata)

  
    m100_ = [1] + [0]*(num_slot-1)
    m100 = heaan.Block(context,encrypted = False, data = m100_)
    cmax = cmax * m100
    check_boot(cmax)
    
    cmax = right_rotate_reduce(context,cmax,num_slot,1)

    # Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    # Possible that c_red may have many 1s. We need to select one (in current situation 20220810)

    c = c - cmax

    c = c + 0.0001

    c_red = c.sign(inplace = False, log_range=0)
    check_boot(c_red)
    

    c_red = c_red + 1
    c_red = c_red * 0.5
    check_boot(c_red)

    # Need to generate a rotate sequence to choose 1 in a random position
    c_out=selectRandomOnePos(c_red,context,ndata)

    return c_out

# in 4
def findMax4(c, context, logN, d,n,ndata):
    
    check_boot(c)
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
    
    ## Step 6

    mk1=msg1
    
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

    c_neg = c0.__neg__()
    c_neg = c_neg+mkall
    print_ctxt(c_neg,10)
    
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
    check_boot(ctmp2)

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
    ctmp2 = ctmp2.__lshift__(3*ndata//4)
    
    cda = ctmp2
    ctmp1 = ctmp1 * ctmp2
    check_boot(ctmp1)

    ctmp2 = c_neg * mk6
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(5*ndata//4)
 
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
    ## Step 9

    cneq = ceq.__neg__()
 
    cneq = cneq + mkall
   
    cneq_da = cneq.__lshift__((3*ndata)//4)
   
    cneq_da = cneq_da * mk1
    check_boot(cneq_da)
    
    cneq_bc = cneq.__lshift__(ndata//4)
    cneq_bc =cneq_bc*mk1
    check_boot(cneq_bc)


    ceq_ab = ceq * mk1
    check_boot(ceq_ab)
    
    ceq_bc = ceq.__lshift__(ndata//4)
    ceq_bc = ceq_bc * mk1
    check_boot(ceq_bc)
    
    ceq_cd = ceq.__lshift__(ndata//2)
    ceq_cd = ceq_cd * mk1
    check_boot(ceq_cd)
    
    ceq_da = cneq_da.__neg__()
    ceq_da = ceq_da + mk1
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

    ctmp1 = ceq_ab * ceq_da
    check_boot(ctmp1)
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
def isReal(y_cy, cy,context):
    
    num_slot = context.num_slots
    real_time = 0 
  
    start = time.time()

    ca = cy * y_cy
    check_boot(ca)

    ca = left_rotate_reduce(context,ca,num_slot,1)

    inv = ca.inverse(greater_than_one=True)
 
 
    tmp_ca = ca * inv
    check_boot(tmp_ca)


    tmp_res = tmp_ca.sign() 
    check_boot(tmp_res)

    tmp_res = tmp_res + 1

    res = tmp_res * 0.5
    check_boot(res)

    end = time.time()
    real_time += end-start
    print('≫≫≫ isReal real time : ',f"{real_time:.8f} sec")
    return res

#6
def data_update(g_list, c_sum, train_ctxt, label_ctxt,n,d,t,context):

    # Update the data to prevent the selection of the same attribute value pair
    
    real_time = 0
    num_slot = context.num_slots
    
   
    m1 = heaan.Block(context,encrypted = False, data=[1]*num_slot)
    ctxt_one = m1.encrypt(inplace=False)
    
        
    start = time.time()
    tmp_sub = ctxt_one - c_sum

    for k in range(t):

        label_ctxt[k] = label_ctxt[k] * tmp_sub
        check_boot(label_ctxt[k])

    for j in range(n*d):
        train_ctxt[j] = train_ctxt[j] * tmp_sub
        check_boot(train_ctxt[j])
    
    for i in range(n*d):

        tmp_sub = ctxt_one - train_ctxt[i]
        tmp_rot = tmp_sub * g_list[i]
        check_boot(tmp_rot)
 
        train_ctxt[i] = train_ctxt[i] + tmp_rot
    end = time.time()
    real_time += end-start

    print('≫≫≫ data_update real time ',f"{real_time:.8f} sec")
    
    return train_ctxt, label_ctxt

#7
def change_rule(g_list, cy, n,d,t,context):
    num_slot = context.num_slots
    real_time = 0
    print('----cy----')
    print_ctxt_1(cy,n*d)
    
    for i in range(n*d):
        print('---g_list[',i,']:-------')
        print_ctxt_1(g_list[i],n*d)
    

    m0 = heaan.Block(context,encrypted = False, data=[0]*num_slot)
    encoding_rule = m0.encrypt(inplace=False)
  
    start = time.time()
    for i in range(n*d):

        tmp_rot = cy.__rshift__(t*i)
        check_boot(tmp_rot)
        tmp = g_list[i] * tmp_rot
        check_boot(tmp)
        encoding_rule = encoding_rule + tmp
        
    end = time.time()
    real_time += end-start

    print('≫≫≫ change_rule real time ',f"{real_time:.8f} sec")
    return encoding_rule


# ============================================
# ================ Inference =================
# ============================================
def inference(test, model_path,d,n,t,logN,context,qqq):
    
    num_slot = context.num_slots
    
    start = time.time()
       
    cy = load_rule(model_path,context,qqq)

    end = time.time()
    print()
    print('≫≫≫ load_rule time : ',f"{end - start:.8f} sec")
    print()
    
    test_ = test.copy()
    for i in range(t):
        test_ = test_.drop(f'label_{i+1}', axis=1)
    
  
    cnt_y = []
    for i in range(t):
        tmp = [0]*i + [1] + [0]*(t-i-1)
        tmp = tmp * (n*d)
        tmp = tmp + [0]*(num_slot-len(tmp))
        cnt_mess = heaan.Block(context,encrypted = False, data=tmp)
        cnt_y.append(cnt_mess)
        
    total_eval = 0
    cy_hat_list=[]

    for i in range(test_.shape[0]):
        print('≫≫≫ row : ',i+1)
        
        x = test_.iloc[i].values.tolist()
        
        start = time.time()
        x_tmp = make_input(x, n, d, t, num_slot, context)
        end = time.time()

        print('≫≫≫ make_input time : ',f"{end - start:.8f} sec")
        
        start_1row = time.time()
        start = time.time()
        ccur = make_ccur(x_tmp, cy, context)
        end = time.time()

        print()
        print('≫≫≫ make_ccur time : ',f"{end - start:.8f} sec")
        
        start = time.time()
        target = find_y_value(ccur,cnt_y,d,n,t,num_slot,context)
        end = time.time()
        print()
        print('≫≫≫≫≫≫ find_y_value time ≪≪≪≪≪≪ ',f"{end - start:.8f} sec")
        end_1row = time.time()
 

        return_y = target.decrypt(inplace=False)
        findmax = []
        for x in range(t):
            findmax.append(np.round(return_y[x]).real)
        
        start = time.time()
        y = find_max_y_plain(findmax,d,n,t)
        end = time.time()
      
        print()
        print('≫≫≫ find_max_y_plain time : ',f"{end - start:.8f} sec")
        print()
        
        for j in range(t):
            if round(y[j].real) == 0:
                pass
            else:
                cy_hat_list.append(j+1)

        print('======= predict value: ',cy_hat_list)
        total_eval += round(end_1row - start_1row, 8)
        
        print('≫≫≫ 1 row evaluation time : ',f"{end_1row - start_1row:.8f} sec")

    print('=====total predict result=====')
    print(cy_hat_list)
    print()
    print('≫≫≫ Total evaluation time : ', total_eval,'sec')
    
    return cy_hat_list

#1
def load_rule(model_path,context,qqq):
    
    empty_msg= heaan.Block(context,encrypted = False)
    cy = empty_msg.encrypt(inplace=False) 
    file_path = str(model_path + f'{qqq}_Rule.ctxt')
    cy = cy.load(file_path)
    if cy.level <=3:
        cy.bootstrap()
    
    return cy

#2
def make_input(x,n,d,t, num_slot,context):
    
    tmp = [0]*t
    tmp = tmp * (n*d)
    cnt = 0
    for i in range(n*d):
        for j in range(t):
            cnt += 1
            tmp[cnt-1] = x[i]
    
    x = tmp + [0]*(num_slot-len(tmp))

    mess = heaan.Block(context,encrypted = False, data=x)
    x_tmp = mess.encrypt(inplace=False)
    
    return x_tmp

#3
def make_ccur(x_tmp, cy, context):

    start = time.time()
    # input * rule
    ccur = x_tmp * cy
    check_boot(ccur)

    end = time.time()
    print('≫≫≫ make_ccur real time ',f"{end - start:.8f} sec")
    return ccur

#4
def find_y_value(ccur,cnt_y,d,n,t,num_slot,context):
    # Count the number of each target value

    m1_ = [1] + [0]*(num_slot-1)
    m1 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    target = m1.encrypt(inplace=False)

    m1 = heaan.Block(context,encrypted = False, data=m1_)

 
    empty_msg= heaan.Block(context,encrypted = False)
    tmp = empty_msg.encrypt(inplace=False) 

    i=0
    start = time.time()
    for i in range(t):
        tmp = ccur * cnt_y[i]
        check_boot(tmp)
        
        tmp = left_rotate_reduce(context,tmp,n*d*t,1)
        tmp = tmp * m1 
        check_boot(tmp)
        
        tmp = tmp.__rshift__(i)
        target = target + tmp
        
    end = time.time()
  
    print('≫≫≫ find_y_value real time ',f"{end-start:.8f} sec")
    
    return target

#5
def find_max_y_plain(findmax,d,n,t):
    # Find the target value with the maximum frequency   
    
    ## Find Max
    target_max = np.max(findmax)
    
    # If there are multiple maximum values, select one randomly
    select_ = []
    for i in range(t):
        if findmax[i] == target_max:
            select_.append(i)
    try:
        one = random.choice(select_)
    except IndexError:
        one = random.choice(range(n*d))
    
    # Find Max Position
    cv = np.array([0]*t)
    for i in range(t):
        if i == one:
            cv[i] = 1
    return cv 


def accurate(df, cy_hat_list):
    
    warnings.filterwarnings('ignore') 

    label_list = df.label

    sum = 0

    try:
        for i in range(len(label_list)):
            
            # cy_hat_list[i] = cy_hat_list[i]+1
            if label_list[i] == cy_hat_list[i]:
                sum += 1
    except IndexError:
        pass

    accuracy = sum/len(label_list)
    return accuracy

