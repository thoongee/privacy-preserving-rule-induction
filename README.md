# Privacy-Preserving Rule Induction using CKKS

The GPU version of HEaaN code is not available for public release as it is a proprietary asset of CryptoLab.

So, these codes are re-constructed by using HEaaN-SDK.

HEaaN-SDK is a package for data science/data analysis under homomorphic encryption using the HEaaN library.

We used the CPU version of the HEaaN-SDK, which is publicly available, to program in the same manner as described in our paper.

Costs such as communication and time might be different which is represented in the paper.

---

## Brief Explanation Of Codes

- test.py : file that calls the main code (You can specify a folder to leave a log, and you can adjust how many times you want to call the main code over and over again for a number of folds to chosen data)
- main.py : main code for training and classfy each data
- module.py : functions for training, and inference
- run.sh : shell script for running the test

## Key File Setting

keys_FGb
├─public_keypack
│  └─PK
└─secret_keypack

- When you run the code for the first time, you need to change **`generated_keys=True`** in the specified section of the **`main.py`** file. This change is necessary for the key file to be generated. It is recommended to set it to **`True`** only for the initial run, and then switch it back to **`False`** once the **`keys_FGb`** folder has been created.

```bash
context = heaan.Context(
    params,
    key_dir_path=key_file_path,
    load_keys="all",
    generate_keys=False,
)
```

## Procedure to Training and Inference the data

1. Install HEaaN.stat docker image (https://hub.docker.com/r/cryptolabinc/heaan-stat)
    
    ```bash
    docker pull cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
    ```
    
2. Create docker container
    
    ```bash
    docker run -d -p 8888:8888 --name <container-id> cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
    ```
    
3. Clone this repository in your docker container
4. According to the instructions in the Key File Setting, create a ‘keys_FGb’ directory
5. Run shell script (run.sh).
