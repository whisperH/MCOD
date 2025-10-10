# SEAOP:

## Installation
Please install the anaconda firstly.
```shell
conda create -n ppOD python=3.7 
conda activate ppOD
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorflow
```


### optional

if using **log2_qnorm** or **qnorm** in **scale_strategy**, please install R and
refer this [link](https://bioconductor.org/packages/release/bioc/html/limma.html) to install limma.

##### windows
- Install [R](https://cloud.r-project.org/bin/windows/base/) software.

- A work around is to download [rpy2](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2) from the Unofficial Windows Binaries for Python Extension Packages to the current working directory. Then use the following command to install rpy2 from the downloaded file:
    ```shell
    cd extension
    pip install rpy2-2.9.5-cp37-cp37m-win_amd64.whl
    ```
- please following [this blog](http://joonro.github.io/blog/posts/install-rpy2-windows-10/) to config the enviroment.

##### Linux
- Install R Software
check R version that the system can get, we need it >=4.2: 
    ```shell
    sudo apt-get remove r-base r-base-core r-base-dev r-recommended
    
    apt policy r-base
    ```
- install R
    ```shell
    # update indices
    sudo apt update -qq
    # install two helper packages we need
    sudo apt install --no-install-recommends software-properties-common dirmngr
    # add the signing key (by Michael Rutter) for these repos
    # To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
    sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
    sudo apt update
    sudo apt install --no-install-recommends r-base
    ```
- install limma by referring [this link](https://bioconductor.org/packages/limma/)

- install rpy2    

    ```shell
    pip install rpy2
    ```
# Run Outliers Detection

## train model

### for SEAOP
please make sure these yields are "True" value in XXX.yaml
```yaml
runSingleOD: true # step 1: train all single model. if false, load candidate pool from args.external_outlier_candidates
runFakeData: true # step 2: (see M4) generate simulated data via feature shuffle with different threshold
runSingleInfer: true # Step 3: test single model on fake data
runFakeStat: true # Step 4: run model on the fake data
runModelSelect: true # Step 4: run model on the fake data
```
```bash for select best parameters and models
python train_SEAOP.py -c ../../configs/Hela/boost_100.yaml
```

## infer model
1. create new folder with path: 
```bash
{logs_dir}\outliers
```
where "logs_dir" is defined in XXXX.yaml with "logs_dir" 

2. putting [model_selection](https://github.com/whisperH/SEAOP/tree/main/Result/Hela_boost_repeat100/outliers) in .\Result\Hela_boost_repeat100\outliers

3. please make sure these yields as following in XXX.yaml
4. infer new dataset
```bash for infer new dataset
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HCC_P.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HCC_T.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile LADC_N.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile LADC_T.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HelaGroups.csv runInferSingle True runBoostTest True multiple_thread False
```

```bash for time cost of single model
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HelaGroups.csv runInferSingle True runBoostTest False multiple_thread False
```

5. validate accuracy on train dataset with fake data
   1. after training, do following bash
```bash for validate on Fake data
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile fake-shuffle_ratio-0.1-repeat_time-0.csv runInferSingle false runBoostTest false validateFakeData true
```

# MC-OD:
## dataset1: test on simulated data of HeLa
``` bash
python3 train_OD.py -c ../../configs/Hela/AE/AE_MEM10_S20.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-1.gz
```

## dataset2: detect outlier in HCC Data
``` bash
python3 train_OD.py -c ../../configs/HCC/AE/AE_MEM_CLS_T.yaml data_dir datasets/Format train_status true infer_status false eval_status false
python3 train_OD.py -c ../../configs/HCC/AE/AE_MEM_CLS_P.yaml data_dir datasets/Format train_status true infer_status false eval_status false
```

## dataset3: detect outlier in LADC Data
``` bash
python3 train_OD.py -c ../../configs/LADC/AE/AE_MEM_CLS_N.yaml data_dir datasets/Format train_status true infer_status false eval_status false
python3 train_OD.py -c ../../configs/LADC/AE/AE_MEM_CLS_T.yaml data_dir datasets/Format train_status true infer_status false eval_status false
```

[comment]: <> (https://jundongl.github.io/scikit-feature/datasets.html)

# https://cdn.sciengine.com/doi/pdfView/f2a156954eec40c09099c50479f81bbd