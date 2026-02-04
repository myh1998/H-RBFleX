# H-RBFleX

## Folder Descriptions

### Image Classification

Dataset: imageNet  
DNN design space: NATS-bench-SSS  
Hardware design space (simulator): Scale-Sim

- **H-RBFleX_Large_noband**  
  H-RBFleX (Proposed method)  
  RBFleX-NAS creates a ranking table before the Bayesian optimization.  
  Ranking table: ```Score_table_241126_ImageNet16-120.csv```

- **H-WOT_Large_noband**  
  Bayesian optimization with a ranking table made by NASWOT.  
  NASWOT creates a ranking table before the Bayesian optimization.  
  Ranking table: ```Score_table_241126_ImageNet16-120.csv```

- **MultiOpt_Large_noband**  
  Bayesian optimization with RBFleX-NAS.  
  Candidate DNNs are evalauted by RBFleX-NAS   
  No ranking table

- **MultiOpt_Large_Acc_noband**  
  Bayesian optimization.  
  Candidate DNNs are evalauted by training  
  No ranking table

- **RandomSearch_Large_Acc_noband**  
  Random Search.  
  Candidate DNNs are evalauted by training  
  No ranking table

- **NSGA_Acc_noband**  
  NSGA-II.  
  Candidate DNNs are evalauted by training  
  No ranking table
  
### Natural Language Processing (NLP)

Dataset: Penn Treebank (PTB)  
DNN design space: NAS-Bench-NLP    
Hardware design space (simulator): Original based on MAESTRO Simulator  

- **H-RBFleX_Large_noband_NLP**  
  H-RBFleX (Proposed method)  
  RBFleX-NAS creates a ranking table before the Bayesian optimization.  
  Ranking table: ```RBF_Score_table_241126_PTB.csv```

- **H-WOT_Large_noband_NLP**  
  Bayesian optimization with a ranking table made by NASWOT.  
  NASWOT creates a ranking table before the Bayesian optimization.  
  Ranking table: ```WOT_Score_table_241126_PTB.csv```

- **MultiOpt_Large_noband_NLP**  
  Bayesian optimization with RBFleX-NAS.  
  Candidate DNNs are evalauted by RBFleX-NAS   
  No ranking table

- **MultiOpt_Large_Acc_noband_NLP**  
  Bayesian optimization.  
  Candidate DNNs are evalauted by training  
  No ranking table

- **RandomSearch_Large_Acc_noband_NLP**  
  Random Search.  
  Candidate DNNs are evalauted by training  
  No ranking table

- **NSGA_Acc_noband_NLP**  
  NSGA-II.  
  Candidate DNNs are evalauted by training  
  No ranking table
    

### Semantic Segmentation (SS)

Dataset: Tasknomy  
DNN design space: NAS-Bench-Trans101 (Semantic Segmentation)  
Hardware design space (simulator): Scale-Sim  

- **H-RBFleX_Large_noband_trans101**  
  H-RBFleX (Proposed method)  
  RBFleX-NAS creates a ranking table before the Bayesian optimization.  
  Ranking table: ```RBF_Score_table_241126_semantic.csv```

- **H-WOT_Large_noband_trans101**  
  Bayesian optimization with a ranking table made by NASWOT.  
  NASWOT creates a ranking table before the Bayesian optimization.  
  Ranking table: ```WOT_Score_table_241126_semantic.csv```

- **MultiOpt_Large_noband_trans101**  
  Bayesian optimization with RBFleX-NAS.  
  Candidate DNNs are evalauted by RBFleX-NAS   
  No ranking table

- **MultiOpt_Large_Acc_noband_trans101**  
  Bayesian optimization.  
  Candidate DNNs are evalauted by training  
  No ranking table

- **RandomSearch_Large_Acc_noband_NLP**  
  Random Search.  
  Candidate DNNs are evalauted by training  
  No ranking table

- **NSGA_Acc_noband_NLP**  
  NSGA-II.  
  Candidate DNNs are evalauted by training  
  No ranking table


## How to Run
Move into each folder that you want to run... 
### Image Classification
```python
python run_sss.py
```
### Natural Language Processing (NLP)
```python
python run_NLP.py
```
### Semantic Segmentation (SS)
```python
python run_trans101.py
```

Each script accepts the following optional parameters that control the optimization process:

| Parameter              | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| `optimized_components` | Specifies which components to optimize         (e.g. optimized_components = {"X1":12, "X2":14, "X3":0, "X4":0}, 0 means the target of optimization)          |
| `n_hyper`              | Number of networks for hyperparameter detection algoerithm for RBFleX-NAS.                               |
| `ref_score`            | Reference score for multi-objective optimization (used for hypervolume calculation).            |
| `acqu_algo`            | Acquisition function for Bayesian optimization (e.g., `qNParEGO`, `qEI`).                       |
| `iters`                | Number of optimization iterations to perform.                                                    |
| `n_init_size`          | Number of initial random samples used to bootstrap the optimization.                            |
| `batch_size`           | Number of configurations evaluated in each batch.                                                |
| `Hardware_Arch`        | Target hardware simulator (e.g., `"ScaleSim"`, `"LAXOR"`, `MAESTRO`).                                     |
| `dataset`              | Dataset used for training/evaluation (e.g., `"ImageNet"`, `"CIFAR-10"`, `""`).                      |
| `benchmark_root`       | Path to the benchmark design space directory (e.g., NATS-Bench).  


## How to Check Results
All results are collected into the `Result` folder on each optimization folder.

## How to Preproduce a Figures
#### Install Requirements
```bash
pip install matplotlib numpy pandas seaborn
```
### Reproducing `Fig.3`

You can generate the figure by running the provided script `modality.py`.

```python
python ./Fig3/modality.py
```

### Reproducing `Fig.5`

You can generate the figure by running the provided script `modality_semantic.py`.

```python
python ./Fig5/modality_semantic.py
```


### Reproducing `Fig.7`

You can generate the figure by running the provided script `plot_regret.py`.

```python
python ./Fig7/plot_regret.py
```


### Reproducing `Fig.8`, `Fig.9`, `Fig.10`, `Fig.11`, `Fig.13`, `Fig.14`, `Fig.17`, `Fig.19`

```python
python ./Fig8-9/plot_hypervolume.py
python ./Fig8-9/plot_pareto.py
```

