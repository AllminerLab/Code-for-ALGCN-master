Environment:
  python = 3.6
  pandas = 1.1.5
  torch = 1.9.1
  numpy = 1.19.5
  tqdm = 4.36.0


Directory:

ALGCN_code
- config
  config file

- data
  ml-1m

- src
  - models
    - ALGCN.py: proposed model
    - MF.py: MF model
  - trainer
    - base_trainer.py: trainer module
  - util
    - spmm.py: sparse matrix multiplication function
  - metrics.py: some metric functions like recall, ndcg
  - data_generator.py: date generator module

- main.py: main file


Run the Codes:
python main.py

you can change hyperparameters by resetting the config file.

Reference:
Ronghai Xu, Haijun Zhao, Zhi-Yuan Li, and Chang-Dong Wang. "ALGCN: Accelerated Light Graph Convolution Network for Recommendation", DASFAA2023.
