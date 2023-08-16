# DCMRL
Decoupling Meta-Reinforcement Learning with Gaussian Task Contexts and Skills


## Requirements

- Python 3.6+
- Mujoco
- D4RL
- pip packages listed in requirements.txt



## Getting Started

1.requirements

2.Install extra package from code

```
cd DCMRL
pip install -e .
```

3. Enter source directory.

```
cd reproduce
```

4. Run & modify the scripts.

```
python DCMRL_meta_train.py --help
```



## SPiRL Pre-trained Model

We don't support SPiRL skill extraction in this repository.

We provide pre-trained SPiRL in  `reproduce/test/` by `--spirl-pretrained-path` or `-s`.

