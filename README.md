## Learning Uncertainty from Internal Dispersion

This repository contains the official implementation of **ACL 2026 main paper** ["Learning Uncertainty from Internal Dispersion"]().

We present *Sequential Internal Variance Representation (SIVR)*, an uncertainty estimation framework that learns patterns indicative of factual error from the internal dispersion or variance of hidden states across layers.
SIVR enjoys **strong generalisation** and is **label-efficient.**

<figure>
  <img height="750" alt="sivr" src="https://github.com/user-attachments/assets/64ab3a68-46e7-4d68-91e2-956b84644be9" />
  <figcaption align="center">SIVR pipeline: (i). At each generated token, we extract LLM hidden states, and compute internal variance; (ii). We utilise these as informative features for sequence classification of response correctness with a simple transformer encoder architecture.</figcaption>
</figure>

### Get Started

0. Create a virtual environment and install dependencies.
```
conda create -n sivr python=3.13
conda activate sivr
pip install -r requirements.txt
```

1. Generate responses, and extract features. The `eval.py` script handles generation, feature extraction, and evaluation of uncertainty estimation baselines.
```
python eval.py
```

2. Train sequence classifier.
```
python train.py
```

The script `scripts/main.sh` can be used to reproduce experiments in the paper for Llama-3.1-8B-Instruct.

### Citation

If you find this work useful, please cite:
```
@inproceedings{srey2026uncertainty,
  title = {Learning Uncertainty from Internal Dispersion},
  author = {Srey, Ponhvoan and Wu, Xiaobao and Nguyen, Cong-Duy T and Luu, Anh Tuan},
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)},
  year = {2026}
}
```
