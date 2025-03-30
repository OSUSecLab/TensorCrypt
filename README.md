# TensorCrypt: Repurposing Neural Networks for Efficient Cryptographic Computation
Implementation for NDSS'2025 paper: "TensorCrypt: Repurposing Neural Networks for Efficient Cryptographic Computation"

## Abstract

While neural networks (NNs) are traditionally associated with tasks such as image recognition and natural language processing, this paper presents a novel application of NNs for efficient cryptographic computations. Leveraging the Turing
completeness and inherent adaptability of NN models, we propose a transformative approach that efficiently accelerates cryptographic computations on various platforms. More specifically, with a program translation framework that converts traditional cryptographic algorithms into NN models, our proof-of-concept implementations in TensorFlow demonstrate substantial performance improvements: encryption speeds for AES, Chacha20, and Salsa20 show increases of up to 4.09×, 5.44×, and 5.06×, respectively, compared to existing GPU-based cryptographic solutions written by human experts. These enhancements are achieved without compromising the security of the original cryptographic algorithms, ensuring that our neural network-based approach maintains robust security standards. This repurposing of NNs opens new pathways for the development of scalable, efficient, and secure cryptographic systems that can adapt to the evolving demands of modern computing environments.

## Table of Content

### Directory Structure
`model`: model checkpoints of AES, Chacha, and Shasa ciphers.

`inference`: scripts to perform model inference for encryption and decryption.

`implementations`: implementations of NN DSLs in terms of ciphers.


## Installation & Requirements

### 1. GPU Setup

Install the [NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx) if you have not. You can use the following command to verify it is installed.

```
nvidia-smi
```

### 2. Create a virtual environment with pip

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Citation

If you find Tensorcrypt to be helpful for your research, please consider citing our paper:

```plaintext
@inproceedings{jin2025tensorcrypt,
  title={Repurposing Neural Networks for Efficient Cryptographic Computation},
  author={Jin, Xin and Ma, Shiqing and Lin, Zhiqiang},
  booktitle={Proceedings of the Network and Distributed System Security (NDSS) Symposium 2025},
  year={2025},
  address={San Diego, CA, USA},
  doi={10.14722/ndss.2025.240955}
}
```