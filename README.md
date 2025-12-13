# <div align="center">Awesome-LLM4Kernel</div>

<div align="center">

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-15-green.svg)](https://github.com/kcxain/Awesome-LLM4Kernel)
[![Last Commit](https://img.shields.io/github/last-commit/kcxain/Awesome-LLM4Kernel)](https://github.com/kcxain/Awesome-LLM4Kernel)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

</div>


GPU kernels are central to modern compute stacks and directly determine training and inference efficiency. Kernel development is difficult because it requires hardware expertise and iterative refinement with multi step tool feedback. Since Stanford released KernelBench in February 2025, the LLM4Kernel field has grown rapidly, with increasing interest in using large language models to support or automate kernel generation, optimization, and verification.

This project provides a continuous and comprehensive survey of the field, covering both benchmarks and methods. On the methodological side, we categorize existing work into three major directions:

 - Agent based pipelines
 - Domain specific Models
 - Agentic RL

We include all relevant top conference papers, arXiv preprints, open source projects, technical reports, and blogs, aiming to build the most complete resource hub for LLM4Kernel research.

## 📖 Benchmarks

- **KernelBench: Can LLMs Write Efficient GPU Kernels?** [![Paper](https://img.shields.io/badge/ICML-25-green)](https://arxiv.org/pdf/2502.10517) [![Code](https://img.shields.io/github/stars/ScalingIntelligence/KernelBench)](https://github.com/ScalingIntelligence/KernelBench)  
	- Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini  
	- **Institution:** Stanford University  
	- **Task:** Torch -> CUDA  

- **TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators** [![Paper](https://img.shields.io/badge/ACL_findings-25-green)](https://aclanthology.org/2025.findings-acl.1183.pdf) [![Code](https://img.shields.io/github/stars/thunlp/TritonBench)](https://github.com/thunlp/TritonBench)  
	- Jianling Li, ShangZhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, WangHaojie WangHaojie, Jianrong Wang, Xu Han, Zhiyuan Liu, Maosong Sun  
	- **Institution:** Tianjin University, Tsinghua University  
	- **Task:** Torch | NL -> Triton  

- **ComputeEval: Evaluating Large Language Models for CUDA Code Generation** [![Code](https://img.shields.io/github/stars/NVIDIA/compute-eval)](https://github.com/NVIDIA/compute-eval)  
	- **Institution:** NVIDIA  
	- **Task:** NL -> CUDA  

- **BackendBench: An Evaluation Suite for Testing How Well LLMs and Humans Can Write PyTorch Backends** [![Blog](https://img.shields.io/badge/Blog-Meta-blue)](https://github.com/meta-pytorch/BackendBench/blob/main/docs/correctness.md) [![Code](https://img.shields.io/github/stars/meta-pytorch/BackendBench)](https://github.com/meta-pytorch/BackendBench)  
	- **Institution:** Meta  
	- **Task:** Torch -> CUDA | Triton  

- **MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/pdf/2507.17773) [![Code](https://img.shields.io/github/stars/wzzll123/MultiKernelBench)](https://github.com/wzzll123/MultiKernelBench)  
	- Zhongzhen Wen, Yinghui Zhang, Zhong Li, Zhongxin Liu, Linna Xie, Tian Zhang  
	- **Institution:** Nanjing University  
	- **Task:** Torch -> CUDA | Pallas | AscendC  

- **robust-kbench: Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.09-red)](https://arxiv.org/pdf/2509.14279) [![Code](https://img.shields.io/github/stars/SakanaAI/robust-kbench)](https://github.com/SakanaAI/robust-kbench)  
	- Robert Tjarko Lange, Qi Sun, Aaditya Prasad, Maxence Faldor, Yujin Tang, David Ha  
	- **Institution:** Sakana AI  
	- **Task:** Torch -> CUDA

- **gpuFLOPBench: Counting Without Running: Evaluating LLMs’ Reasoning About Code Complexity** [![Paper](https://img.shields.io/badge/arXiv-25.12-red)](https://arxiv.org/abs/2512.04355) [![Code](https://img.shields.io/github/stars/Scientific-Computing-Lab/gpuFLOPBench)](https://github.com/Scientific-Computing-Lab/gpuFLOPBench)  
	- Gregory Bolet, Giorgis Georgakoudis, Konstantinos Parasyris, Harshitha Menon, Niranjan Hasabnis, Kirk W. Cameron, Gal Oren
	- **Institution:** Stanford University  
	- **Task:** CUDA -> FLOPs  

## 🔧 Method

### Agent based pipelines

- **STARK: Strategic Team of Agents for Refining Kernels** [![Paper](https://img.shields.io/badge/arXiv-25.10-red)](https://arxiv.org/pdf/2510.16996)
	- Juncheng Dong, Yang Yang, Tao Liu, Yang Wang, Feng Qi, Vahid Tarokh, Kaushik Rangadurai, Shuang Yang
	- **Institution:** Meta Ranking AI Research
	- **Task:** Torch -> CUDA

- **QiMeng-Xpiler: Transcompiling Tensor Programs for Deep Learning Systems with a Neural-Symbolic Approach** [![Paper](https://img.shields.io/badge/OSDI-25-green)](https://arxiv.org/abs/2505.02146) [![Code](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-Xpiler)](https://github.com/QiMeng-IPRC/QiMeng-Xpiler)  
	- Shouyang Dong, Yuanbo Wen, Jun Bi, Di Huang, Jiaming Guo, Jianxing Xu, Ruibai Xu, Xinkai Song, Yifan Hao, Xuehai Zhou, Tianshi Chen, Qi Guo, Yunji Chen
	- **Institution:** University of Science and Technology of China, Cambricon Technologies, Institute of Computing Technology, Institute of Software
	- **Task:** CUDA <-> BangC <-> Hip <-> VNNI  

- **QiMeng-Attention: SOTA Attention Operator is generated by SOTA Attention Algorithm** [![Paper](https://img.shields.io/badge/ACL-25-green)](https://arxiv.org/abs/2506.12355) [![Code](https://img.shields.io/github/stars/chris-chow/QiMeng-Attention)](https://github.com/chris-chow/QiMeng-Attention)  
	- Qirui Zhou, Shaohui Peng, Weiqiang Xiong, Haixin Chen, Yuanbo Wen, Haochen Li, Ling Li, Qi Guo, Yongwei Zhao, Ke Gao, Ruizhi Chen, Yanjun Wu, Chen Zhao, Yunji Chen
	- **Institution:** Institute of Software, Institute of Computing Technology  
	- **Task:** NL -> CUDA (Attention)  

- **QiMeng-TensorOp: Automatically Generating High-Performance Tensor Operators with Hardware Primitives** [![Paper](https://img.shields.io/badge/IJCAI-25-green)](https://arxiv.org/pdf/2505.06302) [![Code](https://img.shields.io/github/stars/zhangxuzhi/QiMeng-TensorOp)](https://github.com/zhangxuzhi/QiMeng-TensorOp)  
	- Xuzhi Zhang, Shaohui Peng, Qirui Zhou, Yuanbo Wen, Qi Guo, Ruizhi Chen, Xinguo Zhu, Weiqiang Xiong, Haixin Chen, Congying Ma, Ke Gao, Chen Zhao, Yanjun Wu, Yunji Chen, Ling Li  
	- **Institution:** Institute of Computing Technology, Institute of Software 
	- **Task:** NL -> Hardware-specific Tensor Operators (RISC-V, ARM, GPU)

- **QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models** [![Paper](https://img.shields.io/badge/AAAI-25-green)](https://ojs.aaai.org/index.php/AAAI/article/view/34461) [![Code](https://img.shields.io/github/stars/chris-chow/QiMeng-GEMM)](https://github.com/chris-chow/QiMeng-GEMM)  
	- Qirui Zhou, Yuanbo Wen, Ruizhi Chen, Ke Gao, Weiqiang Xiong, Ling Li, Qi Guo, Yanjun Wu, Yunji Chen 
	- **Institution:** Institute of Computing Technology, Institute of Software
	- **Task:** NL -> CUDA (GEMM)

### Domain specific Models

- **AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/abs/2507.05687) [![Code](https://img.shields.io/github/stars/AI9Stars/AutoTriton)](https://github.com/AI9Stars/AutoTriton)  
	- Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, Maosong Sun
	- **Institution:** Tsinghua University
	- **Task:** Torch -> Triton

- **QiMeng-MuPa: Mutual-Supervised Learning for Sequential-to-Parallel Code Translation** [![Paper](https://img.shields.io/badge/NeurIPS-25-green)](https://arxiv.org/pdf/2506.11153) [![Code](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-MuPa)](https://github.com/QiMeng-IPRC/QiMeng-MuPa)  
	- Changxin Ke, Rui Zhang, Shuo Wang, Li Ding, Guangli Li, Yuanbo Wen, Shuoming Zhang, Ruiyuan Xu, Jin Qin, Jiaming Guo, Chenxi Wang, Ling Li, Qi Guo, Yunji Chen
	- **Institution:** Institute of Computing Technology
	- **Task:** C -> CUDA

### Agentic RL

- **QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation** [![Paper](https://img.shields.io/badge/AAAI-26-green)](https://arxiv.org/abs/2511.20100) [![Code](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-Kernel)](https://github.com/QiMeng-IPRC/QiMeng-Kernel)  
	- Xinguo Zhu, Shaohui Peng, Jiaming Guo, Yunji Chen, Qi Guo, Yuanbo Wen, Hang Qin, Ruizhi Chen, Qirui Zhou, Ke Gao, Yanjun Wu, Chen Zhao, Ling Li
	- **Institution:** Institute of Software, Institute of Computing Technology
	- **Task:** Torch -> Triton

- **Kevin: Multi-Turn RL for Generating CUDA Kernels** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/abs/2507.11948)
	- Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti
	- **Institution:** Stanford University
	- **Task:** Torch -> CUDA

## Contribution

Feel free to open an [issue](https://github.com/kcxain/Awesome-LLM4Kernel/issues/new) or submit a [pull request](https://github.com/kcxain/Awesome-LLM4Kernel/fork) to correct errors or add work that has not yet been included in this project. You can also email us at kcxain@gmail.com for any form of discussion and collaboration.


## Citation

If you find this work useful, welcome to cite us.

```bib
@article{llm4kernel,
  title={LLM4Kernel: A Survey of Large Language Models for GPU Kernel Development},
  author={Changxin Ke},
  year={2025}
  url={https://github.com/kcxain/Awesome-LLM4Kernel}
}
```
