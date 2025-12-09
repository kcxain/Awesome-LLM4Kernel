# <div align="center">Awesome-LLM4Kernel</div>

<div align="center">

[![Last Commit](https://img.shields.io/github/last-commit/kcxain/Awesome-LLM4Kernel)](https://github.com/kcxain/Awesome-LLM4Kernel)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

</div>


GPU kernels are central to modern compute stacks and directly determine training and inference efficiency. Kernel development is difficult because it requires hardware expertise and iterative refinement with multi step tool feedback. Since Stanford released KernelBench in February 2025, the LLM4Kernel field has grown rapidly, with increasing interest in using large language models to support or automate kernel generation, optimization, and verification.

This project provides a continuous and comprehensive survey of the field, covering both benchmarks and methods. On the methodological side, we categorize existing work into three major directions:

 - Agent based pipelines
 - Domain specific Models
 - Agentic RL

We include all relevant top conference papers, arXiv preprints, open source projects, technical reports, and blogs, aiming to build the most complete resource hub for LLM4Kernel research.

## Benchmarks

- [ICML, 25.02] **KernelBench:** Can LLMs Write Efficient GPU Kernels?
	- Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini
	- **Institution:** Stanford University
	- [Paper](https://arxiv.org/pdf/2502.10517) | [Code](https://github.com/ScalingIntelligence/KernelBench)
	- **Task:** Torch -> CUDA
- [ACL findings, 25.02] **TritonBench:** Benchmarking Large Language Model Capabilities for Generating Triton Operators
	- Jianling Li, ShangZhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, WangHaojie WangHaojie, Jianrong Wang, Xu Han, Zhiyuan Liu, Maosong Sun
	- **Institution:** Tianjin University, Tsinghua University
	- [Paper](https://aclanthology.org/2025.findings-acl.1183.pdf) | [Code](https://github.com/thunlp/TritonBench)
	- **Task:** Torch | NL -> CUDA
- [25.04] **ComputeEval:** Evaluating Large Language Models for CUDA Code Generation
	- **Institution:** NVIDIA
	- [Code](https://github.com/NVIDIA/compute-eval)
	- **Task:** NL -> CUDA
- [25.04] **BackendBench:**  An Evaluation Suite for Testing How Well LLMs and Humans Can Write PyTorch Backends
	- **Institution:** Meta
	- [Blog](https://github.com/meta-pytorch/BackendBench/blob/main/docs/correctness.md) | [Code](https://github.com/meta-pytorch/BackendBench)
	- **Task:** Torch -> CUDA | Triton
- [25.07] **MultiKernelBench:** A Multi-Platform Benchmark for Kernel Generation
	- Zhongzhen Wen, Yinghui Zhang, Zhong Li, Zhongxin Liu, Linna Xie, Tian Zhang
	- **Institution:** Nanjing University
	- [Paper](https://arxiv.org/pdf/2507.17773) | [Code](https://github.com/wzzll123/MultiKernelBench)
	- **Task:** Torch -> CUDA | Pallas | AscendC
- [25.09] **robust-kbench:** Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization
	- Robert Tjarko Lange, Qi Sun, Aaditya Prasad, Maxence Faldor, Yujin Tang, David Ha
	- **Institution:** Sakana AI
	- [Paper](https://arxiv.org/pdf/2509.14279) | [Code](https://github.com/SakanaAI/robust-kbench)
	- **Task:** Torch -> CUDA


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