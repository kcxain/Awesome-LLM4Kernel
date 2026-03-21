# <div align="center">Awesome-LLM4Kernel</div>

<div align="center">

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-74-green.svg)](https://github.com/kcxain/Awesome-LLM4Kernel)
[![Last Commit](https://img.shields.io/github/last-commit/kcxain/Awesome-LLM4Kernel)](https://github.com/kcxain/Awesome-LLM4Kernel)
[![Website](https://img.shields.io/badge/Website-Live-orange)](https://kechang.xin/Awesome-LLM4Kernel/)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

</div>


GPU kernels are central to modern compute stacks and directly determine training and inference efficiency. Kernel development is difficult because it requires hardware expertise and iterative refinement with multi step tool feedback. Since Stanford released KernelBench in February 2025, the LLM4Kernel field has grown rapidly, with increasing interest in using large language models to support or automate kernel generation, optimization, and verification.

This project provides a continuous and comprehensive survey of the field, covering both benchmarks and methods. On the methodological side, we categorize existing work into four major directions:

 - Search-based piplines
 - Agent-based pipelines
 - Domain-specific Models
 - Agentic RL

We include all relevant top conference papers, arXiv preprints, open source projects, technical reports, and blogs, aiming to build the most complete resource hub for LLM4Kernel research.

Online page: https://kechang.xin/Awesome-LLM4Kernel/

## 📖 Benchmarks

- **SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits** [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.19173) [![Code](https://img.shields.io/github/stars/NVIDIA/SOL-ExecBench)](https://github.com/NVIDIA/SOL-ExecBench)  
	- Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang, Zhifan Ye, Nestor Qin, Fengzhe Zhou, Yuan Zhang, Jingquan Wang, Sana Damani, Dheeraj Peri, Ouye Xie, Aditya Kane, Moshe Maor, Michael Behar, Triston Cao, Rishabh Mehta, Vartika Singh, Vikram Sharma Mailthody, Terry Chen, Zihao Ye, Hanfeng Chen, Tianqi Chen, Vinod Grover, Wei Chen, Wei Liu, Eric Chung, Luis Ceze, Roger Bringmann, Cyril Zeller, Michael Lightstone, Christos Kozyrakis, Humphrey Shi
	- **Institution:** NVIDIA
	- **Task:** CUDA Kernel Optimization Benchmarking

- **FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems** [![Paper](https://img.shields.io/badge/arXiv-26.01-red)](https://arxiv.org/abs/2601.00227v1) [![Code](https://img.shields.io/github/stars/flashinfer-ai/flashinfer-bench)](https://github.com/flashinfer-ai/flashinfer-bench)  
	- Shanli Xing, Yiyan Zhai, Alexander Jiang, Yixin Dong, Yong Wu, Zihao Ye, Charlie Ruan, Yingyi Huang, Yineng Zhang, Liangsheng Yin, Aksara Bayyapu, Luis Ceze, Tianqi Chen
	- **Institution:** University of Washington, Carnegie Mellon University, NVIDIA
	- **Task:** CUDA/Triton Optimization

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

- **RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis** [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.11506)  
	- Zhen Bi, Qian Fan, Renjie Liu, Xing Di, Zihao Zhu, Borui Wang, Yiru Chen, Xiaoyi Dong, Rui Liu, Cheng Tan, Nian Liu, Xuhui Fan, Mark Shirman, Gal Oren, Anton Fonin, Konstantinos Parasyris, Yusong Gao, Song Han
	- **Institution:** Huzhou University, Banbu AI Foundation, Chinese Academy of Sciences, Carnegie Mellon University, University of Edinburgh
	- **Task:** On-device LLM -> Roofline

- **Can Large Language Models Predict Parallel Code Performance** [![Paper](https://img.shields.io/badge/Conference-25-green)](https://dl.acm.org/doi/abs/10.1145/3731545.3743645) [![Code](https://img.shields.io/github/stars/Scientific-Computing-Lab/ParallelCodeEstimation)](https://github.com/Scientific-Computing-Lab/ParallelCodeEstimation)  
	- Gregory Bolet, Giorgis Georgakoudis, Harshitha Menon, Konstantinos Parasyris, Niranjan Hasabnis, Hayden Estes, Kirk W. Cameron, Gal Oren
	- **Institution:** Virginia Tech, LLNL, Code Metal, Stanford University, Technion
	- **Task:** CUDA | OpenMP -> Roofline Class

- **NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/abs/2507.14403) [![Code](https://img.shields.io/github/stars/AMDResearch/NPUEval)](https://github.com/AMDResearch/NPUEval)  
	- Sarunas Kalade, Graham Schelle
	- **Institution:** Advanced Micro Devices
	- **Task:** NL -> NPU Kernel

- **ConCuR: Conciseness Makes State-of-the-Art Kernel Generation** [![Paper](https://img.shields.io/badge/arXiv-25.10-red)](https://arxiv.org/abs/2510.07356) [![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/lkongam/KernelCoder)  
	- **Task:** Torch -> CUDA

- **CUDABench: Benchmarking LLMs for Text-to-CUDA Generation** [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.02236) [![Code](https://img.shields.io/github/stars/CUDA-Bench/CUDABench)](https://github.com/CUDA-Bench/CUDABench)  
	- Jiace Zhu, Wentao Chen, Qi Fan, Zhixing Ren, Junying Wu, Xing Zhe Chai, Chotiwit Rungrueangwutthinon, Yehan Ma, An Zou
	- **Institution:** Shanghai Jiao Tong University
	- **Task:** NL -> CUDA

- **KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on Emerging Hardware** [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.08721)  
	- Jiayi Nie, Haoran Wu, Yao Lai, Zeyu Cao, Cheng Zhang, Binglei Lou, Erwei Wang, Jianyi Cheng, Timothy M. Jones, Robert Mullins, Rika Antonova, Yiren Zhao
	- **Task:** NL -> Accelerator Kernel

- **KernelBook: PyTorch to Triton Code Translation Dataset** [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/GPUMODE/KernelBook)  
	- Sahan Paliskara, Mark Saroufim
	- **Institution:** GPUMODE
	- **Task:** Torch -> Triton

## 🔧 Method

### Search-based piplines

- **KernelFoundry: Hardware-aware evolutionary GPU kernel optimization**  
  [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.12440)  
	- Nina Wiedemann, Quentin Leboutet, Michael Paulitsch, Diana Wofk, Benjamin Ummenhofer  
	- **Institution:** University of Freiburg, Infineon Technologies AG  
	- **Task:** CUDA | SYCL Kernel Optimization

- **OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.12305)  
	- Arijit Bhattacharjee, Heng Ping, Son Vu Le, Paul Bogdan, Nesreen K. Ahmed, Ali Jannesari  
	- **Institution:** Iowa State University  
	- **Task:** NL -> CUDA + MCTS Optimization

- **K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.19128) [![Code](https://img.shields.io/github/stars/caoshiyi/K-Search)](https://github.com/caoshiyi/K-Search)
	- Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, Ion Stoica  
	- **Institution:** UC Berkeley  
	- **Task:** CUDA Optimization

- **KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit** [![Paper](https://img.shields.io/badge/aiXiv-25.11-red)](https://arxiv.org/pdf/2511.18868)
	- Dezhi Ran, Shuxiao Xie, Mingfang Ji, Ziyue Hua, Mengzhou Wu, Yuan Cao, Yuzhe Guo, Yu Hao, Linyi Li, Yitao Hu, Tao Xie
	- **Institution:** Peking University
	- **Task:** Torch -> Triton

- **Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling** [![Blog](https://img.shields.io/badge/Blog-NVIDIA-blue)](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)  
	- Terry Chen, Bing Xu, Kirthi Devleker
	- **Institution:** NVIDIA
	- **Task:** NL -> CUDA Attention Kernel

- **Tutoring LLM into a Better CUDA Optimizer** [![Paper](https://img.shields.io/badge/Euro--Par-25-green)](https://link.springer.com/chapter/10.1007/978-3-031-99857-7_18) [![Code](https://img.shields.io/github/stars/matyas-brabec/2025-europar-llm)](https://github.com/matyas-brabec/2025-europar-llm)  
	- Matyas Brabec, Jiri Klepl, Michal Topfer, Martin Krulis
	- **Institution:** Charles University
	- **Task:** NL -> CUDA

- **GPU Performance Portability needs Autotuning** [![Paper](https://img.shields.io/badge/arXiv-25.05-red)](https://arxiv.org/abs/2505.03780)  
	- Burkhard Ringlein, Thomas Parnell, Radu Stoica
	- **Institution:** IBM Research Europe
	- **Task:** Triton Attention -> Cross-GPU Optimization

- **TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.12-red)](https://arxiv.org/abs/2512.09196) [![Code](https://img.shields.io/github/stars/RLsys-Foundation/TritonForge)](https://github.com/RLsys-Foundation/TritonForge)  
	- Haonan Li, Keyu Man, Partha Kanuparthy, Hanning Chen, Wei Sun, Sreen Tallam, Chenguang Zhu, Kevin Zhu, Zhiyun Qian
	- **Task:** Triton Optimization

- **EVOENGINEER: Mastering Automated CUDA Kernel Code Evolution with Large Language Models** [![Paper](https://img.shields.io/badge/arXiv-25.10-red)](https://arxiv.org/abs/2510.03760)  
	- Ping Guo, Chenyu Zhu, Siyuan Chen, Fei Liu, Xi Lin, Zhichao Lu, Qingfu Zhang
	- **Institution:** City University of Hong Kong
	- **Task:** CUDA Optimization

- **From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph** [![Paper](https://img.shields.io/badge/arXiv-25.10-red)](https://arxiv.org/abs/2510.19873) [![Code](https://img.shields.io/github/stars/blacknickwield/ReGraphT)](https://github.com/blacknickwield/ReGraphT)  
	- Junfeng Gong, Zhiyi Wei, Junying Chen, Cheng Liu, Huawei Li
	- **Institution:** Institute of Computing Technology, Chinese Academy of Sciences, University of Chinese Academy of Sciences, South China University of Technology
	- **Task:** Sequential Code -> CUDA

- **MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization** [![Paper](https://img.shields.io/badge/arXiv-26.01-red)](https://arxiv.org/abs/2601.05475)  
	- Jiefu Ou, Sapana Chaudhary, Kaj Bostrom, Nathaniel Weir, Shuai Zhang, Huzefa Rangwala, George Karypis
	- **Institution:** Johns Hopkins University, Amazon Web Services
	- **Task:** CUDA | C++ Optimization

### Agent-based pipelines

- **StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning** [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.02637)  
	- Shiyang Li, Zijian Zhang, Winson Chen, Yuebo Luo, Mingyi Hong, Caiwen Ding
	- **Institution:** University of Minnesota, Twin Cities
	- **Task:** Torch -> End-to-End CUDA

- **KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization**  
  [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.10085) [![Code](https://img.shields.io/github/stars/0satan0/KernelMem)](https://github.com/0satan0/KernelMem)  
	- Qitong Sun, Jun Han, Tianlin Li, Zhe Tang, Sheng Chen, Fei Yang, Aishan Liu, Xianglong Liu, Yang Liu  
	- **Institution:** Beihang University, Beijing Academy of Artificial Intelligence  
	- **Task:** Torch -> CUDA

- **Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts**  
  [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.07169)  
	- Yuxuan Han, Meng-Hao Guo, Zhengning Liu, Wenguang Chen, Shi-Min Hu  
	- **Institution:** Tsinghua University  
	- **Task:** CUDA Optimization

- **CUCo: An Agentic Framework for Compute and Communication Co-design**  
  [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.02376)  
	- Bodun Hu, Yoga Sri Varshan V, Saurabh Agarwal, Aditya Akella  
	- **Institution:** University of Wisconsin-Madison  
	- **Task:** CUDA Compute <-> Communication Co-design

- **AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation**  
  [![Paper](https://img.shields.io/badge/arXiv-26.01-red)](https://arxiv.org/abs/2601.22760)  
	- Zhongzhen Wen, Shudi Shao, Zhong Li, Yu Ge, Tongtong Xu, Yuanyi Lin, Tian Zhang  
	- **Institution:** Nanjing University  
	- **Task:** Torch | NL -> AscendC

- **KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.14293)  
	- Kris Shengjun Dong, Sahil Modi, Dima Nikiforov, Sana Damani, Edward Lin, Siva Kumar Sastry Hari, Christos Kozyrakis  
	- **Institution:** NVIDIA, UC Berkeley  
	- **Task:** CUDA Optimization

- **AKG kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis** [![Paper](https://img.shields.io/badge/aiXiv-25.12-red)](https://arxiv.org/pdf/2512.23424v1) [![Code](https://img.shields.io/github/stars/mindspore-ai/akg)](https://github.com/mindspore-ai/akg/blob/master/aikg/README_CN.md)  
	- Jinye Du, Quan Yuan, Zuyao Zhang, Yanzhi Yi, Jiahui Hu, Wangyi Chen, Yiyang Zhu, Qishui Zheng, Wenxiang Zou, Xiangyu Chang, Zuohe Zheng, Zichun Ye, Chao Liu, Shanni Li, Renwei Zhang, Yiping Deng, Xinwei Hu, Xuefeng Jin, Jie Zhao
	- **Institution:** Huawei
	- **Task:** Torch -> CUDA | Triton | Tilelang | AscendC

- **TritorX: Agentic Operator Generation for ML ASICs** [![Paper](https://img.shields.io/badge/aiXiv-25.12-red)](https://www.arxiv.org/abs/2512.10977)
	- Alec M. Hammond, Aram Markosyan, Aman Dontula, Simon Mahns, Zacharias Fisches, Dmitrii Pedchenko, Keyur Muzumdar, Natacha Supper, Mark Saroufim, Joe Isaacson, Laura Wang, Warren Hunt, Kaustubh Gondkar, Roman Levenstein, Gabriel Synnaeve, Richard Li, Jacob Kahn, Ajit Mathews
	- **Institution:** Meta
	- **Task:** torch ATen Docstring -> Triton

- **KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents** [![Paper](https://img.shields.io/badge/blog-25.11-blue)](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) [![Code](https://img.shields.io/github/stars/meta-pytorch/KernelAgent)](https://github.com/meta-pytorch/KernelAgent)  
	- Laura Wang
	- **Institution:** PyTorch Team at Meta
	- **Task:** Torch -> Triton

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

- **GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.06-red)](https://arxiv.org/abs/2506.20807)  
	- Martin Andrews, Sam Witteveen
	- **Task:** CUDA Optimization

- **Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/abs/2507.23194) [![Code](https://img.shields.io/github/stars/AMD-AIG-AIMA/GEAK-agent)](https://github.com/AMD-AIG-AIMA/GEAK-agent) [![Eval](https://img.shields.io/github/stars/AMD-AIG-AIMA/GEAK-eval)](https://github.com/AMD-AIG-AIMA/GEAK-eval)  
	- Jianghui Wang, Vinay Joshi, Saptarshi Majumder, Xu Chao, Bin Ding, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum
	- **Task:** NL -> Triton

- **How Many Agents Does it Take to Beat PyTorch? (surprisingly not that much)** [![Blog](https://img.shields.io/badge/Blog-Lossfunk-blue)](https://letters.lossfunk.com/p/how-many-agents-does-it-take-to-beat)  
	- Shikhar Mishra, Ayush Nangia
	- **Task:** Torch -> CUDA

- **Astra: A Multi-Agent System for GPU Kernel Performance Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.09-red)](https://arxiv.org/abs/2509.07506) [![Code](https://img.shields.io/github/stars/Anjiang-Wei/Astra)](https://github.com/Anjiang-Wei/Astra)  
	- Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken
	- **Task:** CUDA Optimization

- **CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.11-red)](https://arxiv.org/abs/2511.01884) [![Code](https://img.shields.io/github/stars/OptimAI-Lab/CudaForge)](https://github.com/OptimAI-Lab/CudaForge)  
	- Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding
	- **Institution:** University of Minnesota, Twin Cities
	- **Task:** Torch -> CUDA

- **KForge: Program Synthesis for Diverse AI Hardware Accelerators** [![Paper](https://img.shields.io/badge/arXiv-25.11-red)](https://arxiv.org/abs/2511.13274)  
	- **Task:** NL -> Accelerator Kernel

- **The AI CUDA engineer: Agentic CUDA kernel discovery, optimization and composition** [![Report](https://img.shields.io/badge/Report-Sakana%20AI-blue)](https://pub.sakana.ai/static/paper.pdf)  
	- **Task:** Torch -> CUDA

- **Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems** [![Paper](https://img.shields.io/badge/arXiv-25.11-red)](https://arxiv.org/abs/2511.16964)  
	- **Task:** PyTorch Inference -> CUDA

- **PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.11-red)](https://arxiv.org/abs/2511.06345)  
	- **Task:** CUDA Optimization

- **cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution** [![Paper](https://img.shields.io/badge/arXiv-25.12-red)](https://arxiv.org/abs/2512.16465) [![Code](https://img.shields.io/github/stars/champloo2878/cuPilot-Kernels)](https://github.com/champloo2878/cuPilot-Kernels)  
	- **Task:** CUDA Optimization

- **AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.11-red)](https://arxiv.org/abs/2511.15915) [![Code](https://img.shields.io/github/stars/zhang677/AccelOpt)](https://github.com/zhang677/AccelOpt)  
	- Genghan Zhang, Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, Kunle Olukotun
	- **Institution:** Stanford University, Amazon Web Services, University of Toronto
	- **Task:** NKI -> Trainium Kernel Optimization

- **Adaptive Self-improvement LLM Agentic System for ML Library Development** [![Paper](https://img.shields.io/badge/ICML-25-green)](https://arxiv.org/abs/2502.02534) [![Code](https://img.shields.io/github/stars/zhang677/PCL-liteLLM)](https://github.com/zhang677/PCL-liteLLM)  
	- Genghan Zhang, Weixin Liang, Olivia Hsu, Kunle Olukotun
	- **Institution:** Stanford University
	- **Task:** NL -> ASPL ML Library

### Domain-specific Models

- **InCoder-32B: Code Foundation Model for Industrial Scenarios** [![Paper](https://img.shields.io/badge/arXiv-26.03-red)](https://arxiv.org/abs/2603.16790)  
	- Jian Yang, Wei Zhang, Jiajun Wu, Junhang Cheng, Shawn Guo, Haowen Wang, Weicheng Gu, Yaxin Du, Joseph Li, Fanglin Xu, Yizhi Li, Lin Jing, Yuanbo Wang, Yuhan Gao, Ruihao Gong, Chuan Hao, Ran Tao, Aishan Liu, Tuney Zheng, Ganqu Cui, Zhoujun Li, Mingjie Tang, Chenghua Lin, Wayne Xin Zhao, Xianglong Liu, Ming Zhou, Bryan Dai, Weifeng Lv
	- **Institution:** Beihang University, iQuest Research, Shanghai Jiao Tong University, ELLIS, University of Manchester, Shanghai Artificial Intelligence Laboratory, Sichuan University, Renmin University of China, Langboat
	- **Task:** Code -> GPU Kernel Optimization

- **DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.11715)  
	- Haolei Bai, Lingcheng Kong, Xueyi Chen, Jianmian Wang, Zhiqiang Tao, Huan Wang  
	- **Institution:** Westlake University  
	- **Task:** Torch -> CUDA

- **AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/abs/2507.05687) [![Code](https://img.shields.io/github/stars/AI9Stars/AutoTriton)](https://github.com/AI9Stars/AutoTriton)  
	- Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, Maosong Sun
	- **Institution:** Tsinghua University
	- **Task:** Torch -> Triton

- **QiMeng-MuPa: Mutual-Supervised Learning for Sequential-to-Parallel Code Translation** [![Paper](https://img.shields.io/badge/NeurIPS-25-green)](https://arxiv.org/pdf/2506.11153) [![Code](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-MuPa)](https://github.com/QiMeng-IPRC/QiMeng-MuPa)  
	- Changxin Ke, Rui Zhang, Shuo Wang, Li Ding, Guangli Li, Yuanbo Wen, Shuoming Zhang, Ruiyuan Xu, Jin Qin, Jiaming Guo, Chenxi Wang, Ling Li, Qi Guo, Yunji Chen
	- **Institution:** Institute of Computing Technology
	- **Task:** C -> CUDA

 - **AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units** [![Paper](https://img.shields.io/badge/arXiv-26.01-red)](https://arxiv.org/abs/2601.07160)
 	- Xinzi Cao, Jianyang Zhai, Pengfei Li, Zhiheng Hu, Cen Yan, Bingxu Mu, Guanghuan Fang, Bin She, Jiayu Li, Yihan Su, Dongyang Tao, Xiansong Huang, Fan Xu, Feidiao Yang, Yao Lu, Chang-Dong Wang, Yutong Lu, Weicheng Xue, Bin Zhou, Yonghong Tian
	- **Institution:** Pengcheng Laboratory, HUAWEI, Sun Yat-sen University
	- **Task:** Torch -> AscendC

- **CUDA-LLM: LLMs Can Write Efficient CUDA Kernels** [![Paper](https://img.shields.io/badge/arXiv-25.06-red)](https://arxiv.org/abs/2506.09092)  
	- Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou
	- **Institution:** Shanghai Jiao Tong University
	- **Task:** NL -> CUDA

- **KernelLLM** [![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/facebook/KernelLLM)  
	- **Institution:** Meta
	- **Task:** Torch -> Triton

- **Scaling LLM Test-Time Compute with Mobile NPU on Smartphones** [![Paper](https://img.shields.io/badge/arXiv-25.09-red)](https://arxiv.org/abs/2509.23324) [![Code](https://img.shields.io/github/stars/haozixu/llama.cpp-npu)](https://github.com/haozixu/llama.cpp-npu) [![Library](https://img.shields.io/github/stars/haozixu/htp-ops-lib)](https://github.com/haozixu/htp-ops-lib)  
	- **Task:** LLM Inference -> Mobile NPU

- **CudaLLM: Training Language Models to Generate High-Performance CUDA Kernels** [![Code](https://img.shields.io/github/stars/ByteDance-Seed/cudaLLM)](https://github.com/ByteDance-Seed/cudaLLM) [![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/ByteDance-Seed/cudaLLM-8B)  
	- **Institution:** ByteDance Seed
	- **Task:** NL -> CUDA

- **Omniwise: Predicting GPU Kernels Performance with LLMs** [![Paper](https://img.shields.io/badge/arXiv-25.06-red)](https://arxiv.org/abs/2506.20886)  
	- Zixian Wang, Cole Ramos, Muhammad A. Awad, Keith Lowery
	- **Institution:** University of Illinois Urbana-Champaign, AMD
	- **Task:** CUDA -> Performance Metrics

### Agentic RL

- **CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.24286) [![Code](https://img.shields.io/github/stars/BytedTsinghua-SIA/CUDA-Agent)](https://github.com/BytedTsinghua-SIA/CUDA-Agent)  
	- Weinan Dai, Hanlin Wu, Qiying Yu, Huan-ang Gao, Jiahao Li, Chengquan Jiang, Weiqiang Lou, Yufan Song, Hongli Yu, Jiaze Chen, Wei-Ying Ma, Ya-Qin Zhang, Jingjing Liu, Mingxuan Wang, Xin Liu, Hao Zhou
	- **Institution:** ByteDance Seed, Tsinghua AIR
	- **Task:** Torch -> CUDA

- **Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.05885) [![Code](https://img.shields.io/github/stars/hkust-nlp/KernelGYM)](https://github.com/hkust-nlp/KernelGYM)  
	- Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, Junxian He  
	- **Institution:** HKUST, TikTok  
	- **Task:** Torch -> Triton

- **Fine-Tuning GPT-5 for GPU Kernel Generation**  
  [![Paper](https://img.shields.io/badge/arXiv-26.02-red)](https://arxiv.org/abs/2602.11000)  
	- Ali Tehrani, Yahya Emara, Essam Wissam, Wojciech Paluch, Waleed Atallah, Łukasz Dudziak, Mohamed S. Abdelfattah  
	- **Institution:** Makora  
	- **Task:** Torch -> CUDA

- **QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation** [![Paper](https://img.shields.io/badge/AAAI-26-green)](https://arxiv.org/abs/2511.20100) [![Code](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-Kernel)](https://github.com/QiMeng-IPRC/QiMeng-Kernel)  
	- Xinguo Zhu, Shaohui Peng, Jiaming Guo, Yunji Chen, Qi Guo, Yuanbo Wen, Hang Qin, Ruizhi Chen, Qirui Zhou, Ke Gao, Yanjun Wu, Chen Zhao, Ling Li
	- **Institution:** Institute of Software, Institute of Computing Technology
	- **Task:** Torch -> Triton

- **CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning** [![Paper](https://img.shields.io/badge/arXiv-25.07-red)](https://arxiv.org/abs/2507.14111) [![Code](https://img.shields.io/github/stars/deepreinforce-ai/CUDA-L1)](https://github.com/deepreinforce-ai/CUDA-L1) [![Project](https://img.shields.io/badge/Project-Page-blue)](https://deepreinforce-ai.github.io/cudal1_blog/)  
	- Xiaoya Li, Xiaofei Sun, Albert Wang, Jiwei Li, Chris Shum
	- **Task:** CUDA Optimization

- **TRITONRL: Training LLMs to Think and Code Triton Without Cheating** [![Paper](https://img.shields.io/badge/arXiv-25.10-red)](https://arxiv.org/abs/2510.17891)  
	- Jiin Woo, Shaowei Zhu, Allen Nie, Zhen Jia, Yida Wang, Youngsuk Park
	- **Task:** Torch -> Triton

- **CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning** [![Paper](https://img.shields.io/badge/CGO-25-green)](https://dl.acm.org/doi/abs/10.1145/3696443.3708943)  
	- Guoliang He, Eiko Yoneki
	- **Institution:** University of Cambridge
	- **Task:** SASS Scheduling Optimization

- **Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning** [![Paper](https://img.shields.io/badge/OpenReview-25-green)](https://openreview.net/forum?id=VdLEaGPYWT)  
	- Yaoyu Wang, Hankun Dai, Zhidong Yang, Junmin Xiao, Guangming Tan
	- **Task:** Sparse Matrix -> CUDA

- **SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.08-red)](https://arxiv.org/abs/2508.20258)  
	- Arya Tschand, Muhammad Awad, Ryan Swann, Kesavan Ramakrishnan, Jeffrey Ma, Keith Lowery, Ganesh Dasika, Vijay Janapa Reddi
	- **Task:** CUDA Swizzling Optimization

- **Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization** [![Paper](https://img.shields.io/badge/arXiv-25.10-red)](https://arxiv.org/abs/2510.17158)  
	- Daniel Nichols, Konstantinos Parasyris, Charles Jekel, Abhinav Bhatele, Harshitha Menon
	- **Task:** CUDA Optimization

- **CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning** [![Paper](https://img.shields.io/badge/arXiv-25.12-red)](https://arxiv.org/abs/2512.02551) [![Code](https://img.shields.io/github/stars/deepreinforce-ai/CUDA-L2)](https://github.com/deepreinforce-ai/CUDA-L2)  
	- Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum
	- **Institution:** DeepReinforce Team
	- **Task:** HGEMM Optimization

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
