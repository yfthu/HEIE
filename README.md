
<p align="center">
<a href="https://arxiv.org/abs/2411.17261">
    <strong>HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator</strong>
  </a>
  <br>
    <span>Fan Yang</a><sup></sup>,
    </span>
    <span>Ru Zhen<sup></sup>,</span>
    <span>Jianing Wang<sup></sup>,</span>
    <span>Yanhao Zhang<sup></sup>,</span>
    <span>Haoxiang Chen<sup></sup>,</span>
    <span>Haonan Lu</a><sup></sup>,</span>
    <span>Sicheng Zhao</a><sup></sup>,</span>
    <span>Guiguang Ding<sup></sup></span><br>
    <span><sup></sup>Tsinghua University, BNRist, OPPO AI Center, Peking University</span>


  <br>
</p>
<p align="center">
  <a href="https://yfthu.github.io/HEIE/">
    <img alt="proj" src="https://img.shields.io/badge/%F0%9F%96%BC%20Project-HEIE-blue?&style=flat&link=https://yfthu.github.io/HEIE">
  </a>
  <a href="https://github.com/yfthu/HEIE/tree/main/Expl-AIGI-Eval%20Dataset">
    <img alt="hf" src="https://img.shields.io/badge/%F0%9F%96%BC%20Dataset-orange?&style=flat&link=https://github.com/yfthu/HEIE/tree/main/Expl-AIGI-Eval%20Dataset">
  </a>
  <a href="https://arxiv.org/abs/2411.17261">
    <img alt="paper" src="https://img.shields.io/badge/Paper-2411.17261-brightblue?style=flat&logo=arxiv&link=https://arxiv.org/abs/2411.17261">
  </a>
</p>


### Expl-AIGI-Eval Dataset：https://github.com/yfthu/HEIE/tree/main/Expl-AIGI-Eval%20Dataset
For the RichHF dataset, we adopt the original train, dev, and test splits of the RichHF dataset. For each image in the train set, we annotate multiple chain of thought and explanations data, enhancing the diversity of the training set.


## 📝 Introduction 
🌐 AIGC images are prevalent across various fields, yet they frequently suffer from quality issues like artifacts and unnatural textures. 

🔄 Specialized models aim to predict defect region heatmaps but face two primary challenges: (1) lack of explainability, failing to provide reasons and analyses for subtle defects, and (2) inability to leverage common sense and logical reasoning, leading to poor generalization. 

Multimodal large language models (MLLMs) promise better comprehension and reasoning but face their own challenges: (1) difficulty in fine-grained defect localization due to the limitations in capturing tiny details; and (2) constraints in providing pixel-wise outputs necessary for precise heatmap generation. 

🧩 To address these challenges, we propose HEIE: a novel MLLM-Based Hierarchical Explainable image Implausibility Evaluator. We introduce the CoT-Driven Explainable Trinity Evaluator, which integrates heatmaps, scores, and explanation outputs, using CoT to decompose complex tasks into subtasks of increasing difficulty and enhance interpretability. 

Our Adaptive Hierarchical Implausibility Mapper synergizes low-level image features with high-level mapper tokens from LLMs, enabling precise local-to-global hierarchical heatmap predictions through an uncertainty-based adaptive token approach. 

📊 Moreover, we propose a new dataset: Expl-AIGI-Eval, designed to facilitate interpretable implausibility evaluation of AIGC images. 

👑Our method demonstrates state-of-the-art performance through extensive experiments. 



### Comparison with State-of-the-Art on the RichHF-18K Dataset

| Method                            | PLCC↑  | SRCC↑  | KLD↓   | CC↑    | SIM↑    | AUC-Judd↑ | MSE (GT=0)↓ | MSE (All Data)↓ |
|-----------------------------------|:------:|:------:|:------:|:------:|:-------:|:---------:|:-----------:|:---------------:|
| PickScore (off-the-shelf)         | 0.010  | 0.028  |   –    |   –    |    –    |     –     |      –      |        –        |
| EVA-CLIP encoder (fine-tuned)     | 0.157  | 0.143  | 2.835  | 0.350  |  0.082  |   0.549   |   0.00512   |     0.01614     |
| CLIP encoder (fine-tuned)         | 0.390  | 0.378  | 2.462  | 0.251  |  0.122  |   0.747   |   0.00425   |     0.01437     |
| RAHF (multi-head)                 | 0.666  | 0.654  | 1.971  | 0.425  |  0.302  |   0.877   |   0.00141   |     0.01216     |
| RAHF (augmented prompt)           | 0.693  | 0.681  | 1.652  | 0.556  |  0.409  |   0.913   |   0.00095   |     0.00920     |
| **HEIE (ours)**                   |**0.697**|**0.683**|**1.634**|**0.574**|**0.417**| **0.915** | **0.00014** |  **0.00825**   |

---

### Performance of image implausibility explanations on our \abbdata Dataset

| Method                        | Perplexity↓ | GPT-4o Eval↑ | Human Eval↑ |
|-------------------------------|:-----------:|:------------:|:-----------:|
| Qwen2-VL-7B-Instruct          |   1.924209  |   1.910995   |   1.979058  |
| DeepSeek-VL-7B-chat           |   1.794179  |   1.952880   |   1.883770  |
| InternVL2-8B                  |   1.456884  |   2.695288   |   2.603141  |
| GLM-4V-9B                     |   1.320043  |   2.486911   |   2.653403  |
| GPT-4o                        |      –      |   3.828272   |   3.998953  |
| Claude-3.5-Sonnet             |      –      |   3.938220   |   4.080628  |
| **HEIE (ours)**               | **1.031390**| **4.582199** | **4.352880**|

---

### Results on AbHuman (all models finetuned)

| Method         | All Data MSE↓ | GT=0 MSE↓ | KLD↓  | CC↑   | SIM↑   | AUC-Judd↑ |
|----------------|:-------------:|:---------:|:-----:|:-----:|:------:|:---------:|
| InternViT      |    0.07318    |  0.07248  | 3.515 | 0.019 | 0.091  |   0.524   |
| EVA-CLIP       |    0.00924    |  0.00207  | 3.226 | 0.582 | 0.095  |   0.607   |
| CLIP           |    0.00916    |  0.00920  | 1.953 | 0.244 | 0.154  |   0.636   |
| **HEIE (ours)**| **0.00510**   | **0.00076**| **1.629** | **0.684** | **0.423** | **0.938** |



## 📖 Citation
If you find HEIE useful for your research or applications, please cite our paper:

```
@inproceedings{yang2025heie,
  title={Heie: Mllm-based hierarchical explainable aigc image implausibility evaluator},
  author={Yang, Fan and Zhen, Ru and Wang, Jianing and Zhang, Yanhao and Chen, Haoxiang and Lu, Haonan and Zhao, Sicheng and Ding, Guiguang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3856--3866},
  year={2025}
}
```

## Acknowledgement
- [InternVL2](https://github.com/OpenGVLab/InternVL)
