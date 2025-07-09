# 🌌 StellarSpeak（星语者）——天文学领域专用大语言模型

> “让知识开口说话” —— 中国科学院大学 **星语调校局** 出品

---

## 📖 简介

**StellarSpeak（星语者）** 是由中国科学院大学 **星语调校局** 设计与训练的 13B 级智能语言模型，专为**天文学、科学问答与数理推理**场景开发。模型基于中科院计算所发布的 **BayLing-13B** 基座，结合参数高效的 **LoRA 技术**，在超过 30 万条高质量自构天文语料与 Alpaca 通用指令语料基础上完成微调训练。

**主要能力：**

- 🔭 天文学知识问答与研究支持  
- 📐 数学/物理推理与通识任务  
- 🧠 自我认知与多轮指令响应  
- 🌐 中英文双语理解与生成能力

---

## 🧠 模型架构

- **基座模型**：BayLing-13B（基于 LLaMA2 架构）
- **微调方式**：LoRA（Low-Rank Adaptation）
- **权重设计**：冻结原始参数，仅优化少量低秩矩阵，节省训练资源
- **部署方式**：支持 8bit/16bit 加载，兼容 HuggingFace

---

## 💻 本地部署教程

### 1️⃣ 环境准备

```bash
conda create -n stellar python=3.10
conda activate stellar 
```

### 2️⃣ 下载基座模型 BayLing

在相应 HuggingFace 仓库中下载基座模型参数文件，并命名为：

- bayling-2-7b/ 或 bayling-13b/

源仓库地址：https://huggingface.co/ICTNLP/bayling-2-7b

### 3️⃣ 安装依赖

```bash
pip install -r requirements.txt
pip install -U torch torchvision torchaudio transformers accelerate protobuf==3.19.0
```

### 4️⃣ 启动交互服务（以 7B 为例）

```bash
python chat.py --model-path ./bayling-2-7b --style rich --load-8bit
```

---


## 🧾 自训语料构建

我们构建了包含**通用指令任务**、**天文与数学领域任务**、**模型自我认知**三大类的高质量训练语料。

📁 所有语料已开源发布在：

👉 https://huggingface.co/TQLLab/StellarSpeak_13B_LLM/tree/main/dataset

📄 数据格式为三元组结构：

```json
{
  "instruction": "你是一位资深天文学家，请认真分析并回答下列问题。 假设你是一位天文学家。以下是一道【天文技术与方法】方向的简答题，请简要回答问题，突出重点。",
  "input": "有人说“偏振光有什么应用场景？”，你怎么看？",
  "output": "解释如下：偏光式3D技术普遍用于商业影院和其它高端应用，它是偏振光的典型应用。在技术方式上和快门式是一样的，其不同的是被动接收所以也被称为属于被动式3D技术，辅助设备方面的成本较低，但对输出设备的要求较高，所以非常适合商业影院等需要众多观众的场所使用。"
}
```

📌 **数据合并说明：**

- 通用 + 专业 + 自我认知语料统一整合为一个 `.json` 文件用于训练；
- 若某类数据数量明显偏少，可通过复制扩充其比例，防止训练中被忽视。

💡 **TIP：** 不同 Prompt 模板会导致微小性能差异，相关实验分析见：[Astro-QA](https://github.com/ACMISLab/Astro-QA)

---


## 🔧 LoRA 微调步骤

### 1️⃣ 克隆微调框架

```bash
git clone https://github.com/tloen/alpaca-lora.git
cd alpaca-lora
```

### 2️⃣ 安装依赖

编辑 `requirements.txt`，删除：

```diff
- git+https://github.com/huggingface/peft.git
```

然后执行：

```bash
pip install -r requirements.txt
pip install scipy peft pytest pyyaml
pip install datasets==2.10.1 fsspec==2023.9.2 transformers==4.44.2
```

### 3️⃣ 配置指令模板

在 `alpaca-lora` 仓库中的文件夹中新建文件 `templates/bayling.json`，内容如下：

```json
{
  "description": "Template used by 星语者（StellarSpeak）.",
  "prompt_input": "I am an intelligent language assistant developed by 星语调校局.
Below is a dialog consisting of instructions and responses. Write a response that completes the request.

### Instruction:
{instruction} {input}
### Response:
",
  "prompt_no_input": "I am an intelligent language assistant developed by 星语调校局.
Below is a dialog consisting of instructions and responses. Write a response that completes the request.

### Instruction:
{instruction}
### Response:
",
  "response_split": "### Response:"
}
```

### 4️⃣ 微调脚本参考

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py   --base_model ./bayling-2-7b   --data_path ./alpaca_data.json   --output_dir ./lora-bayling   --batch_size 128   --micro_batch_size 4   --num_epochs 1   --learning_rate 1e-4   --cutoff_len 512   --lora_r 8   --lora_alpha 16   --lora_dropout 0.05   --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'   --train_on_inputs False   --prompt_template_name 'bayling'   --group_by_length
```

📌 **资源需求说明：**

| 模型版本 | cutoff_len | micro_batch_size | 显存需求 |
|----------|------------|------------------|-----------|
| 7B       | 512        | 4                | ≈ 50 GB   |
| 13B      | 256        | 2                | ≈ 80 GB   |

---


## 🛰️ 星语者模型发布信息

我们已经将训练完成的 **星语者-13B** 模型开源发布，**当前发布版本为未与基座模型合并的 LoRA 微调权重**，需手动与 BayLing-13B 进行合并。

📦 模型仓库地址：

👉 https://huggingface.co/TQLLab/StellarSpeak_13B_LLM

包含内容：


- 13BSS_model_tensors/v3/：LoRA 微调权重（adapter 模型）
- dataset/：训练所用指令语料
- README.md：使用说明
---

## 🔗 模型合并与导出
将 LoRA 权重与基座模型合并为 HuggingFace 格式完整模型：

```bash
python export_hf_checkpoint.py   --base-model ./bayling-13b   --lora-model ./13BSS_model_tensors/v3   --output-model ./merged_model_hf
```
合并后可直接用于推理或部署。


---

## 📊 模型测试与性能评估

- 支持通用任务（Vicuna-80）、自我认知任务、特定领域测试
- 支持人工打分与 GPT-4 比较评估

---

## 🧱 硬件需求建议
- 训练推荐：

    - 7B LoRA 训练：50GB 显存

    - 13B LoRA 训练：80GB 显存

    - 推荐显卡：A100 40G、H800、RTX 6000 Ada、RTX 4090

本地部署推荐：

| 模型版本 | 部署显存 | 推荐显卡 |
|----------|-----------|-----------|
| 7B       | ≥ 8GiB    | RTX 3060 (12G)、RTX A2000 (12G)、A10 |
| 13B      | ≥ 16GiB   | RTX 4080 (16G)、RTX 4090 (24G)、A6000、L40 |

训练建议使用 A100 40G、H800、RTX 6000 Ada。

---

## 🤝 参与贡献

| 姓名 | 分工说明 |
|------|----------|
| 王祺森 | 负责**天文领域知识**、数学基础计算方面的模型评估；**构建与优化自我认知语料库** |
| 刘擎天 | 从 Astro-QA 数据集中**整合 10 万条中文天文知识**；**训练并微调星语者-7B 与 13B 模型** |
| 高楚皓 | **收集 10 万条数学推理训练语料**；协助完成星语者-13B 模型的训练流程 |
| 周琝轩 | 在 Vicuna-80 中文/英文测试集中**完成 20 条通用任务的人工评估**；测试小组模型自我认知能力 |
| 高秋阳 | **构建自我认知三元语料集合**；整合团队报告与文档撰写内容 |

---

## 🙏 致谢

- 感谢中科院计算所发布的 **BayLing 开源模型与 GitHub 仓库 (https://github.com/ictnlp/BayLing)** 为星语者模型提供了强大的底座与范式支持。
- 感谢《通用大模型原理及训练实践》课程提供的环境、平台与技术指导。
- 感谢每一位团队成员在语料构建、模型训练、评估实验与文档撰写中做出的贡献！