# StellarSpeak_LLM_Astronomy

#### 介绍
我是星语者(StellarSpeak)，是中国科学院大学的星语调校局设计和训练的智能语言模型，专为天文学和科学领域提供语言支持。我的目标是帮助用户理解科学知识，提高对科学概念的掌握度。

#### 软件架构
星语者基于中国科学院计算机研究所的Bayling自研大模型为基座，使用LoRA微调在大量领域知识语料中开发得到。


#### 安装教程

1.  安装相关依赖和环境

    1️⃣ 准备conda环境，安装指定python版本

    `conda create –n xxx python=3.10.`

    `conda activate xxx`

    2️⃣ clone基座大模型仓库
    
    `git clone https://github.com/ictnlp/BayLing`

    3️⃣ 安装环境依赖
    
    `cd BayLing`

    `pip install -r requirements.txt`

    `pip install -U torch torchvision torchaudio transformers accelerate google`
    
    `pip install protobuf==3.19.0`

2.  下载模型文件

    下载本仓库的模型文件到本地即可

3.  启动模型文件
   
    `python chat.py --model-path ~/bayling-2-7b/ --style rich --load-8bit`

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
