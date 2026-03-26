# 深度学习作业

本作业使用 VSCode 完成。

## 环境要求

- Python 3.9

## 环境配置

### 1. 创建并激活虚拟环境

```bash
conda create --name deeplearning python=3.9
conda activate deeplearning
```

### 2. 配置 Jupyter 内核（用于运行 .ipynb 文件）

在 VSCode 中运行 `.ipynb` 文件，需执行以下命令：

```bash
conda install -n deeplearning ipykernel --update-deps --force-reinstall
```

> **注意：** 运行该命令可能会出现与 conda 相关的报错。
>
> **解决方案：** 找到 Anaconda 安装目录下的 `Library/bin` 文件夹，将其中的以下两个文件复制到 `Anaconda/DLLs` 文件夹中：
> - `libcrypto-1_1-x64.dll`
> - `libssl-1_1-x64.dll`

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

完成以上步骤后，即可正常运行和测试作业。

---

## 课后作业 2

### 额外依赖

本章作业需要用到 `tensorflow`、`torch` 和 `torchvision`，上述依赖已写入 `requirements.txt`，执行第 3 步的 `pip install -r requirements.txt` 即可一并安装。

### TensorFlow 版本说明

> **注意：** `chap5_CNN` 中的 TensorFlow 版本代码原本基于 **TensorFlow 1.x** 编写，使用 `tf.placeholder`、`tf.Session` 等 1.x 风格 API。
>
> 当前环境使用的是 **TensorFlow 2.x**，因此对部分数据处理代码做了适配性修改（如数据加载方式、归一化处理等），**不影响作业核心内容（填空部分）的完成**。

## 作业 3 说明

本章作业（循环神经网络 RNN）同样依赖 `torch` 等库，这些已在 `requirements.txt` 中统一安装，无需额外配置即可直接运行。

## 作业 4 说明

### 额外依赖

本章作业需要用到 `tqdm` 库，该依赖已写入 `requirements.txt`，执行第 3 步的 `pip install -r requirements.txt` 即可一并安装，无需额外配置。