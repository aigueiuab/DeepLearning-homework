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