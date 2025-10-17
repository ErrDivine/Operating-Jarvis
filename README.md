# 鸿蒙系统操控Agent竞赛

## 模型下载

```bash
cd models
bash download.sh
```

## 环境

```bash
conda env create -f environment.yaml
conda activate huawei_agent
```
> **注：** 此环境为测试环境，包含NPU相关包，自行调试时请按需修改。

## 部分文件说明

- `agent.py`：选手需要在此文件中实现自己的Agent。
- `demo_agent.py`：示例Agent，选手可参考此文件实现自己的Agent。
- `bash_run.py`：批跑示例脚本，实际测试平台会使用其他方式调用选手实现的CustomAgent的`run`方法。

## 任务

选手需要在`agent.py`中实现一个Agent，完成对鸿蒙系统的操控任务。具体任务描述和要求请参考赛事手册。修改`agent.py`中的`CustomAgent`类时应当注意：
- `__init__`方法中可以添加初始化代码，但不可修改已有参数。
- `run`方法中需要实现具体的操控逻辑，返回结果为一个字符串，表示要执行的操作，同样不可修改函数参数。
- 所有新增代码必须全部实现在`agent.py`文件中，禁止修改其他文件。

## 上传文件

请将`agent.py`和`models`文件夹按当前结构打包上传，其他文件不需要上传。