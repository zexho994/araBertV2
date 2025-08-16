# NER模型训练Python调试指南

本文档详细说明如何使用IDE的Python调试功能启动和调试NER模型训练流程，帮助开发者快速定位和解决训练过程中的问题。

## 1. 项目CLI结构概览

### 1.1 项目入口点

```
araBertV2/
├── ner_cli.py                    # 主CLI入口文件
├── src/ner/
│   ├── cli/
│   │   ├── main.py               # CLI管理器
│   │   └── commands.py           # 命令实现
│   ├── training/
│   │   └── trainer.py            # 训练器实现
│   └── ...
└── tests/ner/
    └── debug_train.py            # 调试脚本
```

### 1.2 CLI命令结构

**主要命令**:
- `train`: 训练NER模型
- `evaluate`: 评估模型性能
- `predict`: 进行预测
- `config`: 配置管理
- `data`: 数据处理

**训练命令示例**:
```bash
python ner_cli.py train --country uae_xml_roberta_base
python ner_cli.py train --country uae --epochs 10 --batch-size 16
```

## 2. 使用现有debug_train.py脚本调试

### 2.1 调试脚本概述

项目提供了专门的调试脚本 `tests/ner/debug_train.py`，可以在IDE中直接运行和调试。

### 2.2 调试脚本功能

```python
# 模拟训练命令
def debug_train_command():
    sys.argv = [
        'ner_cli.py',
        'train', 
        '--country', 
        'uae_xml_roberta_base'
    ]
    # 执行训练逻辑
    
# 模拟评估命令
def debug_evaluate_command():
    sys.argv = [
        'ner_cli.py',
        'evaluate',
        '--country', 'uae_xml_roberta_base',
        '--model-path', 'data/ner/models/...',
        '--data-path', 'data/ner/data/.../val.jsonl'
    ]
```

### 2.3 使用调试脚本的步骤

1. **打开调试脚本**: 在IDE中打开 `tests/ner/debug_train.py`
2. **选择调试函数**: 修改main函数中的调用
   ```python
   if __name__ == '__main__':
       # 选择要调试的命令
       exit_code = debug_train_command()      # 调试训练
       # exit_code = debug_evaluate_command()  # 调试评估
   ```
3. **设置断点**: 在需要调试的位置设置断点
4. **启动调试**: 使用IDE的调试功能运行脚本

### 2.4 自定义调试参数

可以修改 `sys.argv` 来测试不同的参数组合：

```python
def debug_custom_train():
    """自定义训练参数调试"""
    sys.argv = [
        'ner_cli.py',
        'train',
        '--country', 'uae_xml_roberta_base',
        '--epochs', '5',
        '--batch-size', '8',
        '--learning-rate', '2e-5'
    ]
    # ... 执行逻辑
```

## 3. 在IDE中直接调试ner_cli.py

### 3.1 配置运行参数

#### 方法1: 修改sys.argv

在 `ner_cli.py` 的main函数开始处添加：

```python
def main():
    # 调试时添加这些行
    import sys
    sys.argv = [
        'ner_cli.py',
        'train',
        '--country', 'uae_xml_roberta_base'
    ]
    
    parser = create_parser()
    args = parser.parse_args()
    # ... 其余代码
```

#### 方法2: 使用IDE运行配置

在IDE中配置运行参数，而不修改代码。

### 3.2 关键调试点

**主要调试位置**:
1. **参数解析**: `ner_cli.py` 中的 `create_parser()` 和 `main()`
2. **命令执行**: `src/ner/cli/commands.py` 中的 `TrainCommand.execute()`
3. **配置加载**: `ConfigManager.load_country_config()`
4. **训练器初始化**: `NERTrainer.__init__()`
5. **训练循环**: `NERTrainer.train()` 和 `train_epoch()`

## 4. 不同IDE的调试配置

### 4.1 PyCharm调试配置

#### 配置运行/调试配置

1. **创建新配置**:
   - 点击 "Run" → "Edit Configurations"
   - 点击 "+" → "Python"

2. **配置参数**:
   ```
   Name: NER Train Debug
   Script path: d:\IdeaProjects\araBertV2\ner_cli.py
   Parameters: train --country uae_xml_roberta_base
   Working directory: d:\IdeaProjects\araBertV2
   Python interpreter: 项目解释器
   ```

3. **环境变量** (可选):
   ```
   PYTHONPATH=d:\IdeaProjects\araBertV2\src
   ```

#### 使用调试脚本

1. **直接调试**:
   - 打开 `tests/ner/debug_train.py`
   - 右键 → "Debug 'debug_train'"

2. **设置断点**:
   - 在代码行号左侧点击设置断点
   - 红色圆点表示断点已设置

#### 调试技巧

- **条件断点**: 右键断点 → "More" → 设置条件
- **日志断点**: 不停止执行，只输出日志
- **变量监视**: 在 "Variables" 面板查看变量值
- **表达式求值**: "Evaluate Expression" (Alt+F8)

### 4.2 VSCode调试配置

#### 创建launch.json

在 `.vscode/launch.json` 中添加配置：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "NER Train Debug",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/ner_cli.py",
            "args": [
                "train",
                "--country",
                "uae_xml_roberta_base"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Debug Train Script",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tests/ner/debug_train.py",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        },
        {
            "name": "NER Evaluate Debug",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/ner_cli.py",
            "args": [
                "evaluate",
                "--country", "uae_xml_roberta_base",
                "--model-path", "data/ner/models/uae_xml_roberta_base/uae_xml_roberta_base_model",
                "--data-path", "data/ner/data/uae_xml_roberta_base/val.jsonl"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        }
    ]
}
```

#### 使用调试功能

1. **启动调试**: F5 或点击调试按钮
2. **设置断点**: 点击行号左侧
3. **调试控制**:
   - F10: 单步跳过
   - F11: 单步进入
   - Shift+F11: 单步跳出
   - F5: 继续执行

#### VSCode调试面板

- **Variables**: 查看当前作用域变量
- **Watch**: 监视特定表达式
- **Call Stack**: 查看调用栈
- **Debug Console**: 执行调试命令

### 4.3 Jupyter Notebook调试

#### 创建调试Notebook

```python
# 在Jupyter中调试训练流程
import sys
import os
from pathlib import Path

# 设置项目路径
project_root = Path('/d/IdeaProjects/araBertV2')
sys.path.insert(0, str(project_root / 'src'))
os.chdir(project_root)

# 导入必要模块
from ner.cli.main import NERCLIManager
from ner.config import ConfigManager
from ner.training import NERTrainer

# 加载配置
config_manager = ConfigManager('data/ner/configs')
config = config_manager.load_country_config('uae_xml_roberta_base')

# 初始化训练器
trainer = NERTrainer(config, {})

# 分步调试
trainer.prepare_data()      # 调试数据准备
trainer.prepare_model()     # 调试模型初始化
trainer.prepare_optimizer() # 调试优化器设置

# 训练一个epoch进行测试
# trainer.train_epoch(0)
```

## 5. 调试配置和断点设置技巧

### 5.1 关键断点位置

#### 训练流程关键点

```python
# 1. 配置验证
# src/ner/cli/commands.py:TrainCommand.execute()
if not validator.validate_config(config, args.country):
    # 断点：配置验证失败
    pass

# 2. 训练器初始化
# src/ner/training/trainer.py:NERTrainer.__init__()
self.device = self._setup_device()
# 断点：检查设备配置

# 3. 数据准备
# src/ner/training/trainer.py:prepare_data()
train_examples = processor.load_data_file(data_config['train_file'])
# 断点：检查数据加载

# 4. 模型初始化
# src/ner/training/trainer.py:prepare_model()
self.model = BertNERModel.from_pretrained(...)
# 断点：检查模型加载

# 5. 训练循环
# src/ner/training/trainer.py:train_epoch()
for batch_idx, batch in enumerate(progress_bar):
    # 断点：检查批次数据
    outputs = self.model(**batch)
    # 断点：检查模型输出
    loss = outputs['loss']
    # 断点：检查损失计算
```

### 5.2 条件断点示例

```python
# 只在特定条件下停止
# 条件: batch_idx == 0  (只在第一个批次停止)
# 条件: loss.item() > 1.0  (只在损失过高时停止)
# 条件: epoch >= 2  (只在第3个epoch及以后停止)
```

### 5.3 日志断点

```python
# 不停止执行，只输出信息
# 表达式: f"Batch {batch_idx}, Loss: {loss.item():.4f}"
# 表达式: f"Learning rate: {self.optimizer.param_groups[0]['lr']}"
```

### 5.4 变量监视

**重要变量监视**:
- `config`: 训练配置
- `self.device`: 训练设备
- `train_examples`: 训练样本
- `self.model`: 模型实例
- `loss.item()`: 当前损失值
- `self.optimizer.param_groups[0]['lr']`: 当前学习率

## 6. 常见调试问题和解决方案

### 6.1 导入错误

**问题**: `ModuleNotFoundError: No module named 'ner'`

**解决方案**:
1. 确保PYTHONPATH包含src目录
2. 在IDE中设置正确的工作目录
3. 使用相对导入或绝对导入

```python
# 方法1: 添加路径
import sys
sys.path.insert(0, 'src')

# 方法2: 设置环境变量
os.environ['PYTHONPATH'] = 'src'
```

### 6.2 配置文件找不到

**问题**: `FileNotFoundError: config file not found`

**解决方案**:
1. 检查工作目录是否正确
2. 确认配置文件路径
3. 使用绝对路径

```python
# 检查当前工作目录
print(f"Current working directory: {os.getcwd()}")

# 检查配置文件是否存在
config_path = Path('data/ner/configs/countries/uae_xml_roberta_base.json')
print(f"Config exists: {config_path.exists()}")
```

### 6.3 CUDA内存错误

**问题**: `CUDA out of memory`

**调试方法**:
1. 监视GPU内存使用
2. 减少批次大小
3. 使用CPU进行调试

```python
# 监视GPU内存
import torch
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# 强制使用CPU调试
config['hardware']['device'] = 'cpu'
```

### 6.4 数据加载错误

**问题**: 数据格式或路径错误

**调试技巧**:
```python
# 检查数据文件
data_file = config['data']['train_file']
print(f"Data file: {data_file}")
print(f"File exists: {Path(data_file).exists()}")

# 检查数据格式
with open(data_file, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    print(f"First line: {first_line}")
```

### 6.5 模型加载错误

**问题**: 预训练模型下载或加载失败

**解决方案**:
1. 检查网络连接
2. 使用本地模型路径
3. 设置代理或镜像

```python
# 使用本地模型
config['model']['pretrained_model'] = '/path/to/local/model'

# 或设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## 7. 高级调试技巧

### 7.1 性能分析

```python
# 使用cProfile分析性能
import cProfile
import pstats

def profile_training():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行训练代码
    trainer.train()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 显示前10个最耗时的函数
```

### 7.2 内存分析

```python
# 使用memory_profiler监控内存
from memory_profiler import profile

@profile
def debug_memory_usage():
    trainer.prepare_data()
    trainer.prepare_model()
    # 训练一个小批次
    trainer.train_epoch(0)
```

### 7.3 梯度检查

```python
# 检查梯度是否正常
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.6f}")
            if grad_norm == 0:
                print(f"WARNING: Zero gradient for {name}")
            elif grad_norm > 10:
                print(f"WARNING: Large gradient for {name}")
```

### 7.4 模型输出检查

```python
# 检查模型输出分布
def check_model_outputs(outputs):
    logits = outputs['logits']
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean().item():.4f}")
    print(f"Logits std: {logits.std().item():.4f}")
    print(f"Logits min: {logits.min().item():.4f}")
    print(f"Logits max: {logits.max().item():.4f}")
    
    # 检查预测分布
    predictions = torch.argmax(logits, dim=-1)
    unique, counts = torch.unique(predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
```

## 8. 调试最佳实践

### 8.1 调试准备

1. **备份配置**: 调试前备份原始配置文件
2. **小数据集**: 使用小数据集进行快速调试
3. **简化配置**: 减少epoch数、批次大小等
4. **日志级别**: 设置为DEBUG获取详细信息

### 8.2 分层调试

1. **配置层**: 先验证配置文件正确性
2. **数据层**: 检查数据加载和预处理
3. **模型层**: 验证模型初始化和前向传播
4. **训练层**: 调试训练循环和优化过程

### 8.3 调试记录

```python
# 创建调试日志
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Debug information here")
```

## 9. 总结

本指南提供了完整的NER模型训练调试方法，包括：

- **多种调试方式**: 调试脚本、直接调试、IDE配置
- **不同IDE支持**: PyCharm、VSCode、Jupyter
- **关键调试点**: 配置、数据、模型、训练各阶段
- **问题解决**: 常见错误和解决方案
- **高级技巧**: 性能分析、内存监控、梯度检查

通过遵循本指南，开发者可以高效地调试NER模型训练过程，快速定位和解决问题，提高开发效率。

建议在实际调试时：
1. 从简单的调试脚本开始
2. 逐步深入到具体的训练逻辑
3. 使用合适的断点和监视变量
4. 记录调试过程和发现的问题
5. 建立个人的调试工作流程