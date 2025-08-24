## NER 数据预处理设计与使用指南

本指南介绍 @ner/ 中“地址文本”预处理的设计、架构，以及如何在 train / eval / predict 全流程中使用与配置。目标是：统一输入归一与清洗、保持 tokens/labels 对齐安全、并以可组合的步骤式流水线保障灵活扩展。

### 目录

- 设计目标与原则
- 架构概览（Pipeline/Step）
- 流程集成点：train / eval / predict
- 配置与使用（步骤配置、向后兼容、CLI 覆盖）
- 常见步骤库与参数
- 添加/修改步骤的建议与注意事项
- 调试、日志与回滚策略
- 性能与优化建议
- 常见问题（FAQ）

## 设计目标与原则

- 统一性：在训练、评估、预测使用一致的清洗与归一策略。
- 安全性：训练/评估阶段默认不改变 token 边界与数量，保证 labels 对齐不被破坏。
- 灵活性：以“步骤 Step”为最小单元，按顺序组合为 Pipeline，可自由增删改并配置参数。
- 可观测性：提供可控的调试日志与对齐检查，便于定位数据问题。
- 可回滚：随时一键关闭预处理，快速对比行为与回退风险。

## 架构概览（Pipeline/Step）

### 核心概念

- Preprocessor（流水线）：顺序执行多个步骤（Step），对文本或分词序列进行处理。
- Step（步骤）：单一职责的小步骤，完成一种明确定义的清洗/归一任务。
- 模式：
  - text 模式：输入/输出为字符串，常用于 predict。
  - tokens 模式：输入为 tokens 与可选 labels，常用于 train/eval，要求严格的对齐安全。

### 接口约定（概念性说明）

- BaseStep
  - name: 步骤名
  - is_label_safe: 是否不会改变 token 数量与位置（默认为 true）
  - apply_text(text) -> text
  - apply_tokens(tokens, labels=None) -> (tokens, labels)

约束：

- 训练/评估默认仅执行 is_label_safe=true 的步骤。
- 如确需改变 token 边界（例如缩写替换导致分词不同），必须显式开启，并在 tokens 模式中同步更新 labels（默认不建议）。

### 模块结构（规划）

为便于扩展与阅读，建议在 `src/ner/preprocess/` 下组织：

- `pipeline.py`：Preprocessor 与 Step 抽象、调度逻辑
- `builder.py`：从配置构建 Pipeline，支持向后兼容
- `steps/`：原子步骤实现（见“常见步骤库与参数”）
- `profiles.py`（可选）：预置 profile，如 strict/serving/none

说明：本指南侧重概念与使用。具体实现按项目节奏逐步落地。

## 流程集成点：train / eval / predict

### Train（训练）

- 入口：`NERDataProcessor` 在加载与预处理样本时，对 tokens+labels 调用 Preprocessor 的 tokens 模式。
- 默认仅执行对齐安全步骤（例如 Unicode 归一、空白归一、去修饰、大小写归一等）。
- DataLoader 之后的流程（对齐、tokenize、对齐调试）保持不变。

### Eval（评估）

- 若评估数据来自数据文件（与训练数据相同结构），入口与训练一致，已在处理器中执行。
- 若使用“文本+真值标签”的评估路径：
  - 推荐优先以 tokens 驱动（直接使用数据集的 tokens），模型提供按 tokens 预测的方法时，对齐最稳。
  - 若必须走文本：在送入模型前对文本执行 text 模式预处理，并确保与真值对齐的切分策略一致。

### Predict（预测）

- 在模型推理前对输入文本执行 text 模式预处理。
- 可根据线上需求启用更“激进”的显示归一（如缩写/同义映射）。

## 配置与使用

### 1) 新推荐配置：自定义流水线

在国家/任务配置的 `data` 区域新增可选字段 `preprocessing_pipeline`（步骤按顺序执行）：

```json
{
  "data": {
    "preprocessing_pipeline": [
      { "step": "unicode_normalize", "params": { "form": "NFC" } },
      { "step": "arabic_remove_diacritics" },
      { "step": "whitespace_normalize", "params": { "collapse": true } },
      { "step": "lowercase" },
      { "step": "punctuation_filter", "params": { "keep": [",", "."] } }
    ]
  }
}
```

说明：

- 未提供 `params` 的步骤按默认参数执行。
- 训练/评估阶段会自动过滤非对齐安全步骤，除非显式允许。

### 2) 向后兼容：布尔开关到步骤映射

若未配置 `preprocessing_pipeline`，依旧支持 `data.preprocessing` 下的布尔开关（参考 `docs/ner/NER_Guide.md` 示例），内部自动映射为等价步骤：

- `clean_text` → `unicode_normalize` + 基础字符清洗
- `normalize_arabic` → `arabic_normalize`
- `remove_diacritics` → `arabic_remove_diacritics`
- `handle_mixed_script` → `mixed_script_normalize`
- 其余 tokenizer 相关选项不属于预处理步骤

优先级：`preprocessing_pipeline` > 旧布尔开关。

### 3) CLI 覆盖（可选能力）

为方便快速回滚或对比，预留以下 CLI 参数（实现计划中）：

- `--preprocess-profile {strict,serving,none}`：覆盖配置内的步骤集。
- `--no-preprocess`：完全禁用预处理，便于回归与排查。

## 常见步骤库与参数

以下步骤建议作为基础库（名称用于 `step` 字段）：

- unicode_normalize：Unicode 归一（参数：`form`=NFC/NFKC/...）
- whitespace_normalize：空白归一（`collapse`: 是否合并多空格；`trim`: 是否首尾裁剪）
- lowercase：小写化（对阿语等大小写不敏感语言可禁用）
- punctuation_filter：过滤标点/特殊字符（`keep`: 保留列表；`remove`: 指定移除集）
- arabic_remove_diacritics：移除阿语重音/元音符号
- arabic_tatweel_strip：去阿语延音符（ـ）
- arabic_normalize：阿语字形归一（如不同 Hamza 形态统一）
- number_normalize：数字标准化（阿拉伯数字 ↔ 西文数字，千分位、空格统一）
- abbrev_synonym_map：缩写/同义映射（`map`: {"st": "street", ...}；注意对齐安全）
- mixed_script_normalize：混合文字系统统一（拉丁/阿语数字、符号统一）

注意：

- 标注为 is_label_safe=false 的步骤仅建议在 predict 使用，除非已实现 labels 同步并显式开启。

## 添加/修改步骤的建议与注意事项

- 单一职责：每个步骤只做一件事，便于开关与调试。
- 对齐安全优先：训练/评估阶段默认仅允许不改变 token 边界的步骤。
- 参数化：避免写死阈值或替换表，在 `params` 中暴露可配项。
- 可测试：为步骤准备小样本的 text 与 tokens 用例，验证前后差异与对齐安全。
- 可观测：在调试模式下，输出前后对比样例（受采样与长度限制）。

示例（概念）——新增步骤的配置：

```json
{
  "data": {
    "preprocessing_pipeline": [
      { "step": "unicode_normalize" },
      { "step": "custom_street_synonym", "params": { "map": { "st": "street", "rd": "road" } } }
    ]
  }
}
```

## 调试、日志与回滚策略

- 设置环境变量 `DEBUG_PREPROCESS=1` 可开启调试：
  - 随机采样若干条样本打印前后对比。
  - tokens 模式下检查 tokens 与 labels 长度一致性。
- 回滚：通过配置将 `preprocessing_pipeline` 置空，或使用 `--no-preprocess`（若已实现）临时关闭所有预处理。

## 性能与优化建议

- 步骤均为 O(n) 字符处理，成本低。批量数据在数据加载阶段完成，避免训练时重复开销。
- 尽量减少会导致字符串重构的重复遍历，将相近逻辑合并为一次 pass（例如统一空白与去首尾空格）。
- 对于大映射表（如缩写同义词），使用预编译正则或字典前缀树以降低开销。

## 常见问题（FAQ）

### 训练指标突然变化，可能与预处理有关吗？

可能。先用 `--no-preprocess` 或清空 `preprocessing_pipeline` 做对照。如果差异来自步骤变更，逐步二分定位具体步骤。

### 评估集和预测时表现不一致？

检查两处是否使用了同一套预处理步骤与分词策略。评估若走文本路径，请确保 text 模式预处理与训练数据构造时一致。

### 能否在训练中合并 tokens（如去除连字符并合并两词）？

不建议。此类步骤会破坏 labels 对齐，除非已实现可靠的 labels 同步与验证，并在配置中显式开启。

### 是否有推荐的“预设方案”？

- strict（训练/评估）：`unicode_normalize`、`arabic_remove_diacritics`、`whitespace_normalize`、`lowercase`、必要的 `arabic_normalize`。
- serving（线上预测）：在 strict 基础上，按需加入 `abbrev_synonym_map`、`number_normalize` 等显示归一。

## 与现有文档的关系

- 请配合阅读 `docs/ner/NER_Guide.md` 与 `docs/ner/data_loader_guide.md`、`docs/ner/data_processor_guide.md`，了解数据加载、标签对齐与整体训练流程，以便正确放置预处理环节。

—— 以上为预处理的设计与使用说明。实现细节将按本指南逐步落地，并在 CLI 与配置验证中提供相应支持。


