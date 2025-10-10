# 地址数据模拟器

基于已有标注数据生成合成地址数据，用于扩充 NER 训练集。

## 设计理念

通过从真实标注数据中提取实体词典，按照预定义的模板规则进行自由组合，生成大量新地址数据，同时保证实体标注的准确性。

**优势：**
- ✅ 大幅降低标注成本和时间
- ✅ 可快速生成大量训练数据
- ✅ 实体标注天然准确（来自真实标注数据）
- ✅ 可控制实体分布，解决数据不平衡问题

## 目录结构

```
simulator/
├── README.md                    # 本文件
├── extract_entities.py          # 实体提取脚本
└── {country}/                   # 按国家/地区组织
    ├── README.md               # 该国家/地区的说明文档
    ├── dictionaries/           # 实体词典目录
    │   ├── building.txt        # 建筑物词典
    │   ├── street.txt          # 街道词典
    │   ├── compound.txt        # 社区/小区词典
    │   ├── sub_area.txt        # 子区域词典
    │   ├── city.txt            # 城市词典
    │   ├── emirate.txt         # 酋长国词典（UAE特有）
    │   └── country.txt         # 国家词典
    └── config/
        └── generator_config.json  # 生成器配置
```

## 当前支持的地区

### UAE (阿联酋)
- 路径：`uae/`
- 实体统计：
  - BUILDING
  - STREET
  - COMPOUND
  - SUB_AREA
  - CITY
  - EMIRATE
  - COUNTRY

详见：[uae/README.md](uae/README.md)

## 工作流程

### 1. 准备实体词典

从已标注数据中提取实体：

```bash
python3 data/ner/simulator/extract_entities.py
```

或者手动编辑词典文件，每行一个实体。

### 2. 配置生成器规则

编辑 `{country}/config/generator_config.json`，配置：

- **模板定义**：地址的排列组合规则
  ```json
  {
    "pattern": "{BUILDING} {STREET} {SUB_AREA} {CITY} {COUNTRY}",
    "weight": 0.25
  }
  ```

- **拼写错误**：每个实体类型的拼写错误概率和类型
- **噪音注入**：标点、空白、数字、无意义词等
- **输出配置**：格式、路径、去重等

### 3. 生成地址数据

```bash
# 使用默认配置（UAE）
python3 data/ner/simulator/generator.py

# 或指定配置文件
python3 data/ner/simulator/generator.py data/ner/simulator/uae/config/generator_config.json5

# 生成更多数据：修改配置文件中的 generate_size 参数
```

**生成流程示例：**
```
T1 (原始): Villa 276 Al Yalayis 4 Dubai UAE
    ↓
T2 (分隔): Villa 276, Al Yalayis 4, Dubai, UAE
    ↓
T3 (变体): villa 276, al yalayis 4, dubai, uae
    ↓
T4 (噪音): villa 276, Near al yalayis 4, 1234 dubai, uae
```

### 4. 质量验证

生成的数据会经过：
- 去重处理
- 长度验证
- 格式检查
- 与真实数据分布对比

## 生成逻辑

```python
# 1. 从词典中选择实体（无层级约束，由模板定义顺序）
entities = {
    "BUILDING": random.choice(building_dict),
    "STREET": random.choice(street_dict),
    ...
}

# 2. 按模板组装
address = template.format(**entities)
# 例如: "Villa 276 Al Qudrah Street Dubai UAE"

# 3. 应用分隔符变体
address = apply_separator_variations(address)
# 例如: "Villa 276, Al Qudrah Street, Dubai, UAE"

# 4. 注入拼写错误
address = inject_typos(address)
# 例如: "Vila 276, Al Qudrah Streat, Dubai, UAE"

# 5. 注入噪音
address = inject_noise(address)
# 例如: "Vila 276, Near, Al Qudrah Streat, Dubai, UAE"

# 6. 应用大小写变体
address = apply_case_variations(address)
```

## 配置说明

### Templates（模板配置）

定义地址的组合模式：
- 所有实体都是必填
- 通过 weight 控制不同模板的生成比例

### Typo Injection（拼写错误注入）

支持的错误类型：
- `swap`: 交换相邻字符（如 "Street" → "Steret"）
- `deletion`: 删除字符（如 "Street" → "Stret"）
- `insertion`: 插入字符（如 "Street" → "Streeat"）
- `substitution`: 替换字符（如 "Street" → "Strext"）

### Noise Injection（噪音注入）

支持的噪音类型：
- **Punctuation**: 标点符号（`,`, `.`, `;`, `-`, `/` 等）
- **Whitespace**: 异常空白（多个空格、制表符）
- **Numbers**: 数字（随机数字、电话号码、P.O. Box）
- **Meaningless Words**: 无意义词（"Near", "Opposite", "Floor" 等）

### Output（输出配置）

- 格式：JSONL（每行一个 JSON 对象）
- 字段：`formatted_address` + 各实体字段
- 去重：避免重复地址
- 打乱：随机化数据顺序

## 注意事项

1. **词典质量**：词典数据来自真实标注，但可能包含噪音或错误，需定期审查
2. **无层级约束**：当前不检查实体间的层级关系（如 Dubai + Abu Dhabi），需在词典准备时保证质量
3. **预处理**：生成的地址会经过 `src/ner/preprocess/pipeline.py` 预处理，不需要过分关注格式问题
4. **数据混合**：建议混合使用真实数据和合成数据训练模型

## 扩展到其他地区

添加新地区的步骤：

1. 创建目录结构：
   ```bash
   mkdir -p data/ner/simulator/{country}/dictionaries
   mkdir -p data/ner/simulator/{country}/config
   ```

2. 准备词典文件（从标注数据提取或手动创建）

3. 复制并修改配置文件：
   ```bash
   cp data/ner/simulator/uae/config/generator_config.json \
      data/ner/simulator/{country}/config/
   ```

4. 根据地区特点调整模板和配置

## 已完成功能

- [x] ✅ 实现地址生成器核心代码
- [x] ✅ 添加命令行接口（CLI）
- [x] ✅ 实现数据质量统计报告
- [x] ✅ 支持多语言变体（阿拉伯语/英语混合）
- [x] ✅ 实体词典提取工具
- [x] ✅ 配置化生成流程（T1->T2->T3->T4）

## 下一步改进

- [ ] 添加数据分布分析工具
- [ ] 添加实体共现规则

