# 🚀 快速开始指南

5分钟上手地址数据模拟器！

## 📦 已有内容

✅ **生成器代码**: `generator.py` (600+行，带详细中文注释)  
✅ **词典文件**: 7个实体类型的词典（从真实数据提取）  
✅ **配置文件**: `generator_config.json5`（完整配置）  
✅ **文档**: 5份详细文档  
✅ **测试数据**: 已生成200条样本数据  

---

## ⚡ 快速运行

### 1. 生成100条地址（使用默认配置）

```bash
cd /Users/zexho/Documents/python_script/araBertv2
python3 data/ner/simulator/generator.py
```

**预期输出**:
```
📖 正在加载配置文件...
✅ 配置加载成功

📚 正在加载实体词典...
  ✓ COUNTRY: 3 个实体
  ✓ EMIRATE: 7 个实体
  ✓ CITY: 41 个实体
  ✓ SUB_AREA: 317 个实体
  ✓ COMPOUND: 380 个实体
  ✓ STREET: 491 个实体
  ✓ BUILDING: 1173 个实体
✅ 词典加载完成

🚀 开始生成地址...
   已生成: 100/100

✅ 成功生成 100 条地址
💾 保存到: data/ner/simulator/uae/raw_data/simulated_YYYYMMDD_HHMMSS.csv
```

### 2. 查看生成的数据

```bash
head -5 data/ner/simulator/uae/raw_data/simulated_*.csv | tail -1
```

---

## 🎯 常见使用场景

### 场景1: 生成1000条训练数据

```bash
# 1. 修改配置文件
vim data/ner/simulator/uae/config/generator_config.json5
# 将 "generate_size": 100 改为 "generate_size": 1000

# 2. 运行生成器
python3 data/ner/simulator/generator.py

# 3. 查看结果
wc -l data/ner/simulator/uae/raw_data/simulated_*.csv
```

### 场景2: 生成高质量清洁数据

```bash
# 1. 修改配置文件
vim data/ner/simulator/uae/config/generator_config.json5

# 2. 调整以下参数:
#    "typo_injection": { "enabled": false }
#    "noise_injection": { "global_noise_probability": 0.3 }

# 3. 运行生成器
python3 data/ner/simulator/generator.py
```

### 场景3: 更新词典并重新生成

```bash
# 1. 从新的标注数据提取实体
python3 data/ner/simulator/extract_entities.py

# 2. 查看更新后的词典
wc -l data/ner/simulator/uae/dictionaries/*.txt

# 3. 重新生成数据
python3 data/ner/simulator/generator.py
```

---

## 📂 文件结构

```
simulator/
├── generator.py                    # 🔧 主生成器（运行这个）
├── extract_entities.py             # 🔧 词典提取工具
├── batch_generate.sh               # 🔧 批量生成脚本
│
├── README.md                       # 📖 主文档
├── README_USAGE.md                 # 📖 使用指南
├── CONFIG_GUIDE.md                 # 📖 配置详解
├── EXAMPLES.md                     # 📖 示例展示
├── IMPLEMENTATION.md               # 📖 实现总结
├── QUICKSTART.md                   # 📖 本文档
│
└── uae/                            # 🇦🇪 UAE配置
    ├── config/
    │   └── generator_config.json5  # ⚙️ 配置文件（重点）
    ├── dictionaries/               # 📚 词典目录
    │   ├── building.txt           # 1173个实体
    │   ├── street.txt             # 491个实体
    │   ├── compound.txt           # 380个实体
    │   ├── sub_area.txt           # 317个实体
    │   ├── city.txt               # 41个实体
    │   ├── emirate.txt            # 7个实体
    │   └── country.txt            # 3个实体
    └── raw_data/                   # 📦 输出目录
        └── simulated_*.csv        # 生成的数据
```

---

## ⚙️ 核心配置参数

只需要修改配置文件 `uae/config/generator_config.json5`：

### 控制生成数量
```json5
{
    "generate_size": 100  // 改成你想要的数量
}
```

### 控制噪音水平
```json5
{
    "noise_injection": {
        "global_noise_probability": 0.8  // 0.0-1.0，越大噪音越多
    }
}
```

### 控制大小写变体
```json5
{
    "case_variations": {
        "probability": 0.4  // 0.0-1.0，越大变体越多
    }
}
```

---

## 🎨 生成流程示例

```
T1 (原始):  Villa 276 Al Yalayis 4 Dubai UAE
              ↓
T2 (分隔):  Villa 276, Al Yalayis 4, Dubai, UAE
              ↓
T3 (变体):  villa 276, al yalayis 4, dubai, uae
              ↓
T4 (噪音):  villa 276, Near al yalayis 4, 1234 dubai, uae
```

---

## 📊 生成数据示例

**输出CSV格式**:
```csv
formatted_address,BUILDING,STREET,COMPOUND,SUB_AREA,CITY,EMIRATE,COUNTRY
villa 276, Near al yalayis 4, 1234 dubai, uae,Villa 276,,,Al Yalayis 4,Dubai,,UAE
Burj Al Salam 3603, Trade Center 1, Dubai, UAE,Burj Al Salam,,,Trade Center 1,Dubai,,UAE
```

**特点**:
- ✅ 实体标注100%准确（来自真实词典）
- ✅ 包含阿拉伯语和英语混合
- ✅ 多种格式变体
- ✅ 适度的噪音

---

## 🔍 验证数据质量

```bash
# 1. 统计生成数量
wc -l data/ner/simulator/uae/raw_data/simulated_*.csv

# 2. 查看样本
head -10 data/ner/simulator/uae/raw_data/simulated_*.csv

# 3. 检查字段
head -1 data/ner/simulator/uae/raw_data/simulated_*.csv
```

---

## 💡 下一步

### 1. 阅读详细文档
- [README_USAGE.md](README_USAGE.md) - 完整使用指南
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 配置参数详解
- [EXAMPLES.md](EXAMPLES.md) - 更多示例

### 2. 调整配置
- 修改 `generate_size` 生成更多数据
- 调整噪音概率以适应你的需求
- 添加新的模板以增加多样性

### 3. 扩展词典
- 使用 `extract_entities.py` 从新数据提取
- 或手动编辑词典文件

### 4. 与训练流程集成
```bash
# 将生成的数据用于NER训练
python3 -m src.ner.cli.train_command \
    --data data/ner/simulator/uae/raw_data/simulated_*.csv
```

---

## ❓ 常见问题

**Q: 生成数据不够多？**  
A: 修改配置文件中的 `generate_size` 参数

**Q: 数据质量不好？**  
A: 降低 `noise_injection.global_noise_probability`

**Q: 想要更多样化？**  
A: 增加词典实体数量，提高噪音概率

**Q: 如何添加新实体类型？**  
A: 阅读 [CONFIG_GUIDE.md](CONFIG_GUIDE.md) 了解详情

---

## 🎉 完成！

你已经成功：
- ✅ 了解了项目结构
- ✅ 运行了生成器
- ✅ 生成了第一批数据

现在可以：
1. 调整配置生成更多数据
2. 查看生成结果
3. 将数据用于NER训练

**祝使用愉快！** 🚀

---

**需要帮助？** 查看其他文档或检查代码中的注释。

