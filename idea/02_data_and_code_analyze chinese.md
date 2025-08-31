好的，这是您提供的“代码及数据分析报告”的中文翻译。

# 代码与数据分析报告

## 1. `scripts` 目录代码分析

`scripts` 目录由 `analysis`、`evaluation`、`preprocess`、`training` 四个子目录组成，各脚本的作用如下：

### `preprocess` (预处理)
这些脚本用于将数据加工成适合分析和模型训练的格式。

- **`01_process_excel.py`**: 读取 `data` 目录中的 Excel 文件，将中文列名更改为英文，提取所需信息并将其另存为 JSON 文件到 `processed_data` 目录中。
- **`02_process_pdf.py`**: 提取 `Epidemic_guide.pdf` 文件中的文本并以 JSON 格式保存。
- **`03_unify_and_clean.py`**: 将 `processed_data` 中的所有 JSON 文件合并为一个文件，并执行数据清洗工作，例如规范化数值数据。
- **`04_visualize_data.py`**: 基于合并后的数据，生成各种可视化资料，如数值数据分布图、分类数据频率图和文本数据词云图。
- **`05_prepare_for_tuning.py`**: 基于合并后的数据，为大型语言模型（LLM）的微调（fine-tuning）创建数据集。将数据处理成适合监督式微调（SFT）和偏好学习（RLVR）的格式。
- **`06_advanced_visualization.py`**: 基于最终合并的数据集中的“主要诊断”信息，对罕见诊断进行分组等，执行高级可视化。
- **`07_build_final_dataset.py`**: 以门诊、影像、EMR 等特定数据源为中心，重点关注叙述（narrative）和诊断名称，构建最终数据集。
- **`08_detailed_diagnosis_visualization.py`**: 针对主要诊断名称，执行非常详细的可视化，如前N个分布、树状图、按性别/年龄的分布等。
- **`09_comprehensive_preprocessing.py`**: 这是一个全面的预处理脚本，综合处理住院、检验（LIS）、影像（PACS）、EMR 等多个原始数据，基于入院后前 72 小时的数据，生成用于分析的数据集（`analysis_ready_data.json`）。
- **`10_split_dataset.py`**: 按患者 ID 将 `analysis_ready_data.json` 文件分割为训练集（train）、验证集（validation）和测试集（test）。在分割时防止数据泄露，并保持结果变量（outcome）的分布。
- **`11_generate_outcomes.py`**: 为每个患者生成结果变量（Y）。此处将“重症结果”定义为 14 天内入住或再入住 ICU，并基于此生成 `outcomes.json` 文件。
- **`11_generate_icu_outcomes.py`**: 这是另一个版本的结果生成脚本，通过在 EMR 数据的出院记录中查找“ICU”相关关键词，来生成是否入住 ICU 的结果变量。
- **`12_create_timeseries_dataset.py`**: 将分割后的数据集转换为适合时序深度学习模型（如 Temporal Fusion Transformer, TFT）使用的格式。为每个患者生成 72 小时的序列数据，并以 Parquet 文件格式保存。

### `training` (模型训练)
这些脚本使用预处理后的数据来训练模型。

- **`01_train_baseline_slr.py`**: 训练基本的逻辑回归（Logistic Regression）模型。执行特征工程、数据管道创建、模型训练及保存。
- **`02_train_tft.py`**: 使用时间序列数据集训练 Temporal Fusion Transformer (TFT) 模型，并保存训练好的模型。

### `evaluation` (模型评估)
该脚本用于评估已训练模型的性能。

- **`01_evaluate_tft.py`**: 使用测试数据集评估已训练的 TFT 模型。计算 AUROC、AUPRC 等性能指标，并生成校准图（calibration plot）以评估模型的可靠性。

### `analysis` (分析)
该脚本用于分析数据分割的结果。

- **`01_inspect_data_split.py`**: 检查训练集、验证集和测试集之间的结果变量分布是否保持一致。

## 2. 数据结构分析

我们通过分析预处理流程中的两个核心 JSON 产出文件，来报告其数据结构。

### 1. `final_unified_dataset.json`
该文件由 `07_build_final_dataset.py` 脚本生成，主要围绕叙述性（narrative）数据和诊断名称进行结构化。

**第一个数据记录示例：**
```json
{
    "source": "outpatient",
    "visit_id": 4000004,
    "department": "心血管内科",
    "visit_time": "2023-01-01T11:20:00",
    "clinical_narrative": "Chief Complaint: 胸闷\nHistory of Present Illness: 活动后胸闷、气促\nPast Medical History: 高血压病史10年\nPhysical Examination: 神清，精神可，对答切题，浅表淋巴结未扪及肿大，巩膜无黄染，口唇无发绀，颈软，无抵抗，双肺呼吸音清，未闻及干湿性啰音，心率78次/分，律齐，各瓣膜听诊区未闻及病理性杂音，腹平软，无压痛、反跳痛，肝脾肋下未及，双下肢无水肿。",
    "primary_diagnosis": "高血压病",
    "primary_diagnosis_icd": "I10.x00",
    "medications": [
        {
            "name": "苯磺酸左旋氨氯地平片",
            "dosage_value": 2.5,
            "dosage_unit": "mg",
            "quantity": 1.0
        }
    ]
}
```

**列（键）说明：**
- `source`: 数据来源 (例如: 'outpatient' - 门诊)
- `visit_id`: 就诊 ID (门诊号或住院号)
- `department`: 科室
- `visit_time`: 就诊时间
- `clinical_narrative`: 包含主诉（Chief Complaint）、现病史（History of Present Illness）等的临床叙述
- `primary_diagnosis`: 主要诊断
- `primary_diagnosis_icd`: 主要诊断的 ICD 编码
- `medications`: 处方药物信息 (名称、剂量、单位、数量)

### 2. `analysis_ready_data.json`
该文件由 `09_comprehensive_preprocessing.py` 脚本生成，它汇总了每位患者入院后 72 小时的数据，以便直接用于模型构建。

**第一个数据记录示例：**
```json
{
    "inpatient_id": "3000001",
    "t0_admission_time": "2023-01-15 09:30:00",
    "meds_atc_list": [
        "N02",
        "J01"
    ],
    "lab_features_packed": {
        "WBC": "1.2|0.5|1",
        "CRP": "-0.3|1.0|0"
    },
    "pacs_summary": "双肺纹理增多，主动脉及冠状动脉钙化",
    "emr_history": "高血压病史5年，否认糖尿病、冠心病史",
    "demographics": {
        "age": 68,
        "sex": "男"
    }
}
```

**列（键）说明：**
- `inpatient_id`: 住院 ID (患者标识符)
- `t0_admission_time`: 入院开始时间 (所有事件的基准时间点)
- `meds_atc_list`: 72 小时内使用的药物 ATC 编码列表
- `lab_features_packed`: 72 小时内主要检验结果的特征 (首次测量值的 Z-score、24 小时变化量、是否超出正常范围)
- `pacs_summary`: 72 小时内影像（PACS）检查结果摘要
- `emr_history`: 从 EMR 中提取的患者既往史
- `demographics`: 年龄、性别等人口统计信息

## 3. 原始数据详细分析 (Excel)

这是对位于 `data` 目录中的 5 个原始 Excel 文件的详细分析。基于每个文件的列结构和首位患者数据编写。

### 1. 嘉和EMR数据.xlsx (电子病历)

- **文件名**: 嘉和EMR数据.xlsx
- **数据大小**: 6.8 MB
- **数据量**: 2,321 行, 19 列

#### 列分析

| 列名 | 数据类型 | 说明 |
| --- | --- | --- |
| 患者姓名 | object | 患者姓名 |
| 住院号 | int64 | 住院号 |
| 职业 | object | 职业 |
| 户籍 | object | 户籍 |
| 现住详细地址 | object | 现居住地详细地址 |
| 主诉 | object | 主诉 (Chief Complaint) |
| 现病史 | object | 现病史 |
| 既往史 | object | 既往史 |
| 体格检查 | object | 体格检查 |
| 首次病程记录 | object | 首次病程记录 |
| 出院记录 | object | 出院记录 |
| 主诊断 | object | 主要诊断 |
| 主诊断编码 | object | 主要诊断编码 (ICD) |
| 主诊断时间 | datetime64[ns] | 主要诊断时间 |
| 次诊断 | object | 次要诊断 |
| 次诊断编码 | object | 次要诊断编码 (ICD) |
| 次诊断时间 | object | 次要诊断时间 |
| 次诊断医生 | object | 次要诊断医生 |

#### 首位患者数据 (患者 ID: 464564)

- **患者姓名**: 施木兰
- **主要症状**: 反复关节疼痛7年余，乏力、四肢水肿1周。
- **现病史**: 7年前开始出现多关节（腕、膝、肩、踝）肿痛，伴有晨僵。
- **既往史**: 无特殊记载。
- **主要诊断**: 类风湿性关节炎
- **次要诊断**: 多发性肌炎、窦性心动过速、肺部感染等多个诊断。

### 2. PACS影像数据 去身份证.xlsx (影像)

- **文件名**: PACS影像数据 去身份证.xlsx
- **数据大小**: 9.0 MB
- **数据量**: 40,051 行, 23 列

#### 列分析

| 列名 | 数据类型 | 说明 |
| --- | --- | --- |
| 患者姓名 | object | 患者姓名 |
| 性别 | object | 性别 |
| 年龄 | object | 年龄 |
| 门诊住院号 | object | 门诊/住院号 |
| 检查号 | object | 检查号 |
| 检查日期 | float64 | 检查日期 |
| 检查结论 | object | 检查结论 |
| 检查表现 | object | 检查所见 |
| 报告日期 | int64 | 报告日期 |

#### 首位患者数据 (患者 ID: 0464564)

- **患者姓名**: 施木兰
- **年龄**: 59岁
- **检查号**: CT1599776
- **检查日期**: 2021-11-01
- **检查结论**: 双肺间质性病变并感染，肺气肿；心脏增大及心包积液。
- **检查所见**: 双肺纹理增多并见异常阴影，多发淋巴结肿大，观察到心包积液（约20mm）。

### 3. LIS 去除身份证.xlsx (检验科)

- **文件名**: LIS 去除身份证.xlsx
- **数据大小**: 24.0 MB
- **数据量**: 373,855 行, 16 列

#### 列分析

| 列名 | 数据类型 | 说明 |
| --- | --- | --- |
| TYPE | object | 检验类型 (微生物/常规) |
| OUTPATIENT_ID | object | 门诊 ID |
| INPATIENT_ID | object | 住院 ID |
| PATIENT_NAME | object | 患者姓名 |
| PATIENT_TYPE | object | 患者类型 (住院/门诊) |
| CLINICAL_DIAGNOSES | object | 临床诊断 |
| TEST_ORDER_NAME | object | 检验医嘱名称 |
| INSPECTION_DATE | int64 | 检验日期 |
| CHINESE_NAME | object | 检验项目中文名 |
| QUANTITATIVE_RESULT | object | 定量结果 |
| QUALITATIVE_RESULT | object | 定性结果 |
| TEST_ITEM_REFERENCE | object | 参考值 |
| TEST_ITEM_UNIT | object | 单位 |

#### 首位患者数据 (患者 ID: 10039122)

- **患者姓名**: 吴语彤
- **年龄**: 6岁
- **检验类型**: 微生物
- **临床诊断**: 支气管肺炎
- **检验项目**: 痰培养(住院)
- **检验结果**: 奈瑟菌

### 4. HIS住院.xlsx (住院处方)

- **文件名**: HIS住院.xlsx
- **数据大小**: 4.7 MB
- **数据量**: 135,634 行, 6 列

#### 列分析

| 列名 | 数据类型 | 说明 |
| --- | --- | --- |
| 姓名 | object | 患者姓名 |
| 住院号码 | int64 | 住院号 |
| 药品医嘱 | object | 药品医嘱 |
| 医嘱开始时间 | object | 医嘱开始时间 |
| 医嘱结束时间 | object | 医嘱结束时间 |
| 药品剂量 | object | 药品剂量 |

#### 首位患者数据 (患者 ID: 506186)

- **患者姓名**: 叶博文
- **药品医嘱**: 呋喃硫胺片
- **医嘱开始时间**: 2020-03-18 11:10:12
- **药品剂量**: 12.5mg

### 5. HIS门诊.xlsx (门诊处方)

- **文件名**: HIS门诊.xlsx
- **数据大小**: 2.2 MB
- **数据量**: 17,943 行, 20 列

#### 列分析

| 列名 | 数据类型 | 说明 |
| --- | --- | --- |
| 姓名 | object | 患者姓名 |
| 门诊号 | object | 门诊号 |
| 职业 | object | 职业 |
| 主诉 | object | 主诉 |
| 主诊断 | object | 主要诊断 |
| 就诊科室 | object | 就诊科室 |
| 药品名称 | object | 药品名称 |
| 药品剂量 | float64 | 药品剂量 |
| 药品数量 | float64 | 药品数量 |
| 处方时间 | datetime64[ns] | 处方时间 |

#### 首位患者数据 (患者 ID: 331102201710130261)

- **患者姓名**: 徐梦婕
- **职业**: 散居儿童
- **主要症状**: 呕吐、腹泻1天，发热0.5天
- **主要诊断**: 轮状病毒肠炎
- **就诊科室**: 儿科门诊
- **处方药品**: (肯特令)蒙脱石散，1.5g，5盒

## 4. 基于患者唯一标识符的数据分析结果

### 4.1 各文件患者数量分析

这是以**患者唯一标识符（住院号、门诊号、检查号等）**而非患者姓名为标准，对各文件独立患者数量的分析结果。

#### 分析结果摘要

| 文件名 | 总记录数 | 独立患者数 | 信息完整患者数 | 使用的唯一标识符列 |
|---|---|---|---|---|
| EMR (嘉和EMR数据.xlsx) | 2,321 | 2,224 | 2,159 | 住院号 |
| PACS (PACS影像数据 去身份证.xlsx) | 40,051 | 37,323 | 37,319 | 检查号 |
| LIS (LIS 去除身份证.xlsx) | 373,855 | 29,555 | 29,555 | OUTPATIENT_ID (门诊患者ID) |
| HIS 住院 (HIS住院.xlsx) | 135,634 | 2,573 | 2,573 | 住院号码 |
| HIS 门诊 (HIS门诊.xlsx) | 17,943 | 8,149 | 4,541 | 门诊号 |

#### 主要发现

1.  **拥有最多独立患者的文件**: PACS (37,323名)
2.  **拥有最多总记录的文件**: LIS (373,855条)
3.  **信息完整率最高的文件**: HIS 住院 (100%)
4.  **信息完整率最低的文件**: HIS 门诊 (55.7%)

### 4.2 文件间患者重叠分析

#### 交集分析结果

- **在所有文件中共同出现的患者**: 0名
  - 原因在于各文件使用不同的患者唯一标识符体系
  - EMR: 住院号, PACS: 检查号, LIS: 门诊患者ID, HIS: 住院/门诊号

- **至少在2个文件中出现的患者**: 9,559名
  - 部分患者在多个系统中以不同 ID 被记录

#### 各文件独立患者数量

| 文件名 | 仅在该文件中出现的患者数 | 比例 |
|---|---|---|
| PACS | 37,323名 | 89.2% |
| LIS | 22,104名 | 74.8% |
| HIS 门诊 | 698名 | 8.6% |
| HIS 住院 | 465名 | 18.1% |
| EMR | 116名 | 5.2% |

### 4.3 数据质量分析

#### 唯一标识符缺失情况

- **EMR**: 0条 (100% 完整)
- **PACS**: 0条 (100% 完整)
- **LIS**: 1,381条 (0.4% 缺失)
- **HIS 住院**: 0条 (100% 完整)
- **HIS 门诊**: 0条 (100% 完整)

#### 信息完整患者比例

- **EMR**: 97.1% (2,159/2,224)
- **PACS**: 99.9% (37,319/37,323)
- **LIS**: 100% (29,555/29,555)
- **HIS 住院**: 100% (2,573/2,573)
- **HIS 门诊**: 55.7% (4,541/8,149)

### 4.4 可视化结果

基于分析结果，生成了如下英文可视化资料：

1.  **Unique Patients per File**: 各文件独立患者数比较
2.  **Total Records per File**: 各文件总记录数比较
3.  **Complete Info Patient Ratio (%)**: 各文件信息完整患者比例
4.  **Average Records per Patient**: 各文件每位患者平均记录数

### 4.5 分析脚本信息

- **脚本位置**: `scripts/preprocess/00_patient_analysis.py`
- **结果保存位置**:
  - `processed_data/patient_analysis_results.json`
  - `visualizations/patient_analysis_summary.png`
- **主要功能**:
  - 自动检测患者唯一标识符列
  - 分析文件间的患者重叠情况
  - 评估数据质量
  - 生成英文可视化图表

### 4.6 结论与建议

1.  **数据整合注意事项**: 由于各文件使用不同的患者唯一标识符体系，在进行患者匹配时可能需要基于姓名进行匹配。

2.  **数据质量**: 除 LIS 外，唯一标识符的缺失情况很少，数据质量良好。

3.  **分析优先级**: PACS 和 LIS 拥有最多的患者数据，因此以这两个文件为中心的分析将非常有用。

4.  **可视化利用**: 已生成的英文可视化资料可用于撰写论文或报告。