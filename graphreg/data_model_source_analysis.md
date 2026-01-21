# GraphRAG data_model 模块源码解析报告

> 模块路径: `graphrag/data_model/`

## 1. 这个模块是干什么的？

**一句话总结**：`data_model` 定义了 GraphRAG 系统中所有"东西"的数据结构。

就像建房子需要图纸一样，GraphRAG 处理知识图谱也需要先定义"数据长什么样"。这个模块就是那份图纸。

---

## 2. 用生活例子理解

想象你在整理一本**百科全书**：

```
📚 百科全书 (Document)
    │
    ├── 📄 第1章: 人工智能简介 (TextUnit)
    │       │
    │       ├── 👤 OpenAI (Entity - 组织)
    │       ├── 👤 微软 (Entity - 公司)
    │       └── 🔗 "微软投资OpenAI" (Relationship)
    │
    ├── 📄 第2章: 大语言模型 (TextUnit)
    │       │
    │       ├── 👤 GPT-4 (Entity - 产品)
    │       ├── 📋 "GPT-4是最强模型" (Covariate - 声明)
    │       └── 🔗 "OpenAI开发GPT-4" (Relationship)
    │
    └── ...

然后这些内容被自动分组：

🏘️ 社区 (Community)
    │
    ├── 社区A: {微软, OpenAI, Azure} - "AI商业联盟"
    │       └── 📋 社区报告 (CommunityReport)
    │
    └── 社区B: {GPT-4, Claude, Gemini} - "大模型家族"
            └── 📋 社区报告 (CommunityReport)
```

---

## 3. 模块结构总览

```
data_model/
├── __init__.py          # 包初始化
│
├── identified.py        # 🏷️ 基础类：有ID的东西
├── named.py             # 🏷️ 基础类：有名字的东西
│
├── document.py          # 📚 文档：原始输入
├── text_unit.py         # 📄 文本单元：文档的切片
│
├── entity.py            # 👤 实体：人/地点/组织/概念
├── relationship.py      # 🔗 关系：实体之间的连接
├── covariate.py         # 📋 协变量：额外信息（如声明）
│
├── community.py         # 🏘️ 社区：实体的分组
├── community_report.py  # 📋 社区报告：社区的摘要
│
├── schemas.py           # 📐 字段名常量定义
└── types.py             # 🔧 类型别名
```

---

## 4. 类继承关系图

```
                    ┌─────────────┐
                    │ Identified  │  有ID的东西
                    │ ─────────── │
                    │ • id        │
                    │ • short_id  │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Named     │ │  TextUnit   │ │  Covariate  │
    │ ─────────── │ │             │ │             │
    │ • title     │ │ • text      │ │ • subject_id│
    └──────┬──────┘ │ • entity_ids│ │ • type      │
           │        │ • ...       │ │ • ...       │
           │        └─────────────┘ └─────────────┘
           │
    ┌──────┴──────┬─────────────┬─────────────┬────────────────┐
    │             │             │             │                │
    ▼             ▼             ▼             ▼                ▼
┌────────┐  ┌─────────┐  ┌──────────┐  ┌───────────────┐  ┌──────────┐
│Document│  │ Entity  │  │Community │  │CommunityReport│  │Relationship│
│        │  │         │  │          │  │               │  │(特殊：继承│
│• text  │  │• type   │  │• level   │  │• summary      │  │ Identified)│
│• ...   │  │• desc   │  │• parent  │  │• full_content │  │• source   │
└────────┘  │• ...    │  │• children│  │• rank         │  │• target   │
            └─────────┘  │• ...     │  │• ...          │  │• ...      │
                         └──────────┘  └───────────────┘  └──────────┘
```

---

## 5. 核心数据模型详解

### 5.1 基础类

#### `Identified` - 有身份证的东西

```python
@dataclass
class Identified:
    id: str           # 唯一ID，像身份证号
    short_id: str     # 简短ID，像昵称，方便人阅读
```

**通俗理解**：就像每个人都有身份证号（id）和昵称（short_id）一样。

#### `Named` - 有名字的东西

```python
@dataclass
class Named(Identified):
    title: str        # 名字/标题
```

**通俗理解**：在有身份证的基础上，还有个正式名字。

---

### 5.2 输入层：文档和文本单元

#### `Document` - 原始文档 📚

```python
@dataclass
class Document(Named):
    type: str = "text"              # 文档类型
    text: str = ""                  # 原始文本内容
    text_unit_ids: list[str] = []   # 被切分成了哪些文本单元
    attributes: dict | None = None  # 额外属性（作者、日期等）
```

**通俗理解**：就是你输入给 GraphRAG 的原始文件，比如一篇新闻、一份报告。

**生活例子**：
```
Document {
    id: "doc_001",
    title: "2024年AI发展报告",
    type: "text",
    text: "2024年，人工智能领域迎来重大突破...",
    text_unit_ids: ["tu_001", "tu_002", "tu_003"]
}
```

#### `TextUnit` - 文本单元 📄

```python
@dataclass
class TextUnit(Identified):
    text: str                              # 文本内容
    n_tokens: int | None = None            # token数量

    document_ids: list[str] | None = None  # 来自哪些文档
    entity_ids: list[str] | None = None    # 包含哪些实体
    relationship_ids: list[str] | None = None  # 包含哪些关系
    covariate_ids: dict | None = None      # 包含哪些协变量
```

**通俗理解**：把长文档切成小块，每块就是一个 TextUnit。就像把一本书切成一页一页。

**为什么要切分？**
- LLM 有上下文长度限制
- 小块更容易处理和检索
- 可以精确定位信息来源

**生活例子**：
```
TextUnit {
    id: "tu_001",
    text: "OpenAI是一家人工智能研究公司，微软是其主要投资方...",
    n_tokens: 150,
    document_ids: ["doc_001"],
    entity_ids: ["ent_openai", "ent_microsoft"],
    relationship_ids: ["rel_001"]
}
```

---

### 5.3 知识层：实体和关系

#### `Entity` - 实体 👤

```python
@dataclass
class Entity(Named):
    type: str | None = None                # 类型：PERSON/ORG/LOCATION/...
    description: str | None = None         # 描述
    description_embedding: list[float]     # 描述的向量表示

    community_ids: list[str] | None = None # 属于哪些社区
    text_unit_ids: list[str] | None = None # 出现在哪些文本单元
    rank: int | None = 1                   # 重要性排名
```

**通俗理解**：文本中提到的"东西"——人、公司、地点、概念等。

**实体类型举例**：
| 类型 | 例子 |
|------|------|
| PERSON | 马斯克、萨提亚·纳德拉 |
| ORGANIZATION | 微软、OpenAI、谷歌 |
| LOCATION | 硅谷、北京、上海 |
| PRODUCT | GPT-4、Azure、iPhone |
| CONCEPT | 人工智能、深度学习 |

**生活例子**：
```
Entity {
    id: "ent_microsoft",
    title: "微软",
    type: "ORGANIZATION",
    description: "微软是一家全球领先的科技公司，总部位于美国...",
    rank: 95,
    community_ids: ["comm_001"],
    text_unit_ids: ["tu_001", "tu_005", "tu_012"]
}
```

#### `Relationship` - 关系 🔗

```python
@dataclass
class Relationship(Identified):
    source: str                    # 源实体名称
    target: str                    # 目标实体名称
    weight: float | None = 1.0     # 关系权重/强度
    description: str | None = None # 关系描述

    text_unit_ids: list[str] | None = None  # 出现在哪些文本
    rank: int | None = 1           # 重要性排名
```

**通俗理解**：两个实体之间的连接，描述它们是什么关系。

**关系示例**：
```
微软 ──投资──► OpenAI
     (100亿美元)

OpenAI ──开发──► GPT-4

马斯克 ──联合创立──► OpenAI
       (后来离开)
```

**生活例子**：
```
Relationship {
    id: "rel_001",
    source: "微软",
    target: "OpenAI",
    description: "微软向OpenAI投资超过100亿美元，双方建立战略合作",
    weight: 0.95,
    text_unit_ids: ["tu_001", "tu_003"]
}
```

---

### 5.4 补充层：协变量

#### `Covariate` - 协变量 📋

```python
@dataclass
class Covariate(Identified):
    subject_id: str                # 关联的主体ID（如实体ID）
    subject_type: str = "entity"   # 主体类型
    covariate_type: str = "claim"  # 协变量类型（如：claim声明）

    text_unit_ids: list[str] | None = None  # 来源文本
    attributes: dict | None = None          # 额外属性
```

**通俗理解**：实体的"附加信息"，最常见的是**声明（Claim）**——关于实体的某个论断。

**声明 vs 描述 的区别**：
| 类型 | 性质 | 例子 |
|------|------|------|
| 描述 | 客观事实 | "GPT-4是OpenAI开发的大语言模型" |
| 声明 | 可能有争议的论断 | "GPT-4是目前最强的AI模型" |

**生活例子**：
```
Covariate {
    id: "cov_001",
    subject_id: "ent_gpt4",
    subject_type: "entity",
    covariate_type: "claim",
    attributes: {
        "claim_text": "GPT-4在多项基准测试中超越人类专家",
        "status": "SUPPORTED",  # 有证据支持
        "source": "OpenAI技术报告"
    }
}
```

---

### 5.5 聚合层：社区和报告

#### `Community` - 社区 🏘️

```python
@dataclass
class Community(Named):
    level: str                     # 层级（0最细，往上越粗）
    parent: str                    # 父社区ID
    children: list[str]            # 子社区ID列表

    entity_ids: list[str] | None   # 包含的实体
    relationship_ids: list[str]    # 包含的关系
    text_unit_ids: list[str]       # 相关的文本单元

    size: int | None = None        # 社区大小
```

**通俗理解**：关系紧密的一群实体的"朋友圈"。

**层级结构**：
```
Level 2 (最粗): 整个AI行业
    │
    ├── Level 1: 商业AI公司群
    │       │
    │       ├── Level 0: 微软-OpenAI联盟
    │       └── Level 0: Google-DeepMind联盟
    │
    └── Level 1: 开源AI社区
            │
            ├── Level 0: Meta-LLaMA生态
            └── Level 0: HuggingFace生态
```

#### `CommunityReport` - 社区报告 📋

```python
@dataclass
class CommunityReport(Named):
    community_id: str              # 对应的社区ID
    summary: str = ""              # 摘要
    full_content: str = ""         # 完整报告
    rank: float | None = 1.0       # 重要性评分

    full_content_embedding: list[float]  # 报告的向量表示
    size: int | None = None        # 报告大小
```

**通俗理解**：对社区内容的 LLM 生成摘要，就像给一个"朋友圈"写一份总结报告。

**报告示例**：
```markdown
# 微软-OpenAI战略联盟

本社区涵盖微软与OpenAI的深度合作关系及相关产品生态。

## 核心发现

### 巨额投资奠定合作基础
微软累计向OpenAI投资超过100亿美元...

### Azure成为AI基础设施
所有OpenAI模型都运行在Azure云平台上...

### GPT模型商业化
通过Azure OpenAI服务，企业可以直接调用GPT-4...
```

---

## 6. schemas.py - 字段名字典

这个文件定义了所有**字段名常量**，确保代码各处使用一致的名称。

### 6.1 为什么需要这个？

**问题**：如果代码到处写 `"community_id"`、`"Community_ID"`、`"communityId"`...
- 容易写错
- 不一致会导致 bug
- 难以维护

**解决方案**：统一定义常量
```python
COMMUNITY_ID = "community"  # 只定义一次
```

### 6.2 主要常量分类

```python
# === 基础字段 ===
ID = "id"
SHORT_ID = "human_readable_id"
TITLE = "title"
DESCRIPTION = "description"

# === 节点（实体）相关 ===
NODE_DEGREE = "degree"        # 度数（连接数）
NODE_DETAILS = "node_details"
NODE_X = "x"                  # 可视化坐标
NODE_Y = "y"

# === 边（关系）相关 ===
EDGE_SOURCE = "source"        # 起点
EDGE_TARGET = "target"        # 终点
EDGE_WEIGHT = "weight"        # 权重
EDGE_DEGREE = "combined_degree"

# === 社区相关 ===
COMMUNITY_ID = "community"
COMMUNITY_LEVEL = "level"
COMMUNITY_PARENT = "parent"
COMMUNITY_CHILDREN = "children"
SUB_COMMUNITY = "sub_community"

# === 上下文相关 ===
CONTEXT_STRING = "context_string"
CONTEXT_SIZE = "context_size"
CONTEXT_EXCEED_FLAG = "context_exceed_limit"

# === 报告相关 ===
SUMMARY = "summary"
FINDINGS = "findings"
FULL_CONTENT = "full_content"
RATING = "rank"
```

### 6.3 最终输出列定义

schemas.py 还定义了各种数据表的**最终输出格式**：

```python
# 实体表的列
ENTITIES_FINAL_COLUMNS = [
    ID, SHORT_ID, TITLE, TYPE, DESCRIPTION,
    TEXT_UNIT_IDS, NODE_FREQUENCY, NODE_DEGREE, NODE_X, NODE_Y
]

# 关系表的列
RELATIONSHIPS_FINAL_COLUMNS = [
    ID, SHORT_ID, EDGE_SOURCE, EDGE_TARGET, DESCRIPTION,
    EDGE_WEIGHT, EDGE_DEGREE, TEXT_UNIT_IDS
]

# 社区表的列
COMMUNITIES_FINAL_COLUMNS = [
    ID, SHORT_ID, COMMUNITY_ID, COMMUNITY_LEVEL,
    COMMUNITY_PARENT, COMMUNITY_CHILDREN, TITLE,
    ENTITY_IDS, RELATIONSHIP_IDS, TEXT_UNIT_IDS, PERIOD, SIZE
]

# 社区报告表的列
COMMUNITY_REPORTS_FINAL_COLUMNS = [
    ID, SHORT_ID, COMMUNITY_ID, COMMUNITY_LEVEL,
    TITLE, SUMMARY, FULL_CONTENT, RATING, EXPLANATION,
    FINDINGS, FULL_CONTENT_JSON, PERIOD, SIZE
]
```

---

## 7. types.py - 类型别名

非常简单，只定义了一个类型别名：

```python
TextEmbedder = Callable[[str], list[float]]
```

**意思是**：`TextEmbedder` 是一个函数类型，输入字符串，输出浮点数列表（向量）。

用于文本嵌入模型的类型注解。

---

## 8. 数据流：从文档到报告

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              1. 输入阶段                                    │
│                                                                             │
│    用户输入文档                                                             │
│         │                                                                   │
│         ▼                                                                   │
│    ┌──────────┐                                                            │
│    │ Document │  原始文档                                                  │
│    │ • text   │  "2024年，微软宣布向OpenAI追加投资..."                      │
│    └────┬─────┘                                                            │
│         │ 切分                                                              │
│         ▼                                                                   │
│    ┌──────────┐                                                            │
│    │ TextUnit │  文本块                                                     │
│    │ • text   │  "微软宣布向OpenAI追加投资100亿美元..."                      │
│    └────┬─────┘                                                            │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │
          │ LLM 提取
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              2. 提取阶段                                    │
│                                                                             │
│    ┌──────────┐              ┌──────────────┐         ┌───────────┐        │
│    │  Entity  │              │ Relationship │         │ Covariate │        │
│    │          │              │              │         │           │        │
│    │ • 微软   │───投资──────►│ • source:微软│         │ • claim:  │        │
│    │ • OpenAI │              │ • target:OAI │         │   "最大   │        │
│    │ • GPT-4  │◄──开发───────│ • weight:0.9 │         │    投资"  │        │
│    └──────────┘              └──────────────┘         └───────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ 社区检测算法
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              3. 聚合阶段                                    │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │                        Community (社区)                              │ │
│    │                                                                     │ │
│    │  Level 1: 社区X "AI商业联盟"                                        │ │
│    │     │                                                               │ │
│    │     ├── Level 0: 社区A {微软, Azure}                                │ │
│    │     └── Level 0: 社区B {OpenAI, GPT-4}                              │ │
│    │                                                                     │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ LLM 生成摘要
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              4. 报告阶段                                    │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │                    CommunityReport (社区报告)                        │ │
│    │                                                                     │ │
│    │  # AI商业联盟                                                       │ │
│    │                                                                     │ │
│    │  本社区涵盖微软与OpenAI的战略合作...                                 │ │
│    │                                                                     │ │
│    │  ## 发现1: 巨额投资                                                 │ │
│    │  微软向OpenAI投资超过100亿美元...                                    │ │
│    │                                                                     │ │
│    │  ## 发现2: 技术整合                                                 │ │
│    │  GPT-4已深度集成到Azure平台...                                       │ │
│    │                                                                     │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ 存储为 Parquet 文件
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              5. 输出文件                                    │
│                                                                             │
│    📁 output/                                                               │
│    ├── entities.parquet          # 实体表                                   │
│    ├── relationships.parquet     # 关系表                                   │
│    ├── text_units.parquet        # 文本单元表                               │
│    ├── communities.parquet       # 社区表                                   │
│    ├── community_reports.parquet # 社区报告表                               │
│    └── documents.parquet         # 文档表                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 各模型之间的关系图

```
                                    ┌─────────────┐
                                    │  Document   │
                                    │  (原始文档)  │
                                    └──────┬──────┘
                                           │ 1:N
                                           ▼
                                    ┌─────────────┐
                          ┌─────────│  TextUnit   │─────────┐
                          │         │ (文本单元)   │         │
                          │         └──────┬──────┘         │
                          │                │                │
                          │ N:M            │ N:M            │ N:M
                          ▼                ▼                ▼
                   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                   │   Entity    │  │ Relationship│  │  Covariate  │
                   │   (实体)    │  │   (关系)    │  │  (协变量)   │
                   └──────┬──────┘  └─────────────┘  └─────────────┘
                          │                │
                          │ N:M            │ N:M
                          ▼                ▼
                   ┌──────────────────────────────┐
                   │          Community           │
                   │           (社区)             │
                   │                              │
                   │  ┌──────────────────────┐   │
                   │  │   parent ◄── child   │   │  层级关系
                   │  └──────────────────────┘   │
                   └──────────────┬───────────────┘
                                  │ 1:1
                                  ▼
                          ┌─────────────────┐
                          │ CommunityReport │
                          │   (社区报告)    │
                          └─────────────────┘
```

**关系说明**：
| 关系 | 含义 |
|------|------|
| Document → TextUnit | 一个文档被切分成多个文本单元 |
| TextUnit → Entity | 一个文本单元可以包含多个实体 |
| Entity → TextUnit | 一个实体可以出现在多个文本单元 |
| Entity → Community | 一个实体可以属于多个社区（不同层级） |
| Community → CommunityReport | 每个社区有一份报告 |

---

## 10. 快速参考表

### 10.1 所有模型一览

| 模型 | 继承自 | 用途 | 关键字段 |
|------|--------|------|----------|
| `Identified` | - | 基础类 | id, short_id |
| `Named` | Identified | 有名字的基础类 | title |
| `Document` | Named | 原始文档 | text, text_unit_ids |
| `TextUnit` | Identified | 文本块 | text, entity_ids |
| `Entity` | Named | 实体 | type, description, rank |
| `Relationship` | Identified | 关系 | source, target, weight |
| `Covariate` | Identified | 协变量/声明 | subject_id, covariate_type |
| `Community` | Named | 社区 | level, parent, children |
| `CommunityReport` | Named | 社区报告 | summary, full_content, rank |

### 10.2 生活类比总结

| 模型 | 生活类比 |
|------|----------|
| Document | 一本书 / 一篇文章 |
| TextUnit | 书中的一页 / 一段 |
| Entity | 书中提到的人物、地点、公司 |
| Relationship | 人物之间的关系（朋友、合作、竞争） |
| Covariate | 书中的观点、声明、评价 |
| Community | 一群关系紧密的人物（朋友圈） |
| CommunityReport | 对这个朋友圈的总结描述 |

---

## 11. 总结

`data_model` 模块是 GraphRAG 的**数据基石**，定义了系统中所有核心概念的数据结构：

1. **输入层**：Document（文档）、TextUnit（文本单元）
2. **知识层**：Entity（实体）、Relationship（关系）、Covariate（协变量）
3. **聚合层**：Community（社区）、CommunityReport（社区报告）

这些模型共同构成了从"原始文本"到"结构化知识"再到"可检索报告"的完整数据流。

理解这些数据模型，是理解 GraphRAG 整个系统的基础。
