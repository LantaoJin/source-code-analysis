# GraphRAG text_unit_context 模块源码解析报告

> 模块路径: `graphrag/index/operations/summarize_communities/text_unit_context/`

## 1. 模块概述

`text_unit_context` 模块是 GraphRAG 社区摘要生成的另一种上下文构建方式，与 `graph_context` 不同，它**基于文本单元（Text Units）构建上下文**，而非直接使用图的节点和边。

### 1.1 模块结构

```
text_unit_context/
├── __init__.py          # 包初始化文件
├── context_builder.py   # 上下文构建器（核心）
├── prep_text_units.py   # 文本单元预处理
└── sort_context.py      # 上下文排序与格式化
```

### 1.2 核心职责

| 文件 | 职责 |
|------|------|
| `context_builder.py` | 构建和管理社区的文本单元上下文 |
| `prep_text_units.py` | 预处理文本单元，计算实体度数 |
| `sort_context.py` | 按度数排序文本单元，格式化为字符串 |

### 1.3 与 graph_context 的区别

| 特性 | graph_context | text_unit_context |
|------|---------------|-------------------|
| 上下文来源 | 节点、边、声明 | 文本单元（原始文本块） |
| 排序依据 | 边的度数 | 文本单元关联实体的度数和 |
| 输出格式 | Entities, Claims, Relationships | SOURCES (文本单元) |
| 适用场景 | 结构化知识图谱 | 原始文本检索 |

---

## 2. 文件详解

### 2.1 `__init__.py`

```python
"""Package of context builders for text unit-based reports."""
```

简单的包初始化文件，声明该包用于构建基于文本单元的报告上下文。

---

### 2.2 `prep_text_units.py`

#### 2.2.1 模块说明

负责预处理文本单元数据，计算每个文本单元的"实体度数"（entity degree）。

#### 2.2.2 核心函数

##### `prep_text_units()`

**签名：**
```python
def prep_text_units(
    text_unit_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> pd.DataFrame
```

**功能：** 计算文本单元的度数并整合详情信息。

**返回列：** `[COMMUNITY_ID, TEXT_UNIT_ID, ALL_DETAILS]`

**算法流程：**

```
步骤 1: 展开节点的 TEXT_UNIT_IDS 列
        node_df.explode(TEXT_UNIT_IDS)

        原始: node_1 -> [tu_1, tu_2]
        展开: node_1 -> tu_1
              node_1 -> tu_2

步骤 2: 按 (COMMUNITY_ID, TEXT_UNIT_ID) 分组
        聚合 NODE_DEGREE (求和)

        结果: 每个文本单元在每个社区的总度数

步骤 3: 合并到文本单元数据
        text_unit_df.merge(text_unit_degrees)

步骤 4: 构建 ALL_DETAILS 字典
        {
            SHORT_ID: 文本单元短ID,
            TEXT: 原始文本内容,
            ENTITY_DEGREE: 关联实体度数和
        }
```

**设计意图：**

实体度数越高，表示该文本单元包含的实体在图中越重要，应该在上下文中获得更高优先级。

---

### 2.3 `sort_context.py`

#### 2.3.1 模块说明

负责将文本单元列表按重要性排序，并格式化为上下文字符串。

#### 2.3.2 核心函数

##### `get_context_string()`

**签名：**
```python
def get_context_string(
    text_units: list[dict],
    sub_community_reports: list[dict] | None = None,
) -> str
```

**功能：** 将结构化数据拼接为上下文字符串。

**输出格式：**
```
----REPORTS-----
community_id,full_content
101,"子社区报告内容..."

-----SOURCES-----
id,text,entity_degree
1,"这是文本单元1的内容...",15
2,"这是文本单元2的内容...",12
```

**处理逻辑：**

1. **报告部分** (第22-39行)：
   - 过滤无效报告（空 COMMUNITY_ID）
   - 去重并转换为 CSV 格式
   - 处理 float 类型的 ID 转换

2. **文本单元部分** (第41-53行)：
   - 过滤无效记录（空 id）
   - 去重并转换为 CSV 格式
   - 添加 `-----SOURCES-----` 标题

##### `sort_context()`

**签名：**
```python
def sort_context(
    local_context: list[dict],
    tokenizer: Tokenizer,
    sub_community_reports: list[dict] | None = None,
    max_context_tokens: int | None = None,
) -> str
```

**功能：** 按实体度数降序排序文本单元，生成上下文字符串。

**算法流程：**

```
1. 按 ENTITY_DEGREE 降序排序
   sorted_text_units = sorted(local_context, key=entity_degree, reverse=True)

2. 增量构建上下文
   for record in sorted_text_units:
       current_text_units.append(record)
       if max_context_tokens:
           new_context = get_context_string(current_text_units)
           if num_tokens(new_context) > max_context_tokens:
               break  # 停止添加
           context_string = new_context

3. 返回上下文字符串
   - 如果构建成功，返回 context_string
   - 如果为空（极端情况），返回完整排序后的结果
```

---

### 2.4 `context_builder.py`

#### 2.4.1 模块说明

主控模块，负责构建社区的文本单元上下文，处理超限情况。

#### 2.4.2 核心函数

##### `build_local_context()`

**签名：**
```python
def build_local_context(
    community_membership_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    node_df: pd.DataFrame,
    tokenizer: Tokenizer,
    max_context_tokens: int = 16000,
) -> pd.DataFrame
```

**功能：** 为所有社区构建初始本地上下文。

**输入数据说明：**

`community_membership_df` 包含列：
- `COMMUNITY_ID`: 社区ID
- `COMMUNITY_LEVEL`: 社区层级
- `ENTITY_IDS`: 社区内实体ID列表
- `RELATIONSHIP_IDS`: 社区内关系ID列表
- `TEXT_UNIT_IDS`: 社区内文本单元ID列表

**处理流程：**

```
步骤 1: 预处理文本单元
        prep_text_units(text_units_df, node_df)
        计算每个文本单元的实体度数

步骤 2: 展开社区成员关系
        explode(TEXT_UNIT_IDS)
        将社区的文本单元列表展开为多行

步骤 3: 合并文本单元详情
        merge(prepped_text_units_df)

步骤 4: 构建 ALL_CONTEXT 列
        {
            "id": short_id,
            "text": 文本内容,
            "entity_degree": 实体度数
        }

步骤 5: 按社区分组聚合
        groupby([COMMUNITY_ID, COMMUNITY_LEVEL])
        agg({ALL_CONTEXT: list})

步骤 6: 生成上下文字符串
        sort_context() 排序并格式化

步骤 7: 计算上下文大小和超限标志
        CONTEXT_SIZE = num_tokens(CONTEXT_STRING)
        CONTEXT_EXCEED_FLAG = CONTEXT_SIZE > max_context_tokens
```

##### `build_level_context()`

**签名：**
```python
def build_level_context(
    report_df: pd.DataFrame | None,
    community_hierarchy_df: pd.DataFrame,
    local_context_df: pd.DataFrame,
    level: int,
    tokenizer: Tokenizer,
    max_context_tokens: int = 16000,
) -> pd.DataFrame
```

**功能：** 为指定层级的社区准备最终上下文，处理超限情况。

**策略决策树：**

```
                        报告是否存在?
                             │
              ┌──────────────┴──────────────┐
              │                             │
           无报告                         有报告
              │                             │
       ┌──────┴──────┐               排除已有报告的社区
       │             │                      │
    未超限        超限               ┌──────┴──────┐
       │             │               │             │
    直接返回    截断上下文         未超限        超限
                                     │             │
                                  直接返回    ┌────┴────┐
                                             │         │
                                        有子社区    无子社区
                                             │         │
                                    build_mixed    截断上下文
                                     _context
```

**详细处理逻辑：**

**情况 1: 无报告可替换** (第100-129行)
- 发生在层级结构底层
- 未超限的社区直接返回
- 超限的社区截断本地上下文

**情况 2: 有报告可替换** (第131-235行)

1. **筛选当前层级** (第131-144行)：
   - 过滤当前层级的社区
   - 排除已有报告的社区（使用 outer merge + indicator）

2. **分离有效/无效上下文** (第145-152行)：
   - `valid_context_df`: 未超限
   - `invalid_context_df`: 超限需处理

3. **获取子社区信息** (第159-170行)：
   - 获取 level+1 层的报告
   - 合并子社区的本地上下文和报告

4. **构建混合上下文** (第173-212行)：
   - 收集所有子社区的上下文
   - 构建 `{SUB_COMMUNITY, ALL_CONTEXT, FULL_CONTENT, CONTEXT_SIZE}` 结构
   - 调用 `build_mixed_context()` 生成混合上下文

5. **处理剩余无效记录** (第214-231行)：
   - 没有子社区报告可替换的社区
   - 直接截断本地上下文

---

## 3. 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          输入数据                                   │
│                                                                     │
│  community_membership_df    text_units_df      node_df             │
│  (社区成员关系)              (文本单元)         (节点数据)          │
└──────────┬─────────────────────┬────────────────┬──────────────────┘
           │                     │                │
           │                     ▼                │
           │        ┌────────────────────────┐    │
           │        │   prep_text_units()    │◀───┘
           │        │                        │
           │        │  • 展开节点的文本单元  │
           │        │  • 计算实体度数        │
           │        │  • 构建 ALL_DETAILS    │
           │        └───────────┬────────────┘
           │                    │
           ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    build_local_context()                            │
│                                                                     │
│  • 展开社区的 TEXT_UNIT_IDS                                        │
│  • 合并文本单元详情                                                 │
│  • 构建 ALL_CONTEXT 列                                              │
│  • 按社区分组聚合                                                   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    sort_context()                            │   │
│  │                                                              │   │
│  │  • 按 ENTITY_DEGREE 降序排序                                 │   │
│  │  • 增量构建，检查 token 限制                                 │   │
│  │  • 调用 get_context_string() 格式化                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  计算 CONTEXT_SIZE 和 CONTEXT_EXCEED_FLAG                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     build_level_context()                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ valid_context │    │invalid_context│    │ sub_community_reports│  │
│  │   (未超限)    │    │   (超限)      │    │    (子社区报告)      │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │               │
│         │                   ▼                       │               │
│         │         ┌─────────────────────┐          │               │
│         │         │ build_mixed_context │◀─────────┘               │
│         │         │   (混合策略)        │                          │
│         │         └─────────┬───────────┘                          │
│         │                   │                                       │
│         └───────────────────┼───────────────────────────────────►  │
│                             │                                       │
│                             ▼                                       │
│                    最终上下文 DataFrame                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM 生成社区报告   │
                    └─────────────────────┘
```

---

## 4. 关键数据结构

### 4.1 ALL_DETAILS (文本单元详情)

```python
{
    "short_id": 1,           # 文本单元短ID
    "text": "原始文本...",    # 文本内容
    "entity_degree": 15      # 关联实体度数和
}
```

### 4.2 ALL_CONTEXT (上下文记录)

**本地上下文格式：**
```python
{
    "id": 1,                  # 文本单元ID
    "text": "原始文本...",     # 文本内容
    "entity_degree": 15       # 实体度数
}
```

**混合上下文格式（用于超限处理）：**
```python
{
    "sub_community": "101",           # 子社区ID
    "all_context": [...],             # 子社区的本地上下文
    "full_content": "报告内容...",     # 子社区报告
    "context_size": 1500              # 上下文大小
}
```

### 4.3 输出 DataFrame 结构

| 列名 | 类型 | 说明 |
|------|------|------|
| `COMMUNITY_ID` | str | 社区ID |
| `COMMUNITY_LEVEL` | int | 社区层级 |
| `ALL_CONTEXT` | list[dict] | 原始上下文数据 |
| `CONTEXT_STRING` | str | 格式化后的上下文字符串 |
| `CONTEXT_SIZE` | int | 上下文的 token 数量 |
| `CONTEXT_EXCEED_FLAG` | bool | 是否超过 token 限制 |

---

## 5. 输出格式示例

### 5.1 纯文本单元上下文

```csv
-----SOURCES-----
id,text,entity_degree
1,"在2023年的技术峰会上，微软CEO萨提亚·纳德拉宣布了...",25
2,"OpenAI与微软的合作始于2019年，双方在人工智能领域...",18
3,"Azure云平台是微软的核心业务之一，提供了...",12
```

### 5.2 混合上下文（含子社区报告）

```csv
----REPORTS-----
community_id,full_content
101,"社区101的摘要报告：该社区主要涉及微软的云计算业务..."
102,"社区102的摘要报告：该社区聚焦于AI合作伙伴关系..."

-----SOURCES-----
id,text,entity_degree
5,"纳德拉表示，Azure的增长速度超过了预期...",8
6,"在最新的财报中，微软的云业务收入达到...",6
```

---

## 6. 与 graph_context 的对比

### 6.1 架构对比

```
graph_context                    text_unit_context
─────────────────                ─────────────────
nodes + edges + claims           text_units + nodes
       │                                │
       ▼                                ▼
按边的度数排序                    按实体度数排序
       │                                │
       ▼                                ▼
Entities/Claims/Relationships    SOURCES (原始文本)
```

### 6.2 使用场景

| 场景 | 推荐方式 |
|------|----------|
| 需要精确的实体关系 | graph_context |
| 需要原始文本引用 | text_unit_context |
| 知识密集型问答 | graph_context |
| 长文档摘要 | text_unit_context |

---

## 7. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_context_tokens` | 16,000 | 上下文最大 token 数 |

---

## 8. 依赖关系

```
prep_text_units.py
      │
      │ 被调用
      ▼
context_builder.py ──────► sort_context.py
      │
      │ 调用
      ▼
build_mixed_context.py (外部模块)
```

### 8.1 外部依赖

| 模块 | 用途 |
|------|------|
| `pandas` | 数据处理 |
| `graphrag.data_model.schemas` | 列名常量 |
| `graphrag.tokenizer.tokenizer` | Token 计算 |
| `build_mixed_context` | 混合上下文构建 |

---

## 9. 总结

`text_unit_context` 模块提供了一种基于原始文本的上下文构建方式：

1. **核心思想**：保留原始文本片段，而非仅使用提取的实体和关系
2. **排序策略**：按文本单元关联的实体度数排序，高度数意味着更重要
3. **压缩策略**：与 graph_context 共享 `build_mixed_context`，使用子社区报告替换大社区的详细上下文
4. **适用场景**：需要原始文本引用或上下文溯源的应用

该模块与 `graph_context` 形成互补，用户可根据具体需求选择合适的上下文构建方式。
