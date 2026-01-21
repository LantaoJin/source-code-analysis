# GraphRAG graph_context 模块源码解析报告

> 模块路径: `graphrag/index/operations/summarize_communities/graph_context/`

## 1. 模块概述

`graph_context` 模块是 GraphRAG 社区摘要生成流程中的核心组件，负责**为图中的社区构建上下文字符串**，这些上下文将被传递给 LLM 生成社区报告。

### 1.1 模块结构

```
graph_context/
├── __init__.py          # 包初始化文件
├── context_builder.py   # 上下文构建器（核心）
└── sort_context.py      # 上下文排序与格式化
```

### 1.2 核心职责

| 文件 | 职责 |
|------|------|
| `context_builder.py` | 聚合节点、边、声明数据，按层级构建社区上下文 |
| `sort_context.py` | 按重要性排序上下文元素，格式化为字符串，处理 token 限制 |

---

## 2. 文件详解

### 2.1 `__init__.py`

```python
"""Package of context builders for graph-based reports."""
```

简单的包初始化文件，声明该包用于构建基于图的报告上下文。

---

### 2.2 `sort_context.py`

#### 2.2.1 模块说明

该文件负责将社区的原始上下文数据（节点、边、声明）**排序、去重、格式化**为可供 LLM 消费的字符串。

#### 2.2.2 核心函数

##### `sort_context()`

**签名：**
```python
def sort_context(
    local_context: list[dict],
    tokenizer: Tokenizer,
    sub_community_reports: list[dict] | None = None,
    max_context_tokens: int | None = None,
    ...  # 列名参数
) -> str
```

**功能：** 按度数（degree）降序排序上下文，优化性能并控制 token 数量。

**算法流程：**

```
1. 预处理本地上下文
   ├── 提取所有边 (edges)
   ├── 构建节点详情字典 (node_details)
   └── 构建声明详情字典 (claim_details)

2. 按边的度数降序排序

3. 增量构建上下文
   ├── 遍历排序后的边
   ├── 添加源节点和目标节点（去重）
   ├── 添加相关声明（去重）
   ├── 添加边本身（去重）
   └── 检查 token 是否超限，超限则停止

4. 返回上下文字符串
```

**输出格式：**
```
----Reports-----
community_id,full_content
1,"报告内容..."

-----Entities-----
short_id,title,description,...

-----Claims-----
short_id,subject,claim,...

-----Relationships-----
short_id,source,target,description,...
```

##### `parallel_sort_context_batch()`

**签名：**
```python
def parallel_sort_context_batch(
    community_df,
    tokenizer: Tokenizer,
    max_context_tokens,
    parallel=False
) -> pd.DataFrame
```

**功能：** 批量处理多个社区的上下文排序，支持并行执行。

**处理逻辑：**

| 模式 | 实现方式 |
|------|----------|
| 串行 (`parallel=False`) | 使用 `DataFrame.apply()` 逐行处理 |
| 并行 (`parallel=True`) | 使用 `ThreadPoolExecutor` 多线程处理 |

**输出列：**
- `CONTEXT_STRING`: 格式化后的上下文字符串
- `CONTEXT_SIZE`: 上下文的 token 数量
- `CONTEXT_EXCEED_FLAG`: 是否超过 token 限制

---

### 2.3 `context_builder.py`

#### 2.3.1 模块说明

该文件是上下文构建的**主控模块**，负责：
1. 从原始数据（节点、边、声明）构建本地上下文
2. 处理上下文超限情况（使用子社区报告替换）
3. 按层级组织上下文生成流程

#### 2.3.2 核心函数

##### `build_local_context()`

**签名：**
```python
def build_local_context(
    nodes, edges, claims,
    tokenizer: Tokenizer,
    callbacks: WorkflowCallbacks,
    max_context_tokens: int = 16_000,
) -> pd.DataFrame
```

**功能：** 为所有社区构建初始本地上下文。

**流程：**
```
获取所有层级 (levels)
    │
    ▼
┌─────────────────────────────┐
│  遍历每个层级              │
│  ├── _prepare_reports_at_level()
│  └── 添加 COMMUNITY_LEVEL 列
└─────────────────────────────┘
    │
    ▼
合并所有层级的 DataFrame
```

##### `_prepare_reports_at_level()`

**签名：**
```python
def _prepare_reports_at_level(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    claim_df: pd.DataFrame | None,
    tokenizer: Tokenizer,
    level: int,
    max_context_tokens: int = 16_000,
) -> pd.DataFrame
```

**功能：** 准备特定层级的社区报告上下文。

**详细流程：**

```
步骤 1: 筛选当前层级节点
        level_node_df = node_df[community_level == level]

步骤 2: 筛选层级内的边
        level_edge_df = edges where source AND target in nodes_set

步骤 3: 构建边详情
        edge_details = {short_id, source, target, description, degree}

步骤 4: 筛选相关声明
        level_claim_df = claims where subject in nodes_set

步骤 5: 合并边到节点
        ├── source_edges: 以节点为源的边
        ├── target_edges: 以节点为目标的边
        └── 合并到 merged_node_df

步骤 6: 聚合节点数据
        按 (TITLE, COMMUNITY_ID, COMMUNITY_LEVEL, NODE_DEGREE) 分组
        聚合 NODE_DETAILS 和 EDGE_DETAILS

步骤 7: 添加声明详情 (如果有)

步骤 8: 创建 ALL_CONTEXT 列
        包含: title, node_degree, node_details, edge_details, claim_details

步骤 9: 按社区分组
        community_df = grouped by COMMUNITY_ID with ALL_CONTEXT as list

步骤 10: 批量生成上下文字符串
         parallel_sort_context_batch()
```

##### `build_level_context()`

**签名：**
```python
def build_level_context(
    report_df: pd.DataFrame | None,
    community_hierarchy_df: pd.DataFrame,
    local_context_df: pd.DataFrame,
    tokenizer: Tokenizer,
    level: int,
    max_context_tokens: int,
) -> pd.DataFrame
```

**功能：** 为指定层级的社区准备最终上下文，处理超限情况。

**策略决策树：**

```
                    本地上下文
                        │
            ┌───────────┴───────────┐
            │                       │
        未超限                    超限
            │                       │
        直接使用            ┌───────┴───────┐
                           │               │
                      有子社区报告      无子社区报告
                           │               │
                   build_mixed_context  截断本地上下文
                           │
                   混合策略：
                   用子社区报告替换
                   大社区的本地上下文
```

#### 2.3.3 辅助函数

| 函数 | 功能 |
|------|------|
| `_drop_community_level()` | 删除 DataFrame 的 `COMMUNITY_LEVEL` 列 |
| `_at_level()` | 筛选指定层级的记录 |
| `_antijoin_reports()` | 返回不在报告中的记录（反连接） |
| `_sort_and_trim_context()` | 对上下文排序并截断到 token 限制 |
| `_build_mixed_context()` | 包装 `build_mixed_context` 用于 DataFrame 操作 |
| `_get_subcontext_df()` | 获取子社区的上下文和报告 |
| `_get_community_df()` | 为超限社区构建混合上下文 |

---

## 3. 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          输入数据                                   │
│                                                                     │
│    nodes (节点)      edges (边)       claims (声明)                │
└──────────┬─────────────────┬────────────────┬──────────────────────┘
           │                 │                │
           ▼                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    build_local_context()                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              _prepare_reports_at_level()                     │   │
│  │                                                              │   │
│  │  • 筛选当前层级数据                                          │   │
│  │  • 合并节点、边、声明                                        │   │
│  │  • 构建 ALL_CONTEXT 列                                       │   │
│  │  • 按社区分组                                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           parallel_sort_context_batch()                      │   │
│  │                                                              │   │
│  │  • 调用 sort_context() 排序                                  │   │
│  │  • 计算 CONTEXT_SIZE                                         │   │
│  │  • 设置 CONTEXT_EXCEED_FLAG                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
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

## 4. 关键设计模式

### 4.1 贪心压缩策略

当上下文超过 token 限制时，采用贪心算法：
1. 按社区大小降序排列
2. 优先用摘要报告替换大社区的详细上下文
3. 保留小社区的详细信息

### 4.2 增量构建

`sort_context()` 采用增量构建方式：
- 每添加一个元素就检查 token 数
- 超限时立即停止，避免浪费计算

### 4.3 层级处理

按社区层级（level）从底向上处理：
- 底层社区：只有本地上下文
- 上层社区：可以使用子社区的摘要报告

---

## 5. 依赖关系

```
sort_context.py
      │
      │ 被调用
      ▼
context_builder.py
      │
      │ 调用
      ▼
build_mixed_context.py (外部模块)
```

### 5.1 外部依赖

| 模块 | 用途 |
|------|------|
| `pandas` | 数据处理 |
| `graphrag.data_model.schemas` | 列名常量 |
| `graphrag.tokenizer.tokenizer` | Token 计算 |
| `graphrag.callbacks.workflow_callbacks` | 进度回调 |
| `graphrag.index.utils.dataframes` | DataFrame 工具函数 |

---

## 6. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_context_tokens` | 16,000 | 上下文最大 token 数 |
| `parallel` | False | 是否启用并行处理 |

---

## 7. 输出示例

### 7.1 上下文字符串格式

```csv
----Reports-----
community_id,full_content
101,"这是子社区101的摘要报告..."
102,"这是子社区102的摘要报告..."

-----Entities-----
short_id,title,type,description
1,Entity A,PERSON,"描述A..."
2,Entity B,ORGANIZATION,"描述B..."

-----Claims-----
short_id,subject,claim_type,description
10,Entity A,FACT,"声明内容..."

-----Relationships-----
short_id,source,target,description,degree
100,Entity A,Entity B,"关系描述...",5
```

---

## 8. 总结

`graph_context` 模块是 GraphRAG 社区摘要生成的核心基础设施，通过精心设计的排序、聚合和压缩策略，将图结构数据转换为适合 LLM 处理的文本上下文，同时有效控制 token 消耗。
