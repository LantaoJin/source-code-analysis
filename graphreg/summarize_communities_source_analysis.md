# GraphRAG summarize_communities 模块源码解析报告

> 模块路径: `graphrag/index/operations/summarize_communities/`

## 1. 模块概述

`summarize_communities` 模块是 GraphRAG 知识图谱索引流程中的**核心组件**，负责为图中检测到的社区生成结构化摘要报告。这些报告将被用于后续的查询和检索。

### 1.1 模块结构

```
summarize_communities/
├── __init__.py                      # 包初始化
├── build_mixed_context.py           # 混合上下文构建
├── community_reports_extractor.py   # LLM 报告提取器
├── explode_communities.py           # 社区展开工具
├── strategies.py                    # 报告生成策略
├── summarize_communities.py         # 主入口函数
├── typing.py                        # 类型定义
├── utils.py                         # 工具函数
│
├── graph_context/                   # 基于图的上下文构建
│   ├── __init__.py
│   ├── context_builder.py
│   └── sort_context.py
│
└── text_unit_context/               # 基于文本单元的上下文构建
    ├── __init__.py
    ├── context_builder.py
    ├── prep_text_units.py
    └── sort_context.py
```

### 1.2 功能概览

| 组件 | 功能 |
|------|------|
| 主流程 | 协调社区报告生成的完整流程 |
| 上下文构建 | 为每个社区准备 LLM 输入上下文 |
| LLM 提取 | 调用 LLM 生成结构化报告 |
| 策略模式 | 支持不同的报告生成策略 |

---

## 2. 核心流程架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           summarize_communities()                            │
│                              (主入口函数)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           上下文构建阶段                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐            │
│  │    graph_context/       │    │    text_unit_context/       │            │
│  │  (基于图结构)            │ OR │  (基于原始文本)              │            │
│  │                         │    │                             │            │
│  │  • build_local_context  │    │  • build_local_context      │            │
│  │  • build_level_context  │    │  • build_level_context      │            │
│  │  • sort_context         │    │  • sort_context             │            │
│  └─────────────────────────┘    └─────────────────────────────┘            │
│                    │                          │                             │
│                    └──────────┬───────────────┘                             │
│                               ▼                                             │
│                    build_mixed_context()                                    │
│                    (处理超限，混合子社区报告)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           报告生成阶段                                       │
│                                                                             │
│  strategies.py                  community_reports_extractor.py              │
│  ┌────────────────────┐         ┌────────────────────────────────┐         │
│  │ run_graph_         │         │  CommunityReportsExtractor     │         │
│  │ intelligence()     │────────►│                                │         │
│  │                    │         │  • 构建 Prompt                 │         │
│  │ • 加载 LLM 模型    │         │  • 调用 LLM                    │         │
│  │ • 调用 Extractor   │         │  • 解析 JSON 响应              │         │
│  └────────────────────┘         │  • 格式化输出                  │         │
│                                 └────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输出结果                                        │
│                                                                             │
│  CommunityReport (TypedDict)                                                │
│  ├── community: 社区ID                                                      │
│  ├── title: 报告标题                                                        │
│  ├── summary: 摘要                                                          │
│  ├── findings: 发现列表                                                     │
│  ├── rank: 评分                                                             │
│  └── full_content: 完整报告文本                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心文件详解

### 3.1 `summarize_communities.py` - 主入口

#### 主函数签名

```python
async def summarize_communities(
    nodes: pd.DataFrame,
    communities: pd.DataFrame,
    local_contexts,
    level_context_builder: Callable,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    strategy: dict,
    tokenizer: Tokenizer,
    max_input_length: int,
    async_mode: AsyncType = AsyncType.AsyncIO,
    num_threads: int = 4,
) -> pd.DataFrame
```

#### 处理流程

```python
# 1. 初始化
reports = []
strategy_exec = load_strategy(strategy["type"])

# 2. 构建社区层级结构
community_hierarchy = communities.explode("children")
                                 .rename({"children": "sub_community"})

# 3. 获取层级列表（从高到低）
levels = get_levels(nodes)  # [2, 1, 0] 降序

# 4. 按层级构建上下文
for level in levels:
    level_context = level_context_builder(
        reports,              # 已生成的报告
        community_hierarchy,  # 层级关系
        local_contexts,       # 本地上下文
        level,               # 当前层级
        ...
    )

# 5. 并发生成报告
for level_context in level_contexts:
    local_reports = await derive_from_rows(
        level_context,
        run_generate,        # 生成函数
        num_threads=4,       # 并发数
    )
    reports.extend(local_reports)

return pd.DataFrame(reports)
```

#### 关键设计

1. **层级处理顺序**：从高层级到低层级（降序），确保子社区报告先生成
2. **增量报告传递**：每层生成的报告传递给下一层，用于构建混合上下文
3. **并发执行**：使用 `derive_from_rows` 支持异步并发

---

### 3.2 `typing.py` - 类型定义

#### 核心类型

```python
class Finding(TypedDict):
    """单个发现"""
    summary: str        # 发现摘要
    explanation: str    # 详细解释

class CommunityReport(TypedDict):
    """社区报告"""
    community: str | int           # 社区ID
    title: str                     # 报告标题
    summary: str                   # 报告摘要
    full_content: str              # 完整文本内容
    full_content_json: str         # JSON 格式内容
    rank: float                    # 重要性评分
    level: int                     # 社区层级
    rating_explanation: str        # 评分解释
    findings: list[Finding]        # 发现列表

class CreateCommunityReportsStrategyType(str, Enum):
    """策略类型枚举"""
    graph_intelligence = "graph_intelligence"
```

#### 策略函数类型

```python
CommunityReportsStrategy = Callable[
    [
        str | int,           # community_id
        str,                 # input_context
        int,                 # level
        WorkflowCallbacks,   # callbacks
        PipelineCache,       # cache
        StrategyConfig,      # config
    ],
    Awaitable[CommunityReport | None],
]
```

---

### 3.3 `community_reports_extractor.py` - LLM 报告提取器

#### Pydantic 响应模型

```python
class FindingModel(BaseModel):
    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")

class CommunityReportResponse(BaseModel):
    title: str
    summary: str
    findings: list[FindingModel]
    rating: float
    rating_explanation: str
```

#### CommunityReportsExtractor 类

```python
class CommunityReportsExtractor:
    def __init__(
        self,
        model_invoker: ChatModel,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,  # 默认 1500
    ):
        ...

    async def __call__(self, input_text: str) -> CommunityReportsResult:
        # 1. 格式化 Prompt
        prompt = self._extraction_prompt.format(
            input_text=input_text,
            max_report_length=str(self._max_report_length),
        )

        # 2. 调用 LLM（JSON 模式）
        response = await self._model.achat(
            prompt,
            json=True,
            json_model=CommunityReportResponse,
        )

        # 3. 返回结构化结果
        return CommunityReportsResult(
            structured_output=response.parsed_response,
            output=self._get_text_output(report),
        )

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        """将结构化报告转换为 Markdown 格式"""
        report_sections = "\n\n".join(
            f"## {f.summary}\n\n{f.explanation}"
            for f in report.findings
        )
        return f"# {report.title}\n\n{report.summary}\n\n{report_sections}"
```

#### 输出格式

```markdown
# 社区报告标题

报告摘要内容...

## 发现1摘要

发现1的详细解释...

## 发现2摘要

发现2的详细解释...
```

---

### 3.4 `strategies.py` - 报告生成策略

#### run_graph_intelligence 函数

```python
async def run_graph_intelligence(
    community: str | int,
    input: str,
    level: int,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    args: StrategyConfig,
) -> CommunityReport | None:
    # 1. 加载 LLM 配置
    llm_config = LanguageModelConfig(**args["llm"])

    # 2. 获取或创建 LLM 模型
    llm = ModelManager().get_or_create_chat_model(
        name="community_reporting",
        model_type=llm_config.type,
        config=llm_config,
        callbacks=callbacks,
        cache=cache,
    )

    # 3. 运行提取器
    return await _run_extractor(llm, community, input, level, args)

async def _run_extractor(model, community, input, level, args):
    extractor = CommunityReportsExtractor(
        model,
        extraction_prompt=args.get("extraction_prompt"),
        max_report_length=args.get("max_report_length"),
    )

    results = await extractor(input)
    report = results.structured_output

    return CommunityReport(
        community=community,
        full_content=results.output,
        level=level,
        rank=report.rating,
        title=report.title,
        summary=report.summary,
        findings=[...],
        full_content_json=report.model_dump_json(indent=4),
    )
```

---

### 3.5 `build_mixed_context.py` - 混合上下文构建

#### 核心算法

```python
def build_mixed_context(
    context: list[dict],
    tokenizer: Tokenizer,
    max_context_tokens: int
) -> str:
    """
    当本地上下文超过 token 限制时，使用子社区报告替换。

    策略：贪心压缩
    1. 按上下文大小降序排列子社区
    2. 从最大的子社区开始，用报告替换本地上下文
    3. 每次替换后检查是否满足 token 限制
    4. 满足限制时停止替换
    """
    sorted_context = sorted(context, key=lambda x: x[CONTEXT_SIZE], reverse=True)

    substitute_reports = []
    final_local_contexts = []

    for idx, sub_community in enumerate(sorted_context):
        if exceeded_limit:
            # 用报告替换
            substitute_reports.append({
                COMMUNITY_ID: sub_community[SUB_COMMUNITY],
                FULL_CONTENT: sub_community[FULL_CONTENT],
            })

            # 保留剩余子社区的本地上下文
            remaining = [sorted_context[i][ALL_CONTEXT] for i in range(idx+1, len(sorted_context))]

            # 构建新上下文
            new_context_string = sort_context(
                local_context=remaining + final_local_contexts,
                sub_community_reports=substitute_reports,
            )

            if tokenizer.num_tokens(new_context_string) <= max_context_tokens:
                exceeded_limit = False
                context_string = new_context_string
                break

    return context_string
```

#### 设计思路

```
原始上下文（超限）:
┌─────────────────────────────────────┐
│ 子社区A (大)  │ 子社区B (中) │ 子社区C (小) │
│ 本地上下文    │ 本地上下文   │ 本地上下文   │
│ 5000 tokens   │ 3000 tokens  │ 1000 tokens  │
└─────────────────────────────────────┘
                    │
                    ▼ 贪心压缩
┌─────────────────────────────────────┐
│ 子社区A (大)  │ 子社区B (中) │ 子社区C (小) │
│ 摘要报告      │ 本地上下文   │ 本地上下文   │
│ 500 tokens    │ 3000 tokens  │ 1000 tokens  │
└─────────────────────────────────────┘
                    │
                    ▼ 如果仍超限，继续压缩
┌─────────────────────────────────────┐
│ 子社区A (大)  │ 子社区B (中) │ 子社区C (小) │
│ 摘要报告      │ 摘要报告     │ 本地上下文   │
│ 500 tokens    │ 400 tokens   │ 1000 tokens  │
└─────────────────────────────────────┘
```

---

### 3.6 `explode_communities.py` - 社区展开

```python
def explode_communities(
    communities: pd.DataFrame,
    entities: pd.DataFrame
) -> pd.DataFrame:
    """
    将社区的实体ID列表展开，与实体表连接。

    输入:
    communities: [community, level, entity_ids=[e1,e2,e3]]
    entities: [id, title, ...]

    输出:
    nodes: [id, title, ..., community, level]
    """
    # 展开 entity_ids 列
    community_join = communities.explode("entity_ids")

    # 与实体表连接
    nodes = entities.merge(
        community_join,
        left_on="id",
        right_on="entity_ids"
    )

    # 过滤无效社区
    return nodes[nodes[COMMUNITY_ID] != -1]
```

---

### 3.7 `utils.py` - 工具函数

```python
def get_levels(
    df: pd.DataFrame,
    level_column: str = schemas.COMMUNITY_LEVEL
) -> list[int]:
    """
    获取社区层级列表（降序）。

    返回: [2, 1, 0]  # 从高层级到低层级
    """
    levels = df[level_column].dropna().unique()
    levels = [int(lvl) for lvl in levels if lvl != -1]
    return sorted(levels, reverse=True)
```

---

## 4. 子模块详解

### 4.1 graph_context 子模块

基于图结构（节点、边、声明）构建上下文。

| 文件 | 功能 |
|------|------|
| `context_builder.py` | 聚合图数据，构建社区上下文 |
| `sort_context.py` | 按边的度数排序，格式化输出 |

**输出格式：**
```
----Reports-----
community_id,full_content
...

-----Entities-----
short_id,title,type,description
...

-----Claims-----
short_id,subject,claim_type,description
...

-----Relationships-----
short_id,source,target,description,degree
...
```

### 4.2 text_unit_context 子模块

基于原始文本块构建上下文。

| 文件 | 功能 |
|------|------|
| `context_builder.py` | 管理文本单元上下文 |
| `prep_text_units.py` | 计算文本单元的实体度数 |
| `sort_context.py` | 按实体度数排序，格式化输出 |

**输出格式：**
```
----REPORTS-----
community_id,full_content
...

-----SOURCES-----
id,text,entity_degree
...
```

### 4.3 两种上下文方式对比

| 特性 | graph_context | text_unit_context |
|------|---------------|-------------------|
| 数据来源 | 节点、边、声明 | 文本单元 |
| 排序依据 | 边的度数 | 实体度数和 |
| 信息粒度 | 结构化知识 | 原始文本 |
| 适用场景 | 知识密集查询 | 文本溯源 |

---

## 5. 配置参数

### 5.1 主函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_input_length` | int | - | 上下文最大 token 数 |
| `async_mode` | AsyncType | AsyncIO | 异步模式 |
| `num_threads` | int | 4 | 并发线程数 |

### 5.2 策略配置 (strategy dict)

```python
{
    "type": "graph_intelligence",
    "llm": {
        "type": "openai",
        "model": "gpt-4",
        ...
    },
    "extraction_prompt": "...",  # 可选，自定义 Prompt
    "max_report_length": 1500,   # 报告最大长度
}
```

### 5.3 上下文构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_context_tokens` | 16,000 | 上下文最大 token 数 |

---

## 6. 数据流完整视图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输入数据                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  nodes          communities       edges          claims       text_units    │
│  (实体)          (社区)            (关系)          (声明)       (文本块)      │
└──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────┘
       │              │              │              │              │
       │              │              └──────┬───────┘              │
       │              │                     │                      │
       ▼              ▼                     ▼                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        build_local_context()                                 │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │     graph_context              │  │     text_unit_context              │ │
│  │  _prepare_reports_at_level()   │  │  prep_text_units()                 │ │
│  │  • 筛选层级数据                │  │  • 计算实体度数                    │ │
│  │  • 合并节点/边/声明            │  │  • 展开文本单元                    │ │
│  │  • 按社区分组                  │  │  • 按社区分组                      │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                         │                              │                     │
│                         └──────────────┬───────────────┘                     │
│                                        ▼                                     │
│                              sort_context()                                  │
│                     • 按度数/实体度数降序排序                                │
│                     • 增量构建，控制 token                                   │
│                     • 格式化为 CSV 字符串                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         build_level_context()                                │
│                                                                              │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────────────┐ │
│  │ 未超限社区   │         │  超限社区    │         │  子社区报告          │ │
│  │              │         │              │         │                      │ │
│  │ 直接使用     │         │              │         │                      │ │
│  │ 本地上下文   │         │              │         │                      │ │
│  └──────┬───────┘         └──────┬───────┘         └──────────┬───────────┘ │
│         │                        │                            │              │
│         │                        ▼                            │              │
│         │              build_mixed_context()◀─────────────────┘              │
│         │              • 贪心压缩策略                                        │
│         │              • 用报告替换大社区                                    │
│         │              • 保留小社区详情                                      │
│         │                        │                                           │
│         └────────────────────────┴───────────────────────────────────────►  │
│                                  │                                           │
│                                  ▼                                           │
│                           最终上下文                                         │
│                     CONTEXT_STRING (格式化字符串)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          summarize_communities()                             │
│                                                                              │
│  for level in [2, 1, 0]:  # 从高到低                                        │
│      │                                                                       │
│      ├── 构建当前层级上下文                                                  │
│      │   level_context = level_context_builder(reports, ...)                │
│      │                                                                       │
│      ├── 并发生成报告                                                        │
│      │   ┌─────────────────────────────────────────────────────────────┐    │
│      │   │  derive_from_rows(level_context, run_generate)              │    │
│      │   │                                                             │    │
│      │   │  for each community in level_context:                       │    │
│      │   │      └── _generate_report()                                 │    │
│      │   │              └── strategy_exec() [run_graph_intelligence]   │    │
│      │   │                      └── CommunityReportsExtractor()        │    │
│      │   │                              └── LLM.achat(prompt)          │    │
│      │   └─────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      └── reports.extend(local_reports)                                       │
│                                                                              │
│  return pd.DataFrame(reports)                                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              输出结果                                        │
│                                                                              │
│  DataFrame[CommunityReport]                                                  │
│  ┌────────────┬─────────────┬─────────────┬───────────────┬────────────────┐│
│  │ community  │ title       │ summary     │ findings      │ full_content   ││
│  ├────────────┼─────────────┼─────────────┼───────────────┼────────────────┤│
│  │ 0          │ "社区0..."  │ "摘要..."   │ [{...}, ...]  │ "# 标题\n..."  ││
│  │ 1          │ "社区1..."  │ "摘要..."   │ [{...}, ...]  │ "# 标题\n..."  ││
│  │ ...        │ ...         │ ...         │ ...           │ ...            ││
│  └────────────┴─────────────┴─────────────┴───────────────┴────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键设计模式

### 7.1 策略模式

```python
# 定义策略类型
class CreateCommunityReportsStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"

# 加载策略
def load_strategy(strategy_type) -> CommunityReportsStrategy:
    match strategy_type:
        case CreateCommunityReportsStrategyType.graph_intelligence:
            return run_graph_intelligence
        case _:
            raise ValueError(f"Unknown strategy: {strategy_type}")
```

**扩展性**：添加新策略只需：
1. 在 `typing.py` 添加枚举值
2. 在 `strategies.py` 实现策略函数
3. 在 `load_strategy` 添加 case 分支

### 7.2 层级递归处理

```
Level 2 (最高层)
    │
    ├── 构建本地上下文
    ├── 生成报告 → reports[]
    │
    ▼
Level 1
    │
    ├── 构建上下文（可使用 Level 2 的报告）
    ├── 生成报告 → reports[]
    │
    ▼
Level 0 (最底层)
    │
    ├── 构建上下文（可使用 Level 1 的报告）
    └── 生成报告 → reports[]
```

### 7.3 贪心压缩

当上下文超限时，优先压缩大社区：
- 大社区的报告压缩比更高
- 小社区保留详细信息
- 平衡信息量和 token 消耗

### 7.4 增量构建

`sort_context` 使用增量方式构建上下文：
- 每添加一个元素检查 token
- 超限立即停止
- 避免构建过大上下文后再截断

---

## 8. 依赖关系图

```
                          summarize_communities.py
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             strategies.py     typing.py        utils.py
                    │
                    ▼
    community_reports_extractor.py
                    │
                    ▼
            ChatModel (LLM)


    build_mixed_context.py ◄────────────────────────┐
            │                                       │
            ▼                                       │
    ┌───────────────────┐         ┌─────────────────────────┐
    │  graph_context/   │         │  text_unit_context/     │
    │                   │         │                         │
    │  context_builder  │         │  context_builder        │
    │        │          │         │        │                │
    │        ▼          │         │        ▼                │
    │  sort_context     │         │  sort_context           │
    └───────────────────┘         │        │                │
                                  │        ▼                │
                                  │  prep_text_units       │
                                  └─────────────────────────┘
```

---

## 9. 外部依赖

| 模块 | 用途 |
|------|------|
| `pandas` | 数据处理 |
| `pydantic` | 数据验证和模型定义 |
| `graphrag.language_model` | LLM 调用 |
| `graphrag.cache.pipeline_cache` | 缓存管理 |
| `graphrag.callbacks` | 工作流回调 |
| `graphrag.tokenizer` | Token 计算 |
| `graphrag.data_model.schemas` | 列名常量 |
| `graphrag.prompts.index.community_report` | 默认 Prompt |

---

## 10. 使用示例

```python
from graphrag.index.operations.summarize_communities import summarize_communities
from graphrag.index.operations.summarize_communities.graph_context.context_builder import (
    build_local_context,
    build_level_context,
)

# 1. 构建本地上下文
local_contexts = build_local_context(
    nodes=nodes_df,
    edges=edges_df,
    claims=claims_df,
    tokenizer=tokenizer,
    callbacks=callbacks,
    max_context_tokens=16000,
)

# 2. 生成社区报告
reports_df = await summarize_communities(
    nodes=nodes_df,
    communities=communities_df,
    local_contexts=local_contexts,
    level_context_builder=build_level_context,
    callbacks=callbacks,
    cache=cache,
    strategy={
        "type": "graph_intelligence",
        "llm": {"type": "openai", "model": "gpt-4"},
        "max_report_length": 1500,
    },
    tokenizer=tokenizer,
    max_input_length=16000,
    num_threads=4,
)
```

---

## 11. 总结

`summarize_communities` 模块是 GraphRAG 的核心组件，实现了从图数据到结构化社区报告的完整流程：

1. **灵活的上下文构建**：支持基于图结构和基于文本单元两种方式
2. **智能压缩策略**：贪心算法处理 token 超限
3. **层级递归处理**：从高到低生成报告，支持报告复用
4. **可扩展的策略模式**：易于添加新的报告生成策略
5. **高效的并发执行**：支持多线程/异步处理

该模块将复杂的图结构知识转化为人类可读的报告，是 GraphRAG 知识增强检索的关键基础设施。
