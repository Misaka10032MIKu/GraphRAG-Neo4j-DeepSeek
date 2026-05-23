import os
from dotenv import load_dotenv  # 第三方环境管理工具

# 🚨【核心修正 1】：将 Neo4jGraph 和 GraphCypherQAChain 全部从最新的统一生态包中引入，解决 Pydantic 校验冲突报错
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量（确保里面有 NEO4J_PASSWORD 和 DEEPSEEK_API_KEY）
load_dotenv()


def main():
    # 1. 严格的环境变量安全卫士检查
    password = os.getenv("NEO4J_PASSWORD")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")  # 读取你配置的 DeepSeek 键名

    if not password or not deepseek_key:
        print("❌ 错误：.env 文件中缺少 NEO4J_PASSWORD 或 DEEPSEEK_API_KEY！")
        print("请检查你的 .env 文件内容是否长成这样：")
        print('NEO4J_PASSWORD="你的密码"')
        print('DEEPSEEK_API_KEY="sk-..."')
        return

    # 2. 初始化 Neo4j 连接（使用 127.0.0.1 彻底绕过 Windows 的 IPv6 积极拒绝连接陷阱）
    graph = Neo4jGraph(
        url="bolt://127.0.0.1:7687",
        username="neo4j",
        password=password
    )

    # 3. 自动自省（Introspection）：刷新并打印数据库当前的元数据网状结构
    graph.refresh_schema()
    print("--- 成功连接图数据库 ---")
    print("当前图谱 schema:\n", graph.schema)
    print("-" * 30 + "\n")

    # 4. 初始化 LLM（通过 OpenAI 兼容类完美桥接到 DeepSeek-V3）
    llm = ChatOpenAI(
        model="deepseek-chat",  # 官方对话模型
        openai_api_key=deepseek_key,  # 你的 DeepSeek 独家令牌
        openai_api_base="https://api.deepseek.com/v1",  # 强制改变网络路由，将请求精准发送给 DeepSeek 服务器
        temperature=0  # 严格将创造力限制为 0，确保生成的 Cypher 语句符合数据库严谨的语法标准
    )
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain_neo4j import GraphCypherQAChain

    # 定义 Few-shot 示例库（可以根据业务无限追加）
    examples = [  #python列表，每个实例都是一个字典，存放键值对（问题及对应的cypher模板）
        {
            "question": "供应商 A 的哪些零件间接导致了 Q3 召回批次的问题？",
            "cypher": "MATCH (s:Supplier)-[:供应]->(p:Part)-[:用于|涉及召回*1..3]->"
                      "(r:Recall {{batch: 'Q3'}}) WHERE s.name CONTAINS 'A' RETURN DISTINCT p.name"
        },
        {
            "question": "有哪些核心构件批次涉及到了高温疲劳裂纹的召回？",
            "cypher": "MATCH (b:Batch {{type: '核心构件批次'}})-[:涉及召回]->"
                      "(r:Recall {{reason: '高温疲劳裂纹'}}) RETURN b.id"
        }
    ]

    # 配置单个示例的组装模板
    example_prompt = PromptTemplate(  #它是一个 PromptTemplate 实例（提示词模板），为大模型提供统一的展现格式
        input_variables=["question", "cypher"],
        template="用户问题: {question}\n生成的Cypher: {cypher}"
    )

    # 构造大模型看图写代码的“总指挥提示词”（System Prompt）
    CYPHER_GENERATION_TEMPLATE = """Task:Generate a Cypher statement to answer the following user question.
    You must use only the provided relationship types and properties from the schema.
    Do not use any other relationships or properties that are not mentioned.
    注意：生成的 Cypher 语句请直接输出，不要包裹在 ```cypher ... ``` 这样的 Markdown 代码块中，也不要包含任何解释性的文字。

    下面是数据库的真实 Schema 结构供你参考:
    {schema}

    下面是几个正确的查询示例，请严格模仿它们的逻辑和字符匹配方式:
    """

    # 组装成完整的 FewShot 模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=CYPHER_GENERATION_TEMPLATE,
        suffix="用户问题: {question}\n生成的Cypher:",
        input_variables=["question", "schema"]  # 这两个变量会由链条在运行时自动注入
    )


    # 5. 构建全新的智能化图谱查询问答链（GraphCypherQAChain）
    # 【最终修改】：加入 allow_dangerous_requests=True 协议开关
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,  # 开启内视：在控制台打印大模型思考、写 Cypher 的全过程
        return_intermediate_steps=True,  # 抓取中间数据：允许我们截获生成的原生 Cypher 语句进行调试，同意本地调试风险
        allow_dangerous_requests=True,  # 🔒 显式签署安全免责协议，允许链条执行生成的 Cypher
        cypher_prompt=few_shot_prompt
    )

    # 6. 开启自然语言图谱多跳（Multi-hop）推理检索
    query_text = "供应商 A 的哪些零件间接导致了 Q3 召回批次的问题？"
    print(f"正在向系统提问: '{query_text}'\n")

    # 链条内部开始联动：自然语言 -> 大模型生成 Cypher -> 数据库执行 -> 大模型组织人话
    result = chain.invoke({"query": query_text})

    # 7. 优雅地拆包并展示结果
    print("\n" + "=" * 20 + " 检索报告 " + "=" * 20 + "\n")

    # 从拦截到的中间步骤中把生成的原生 Cypher 代码揪出来展示
    if "intermediate_steps" in result and len(result["intermediate_steps"]) > 0:
        # 新版 langchain-neo4j 的返回结构通常在第一个元素的 query 键中
        generated_cypher = result["intermediate_steps"][0].get("query")
        print("🤖 DeepSeek 为你量身生成的 Cypher 查询语句:\n", generated_cypher)
        print("-" * 50)

    # 打印最终反馈人类通俗易懂的自然语言结论
    print("📝 最终答案:", result["result"])
    print("\n" + "=" * 48)


if __name__ == "__main__":
    main()