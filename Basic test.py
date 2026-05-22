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

    # 5. 构建全新的智能化图谱查询问答链（GraphCypherQAChain）
    # 🛠️ 【最终修改】：加入 allow_dangerous_requests=True 协议开关
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,  # 开启内视：在控制台打印大模型思考、写 Cypher 的全过程
        return_intermediate_steps=True,  # 抓取中间数据：允许我们截获生成的原生 Cypher 语句进行调试，同意本地调试风险
        allow_dangerous_requests=True  # 🔒 显式签署安全免责协议，允许链条执行生成的 Cypher
    )

    # 6. 开启自然语言图谱多跳（Multi-hop）推理检索
    query_text = "名称包含'供应商 A'的供应商，其供应的哪些零件间接导致了 Q3 召回批次的问题？"
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