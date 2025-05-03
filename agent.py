from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
import os

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. 定义一个 mock Search 工具（你也可以改成调用真实 API）
def mock_search_tool(query: str) -> str:
    if "Oscar" in query:
        return "《奥本海默》获得2024年奥斯卡最佳影片。"
    elif "Oppenheimer" in query:
        return "《奥本海默》是诺兰导演的传记片，讲述原子弹之父的故事。"
    else:
        return "暂无结果。"

# 2. 把它注册为一个 LangChain Tool
search_tool = Tool(
    name="Search",
    func=mock_search_tool,
    description="用于查询事实性问题，比如获奖名单、人物介绍等"
)

# 3. 初始化 OpenAI LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# 4. 创建 Agent（ReAct 类型）
agent_executor = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 使用 ReAct Prompt 类型
    verbose=True  # 打印思考过程
)

# 5. 提问并执行 Agent
query = "2024 年谁获得了奥斯卡最佳影片？介绍一下这部电影。"
response = agent_executor.run(query)

print("\n✅ 最终答案：")
print(response)
