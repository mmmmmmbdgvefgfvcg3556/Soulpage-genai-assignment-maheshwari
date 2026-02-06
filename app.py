import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# API Keys (set as environment variables)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Tool
search_tool = TavilySearchResults()

tools = [
    Tool(
        name="Web Search",
        func=search_tool.run,
        description="Search the web for current information"
    )
]

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Chat loop
print("Conversational Knowledge Bot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.run(user_input)
    print("Bot:", response)
