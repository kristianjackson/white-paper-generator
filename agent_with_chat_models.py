from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

TEMPERATURE = 0.1

chat = ChatOpenAI(temperature=TEMPERATURE)

llm = OpenAI(temperature=TEMPERATURE)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

result = agent.run("Who is Olivia Wilde's boyfriend and what is his age?")
print(result)