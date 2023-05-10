from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

TEMPERATURE = 0

llm = OpenAI(temperature=TEMPERATURE)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)

result = agent.run(
    "What was the temperature in King George, VA yesterday in Fahrenheit? What is that number raised to the 0.023 power?"
)

print(result)
