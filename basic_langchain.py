from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

TEMPERATURE = 0.1

llm = OpenAI(temperature=TEMPERATURE)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("dingy used clothing")

print(result)

result = chain.run("sports cars")

print(result)

result = chain.run("used XBox gear")

print(result)