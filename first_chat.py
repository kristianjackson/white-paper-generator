from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

TEMPERATURE = 0.1

chat = ChatOpenAI(temperature=TEMPERATURE)

result = chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
print(result)

batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]

result = chat.generate(batch_messages)

print(result)
print(result.llm_output['token_usage'])