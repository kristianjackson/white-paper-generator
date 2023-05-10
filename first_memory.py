from langchain import OpenAI, ConversationChain

TEMPERATURE = 0.1

llm = OpenAI(temperature=TEMPERATURE)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!!")
print(output)

output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)
