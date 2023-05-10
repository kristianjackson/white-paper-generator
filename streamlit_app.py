# Bring in deps
import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain.agents import load_tools, initialize_agent, AgentType

# App framework
st.title("🦜🔗 White Paper Generator")
prompt = st.text_input("What topic would you like the white paper to be about?")

# Prompt templates
# search_template = PromptTemplate(
#     input_variables=["topic"],
#     template="Search for 5 current news items or items of interest related to: {topic}"
# )

title_template = PromptTemplate(
    input_variables=["topic", "search_results"],
    template="Generate a white paper title about {topic} based on these findings from a search: {search_results}",
)

exec_summary_template = PromptTemplate(
    input_variables=["title", "wikipedia_research", "search_results"],
    template="Generate an executive summary for a white paper based on: {title} while leveraging wikipedia reserch: {wikipedia_research} and search results: {search_results}",
)

intro_paragraph_template = PromptTemplate(
    input_variables=["title", "wikipedia_research", "exec_summary", "search_results"],
    template="Generate an introductory paragrapgh for a white paper based on: {title} and {exec_summary} while leveraging wikipedia reserch: {wikipedia_research} and these search results: {search_results}",
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
exec_summary_memory = ConversationBufferMemory(
    input_key="title", memory_key="chat_history"
)
intro_paragraph_memory = ConversationBufferMemory(
    input_key="exec_summary", memory_key="chat_history"
)

# Llms
llm = OpenAI(temperature=0.1)
creative_llm = OpenAI(temperature=0.4)

# Chains
title_chain = LLMChain(
    llm=creative_llm,
    prompt=title_template,
    verbose=True,
    output_key="title",
    memory=title_memory,
)
exec_summary_chain = LLMChain(
    llm=creative_llm,
    prompt=exec_summary_template,
    verbose=True,
    output_key="exec_summary",
    memory=exec_summary_memory,
)
intro_paragraph_chain = LLMChain(
    llm=llm,
    prompt=intro_paragraph_template,
    verbose=True,
    output_key="intro_paragraph",
    memory=intro_paragraph_memory,
)


wiki = WikipediaAPIWrapper()
search = SerpAPIWrapper()
tools = load_tools(["serpapi"])
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)

# Show stuff to the screen if there's a prompt
if prompt:
    search_results = agent.run(
        f"Find content related to {prompt} for use in generating a white paper based on current information no older than 1 year from today's date.",
        verbose=True,
    )
    title = title_chain.run(topic=prompt, search_results=search_results)
    wiki_research = wiki.run(prompt)
    exec_summary = exec_summary_chain.run(
        title=title, wikipedia_research=wiki_research, search_results=search_results
    )
    into_paragraph = intro_paragraph_chain.run(
        title=title,
        wikipedia_research=wiki_research,
        exec_summary=exec_summary,
        search_results=search_results,
    )

    st.write(search_results)
    st.write(title)
    st.write("Executive Summary")
    st.write(exec_summary)
    st.write("Analysis")
    st.write(into_paragraph)
