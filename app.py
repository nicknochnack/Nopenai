import streamlit as st 

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

PATH = 'C:/Users/User/AppData/Local/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin'
llm = GPT4All(model=PATH, verbose=True)

agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

st.title('🦜🔗 GPT For Y\'all')

prompt = st.text_input('Enter your prompt here!')

if prompt: 
    response = agent_executor.run(prompt)
    st.write(response)

