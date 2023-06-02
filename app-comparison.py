# App dev framework
import streamlit as st
import os

# Import depdencies 
from langchain.llms import GPT4All, OpenAI
from langchain import PromptTemplate, LLMChain
from ctransformers.langchain import CTransformers

# Python toolchain imports 
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool


# Path to weights 
BASE_PATH = 'C:/Users/User/AppData/Local/nomic.ai/GPT4All/'

os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI API KEY HERE'

# Title 
st.title('🦜🔗 GPT For Y\'all')

with st.sidebar:
    st.info('This application allows you to use LLMs for a range of tasks. The selections displayed below leverage prompt formatting to streamline your ability to do stuff!')
    option = st.radio('Choose your task', ['Base Gen', 'Creative', 'Summarization', 'Few Shot', 'Python'])
    models =  [*list(os.listdir(BASE_PATH)), 'OpenAI']
    model = st.radio('Choose your model', models)
    st.write(model)

    if model != 'OpenAI': 
        PATH = f'{BASE_PATH}{model}'
        # Instance of llm
        llm = GPT4All(model=PATH, verbose=True, temp=0.1, n_predict=4096, top_p=.95, top_k=40, n_batch=9, repeat_penalty=1.1, repeat_last_n=1.1) 
        
    else: 
        llm = OpenAI(temperature=0.5)

if option=='Base Gen': 
    st.info('Use this application to perform standard chat generation tasks.')
    
    # Prompt box 
    prompt = st.text_input('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            As a creative agent, {action}
    """)
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)    

       
if option=='Creative': 
    st.info('Use this application to perform creative tasks like writing stories and poems.')
    
    # Prompt box 
    prompt = st.text_input('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            As a creative agent, {action}
    """)
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)

if option=='Summarization': 
    st.info('Use this application to perform summarization on blocks of text.')

    # Prompt box 
    prompt = st.text_area('Plug in your prompt here!')
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a passage to summarize. Using the prompt, provide a summarized response. 
            ### Prompt: 
            {action}
            ### Summary:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        # do this
        st.write(response)

if option=='Few Shot': 
    
    st.info('Pass through some examples of task-output to perform few-shot prompting.')
    # Examples for few shots 
    examples = st.text_area('Plug in your examples!')
    prompt = st.text_area('Plug in your prompt here!')

    template = PromptTemplate(input_variables=['action','examples'], template="""
        ### Instruction: 
        The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
        ### Examples: 
        {examples}
        ### Prompt: 
        {action}
        ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = chain.run(examples=examples, action=prompt) 
        print(response)
        # do this
        st.write(response)

if option=='Python': 
    st.info('Leverage a Python agent by using the PythonREPLTool inside of Langchain.')
    # Python agent
    python_agent = create_python_agent(llm=llm, tool=PythonREPLTool(), verbose=True)
    # Prompt text box
    prompt = st.text_input('Plug in your prompt here!')
    # if we hit enter  
    if prompt:
        # Pass the prompt to the LLM Chain
        response = python_agent.run(prompt) 

        # do this
        st.write(response)
