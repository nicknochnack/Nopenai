# App dev framework
import streamlit as st
import os
import time

# Import depdencies 
from langchain.llms import GPT4All, OpenAI
from langchain import PromptTemplate, LLMChain
from ctransformers.langchain import CTransformers

# Python toolchain imports 
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Path to weights 
BASE_PATH = '/home/shaker/models/GPT4All/'

os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI API KEY HERE'

from PIL import Image
import streamlit as st

# You can always call this function where ever you want

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo


# OR

# st.sidebar.image(add_logo(logo_path="your/logo/path", width=50, height=60)) 

st.set_page_config(
        page_title="Oracle Generation Tool with LLM",
)

# Title 
st.title('Oracle Generation Tool with LLM')

with st.sidebar:
    my_logo = add_logo(logo_path="/home/shaker/git/Nopenai/FbK_02.png", width=200, height=200)
    st.sidebar.image(my_logo)

#     st.info('This application allows you to use LLMs for a range of tasks. The selections displayed below leverage prompt formatting to streamline your ability to do stuff!')
    #option = st.radio('Choose your task', ['Base Gen', 'Creative', 'Summarization', 'Few Shot', 'Python'])
    option = st.radio('Choose your task', ['Chat', 'JUnit Test Gen', 'Assertion', 'Code Summary'])
    models =  [*list(os.listdir(BASE_PATH)), 'OpenAI']
    model = st.radio('Choose your model', models)
    #st.write(model)

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    
    if model != 'OpenAI': 
        PATH = f'{BASE_PATH}{model}'
        # Instance of llm
        #llm = GPT4All(model=PATH, backend=None, verbose=True, temp=0.1, n_predict=None, top_p=.95, top_k=40, n_batch=9, repeat_penalty=1.1, repeat_last_n=1.1) 
        
        # Verbose is required to pass to the callback manager
        #llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
        # If you want to use a custom model add the backend parameter
        # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
        llm = GPT4All(model=PATH, backend=None, callbacks=callbacks, verbose=True, temp=0.1, n_predict=4096, top_p=.95, top_k=40, n_batch=9, repeat_penalty=1.1, repeat_last_n=1.1)
        
    else: 
        llm = OpenAI(temperature=0.5)

if option=='Chat': 
    st.info('Use this application to chat with the LLM.')
    
    # Prompt box 
    prompt = st.text_area("What's on your mind today?")
    
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a question. Write an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        start_time = time.time()
        
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        
        end_time = time.time()
        execution_time = end_time - start_time

        st.info("Execution time: "+ str(round(execution_time, 2)) + " seconds")

        # do this
        st.write(response)

if option=='JUnit Test Gen': 
    st.info('Use this application to generate JUnit Tests.')
    
    # Prompt box 
    prompt = st.text_area('Plug in your prompt here!')
    
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a JAVA code. Write some JUnit tests with assertions as an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        start_time = time.time()
        
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        
        end_time = time.time()
        execution_time = end_time - start_time

        st.info("Execution time: "+ str(round(execution_time, 2)) + " seconds")

        # do this
        st.write(response)    

if option=='Assertion': 
    st.info('Use this application to generate Assertion for a test.')
    
    # Prompt box 
    prompt = st.text_area('Plug in your prompt here!')
    
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a JAVA code for JUnit tests. Write some assertions as an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a JAVA code for JUnit tests. Write some assertions as an appropriate response. Write what you are testing in comments and just write the assertion statement after that.
            ### Prompt: 
            {action}
            ### Response:""")
    
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a JAVA code for JUnit tests cases. Replace "generate assertion" by generating some assertions as an appropriate response in "// comment on what you are testing\n assert statement" format:

            ### Prompt: 
            {action}
            ### Response:""")
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        start_time = time.time()
        
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        
        end_time = time.time()
        execution_time = end_time - start_time

        st.info("Execution time: "+ str(round(execution_time, 2)) + " seconds")

        # do this
        st.write(response)     
    
if option=='Code Summary': 
    st.info('Use this application to explain code.')
    
    # Prompt box 
    prompt = st.text_area('Plug in your prompt here!')
    
    template = PromptTemplate(input_variables=['action'], template="""
            ### Instruction: 
            The prompt below is a JAVA code. Explain the code as an appropriate response.
            ### Prompt: 
            {action}
            ### Response:""")
    
    chain = LLMChain(llm=llm, prompt=template, verbose=True) 
    
    # if we hit enter  
    if prompt:
        start_time = time.time()
        
        # Pass the prompt to the LLM Chain
        response = chain.run(prompt) 
        
        end_time = time.time()
        execution_time = end_time - start_time

        st.info("Execution time: "+ str(round(execution_time, 2)) + " seconds")

        # do this
        st.write(response)   

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