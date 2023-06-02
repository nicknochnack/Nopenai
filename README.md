# Building LLM Apps...without OpenAI 
Want to build LLM Apps...but without OpenAI dependencies? Well have I got the code for you my friend. In this project I walk through how to build a langchain x streamlit app using GPT4All. We start off with a simple app and then build up to a langchain PythonREPL agent. 

## See it live and in action 📺
[![Tutorial](https://i.imgur.com/qBoUX8m.jpg)](https://youtu.be/5JpPo-NOq9s 'Tutorial')

# Startup 🚀
1. Create a virtual environment `python -m venv nonopenai`
2. Activate it: 
   - Windows:`.\nonopenai\Scripts\activate`
   - Mac: `source nonopenai/bin/activate`
3. Install the GPT4All Installer using GUI based installers
   - Windows: https://gpt4all.io/installers/gpt4all-installer-win64.exe 
   - Mac: https://gpt4all.io/installers/gpt4all-installer-darwin.dmg
   - Ubuntu: https://gpt4all.io/installers/gpt4all-installer-linux.run
4. Download the required LLM models and take note of the PATH they're installed to
5. Clone this repo `git clone https://github.com/nicknochnack/Nopenai`
6. Go into the directory `cd NonOpenAI`
7. Install the required dependencies `pip install -r requirements.txt`
8. Update the path of the models in line 9 of `app.py` and line 5 of `app-chain.py`
9.  Start the python agent app by running `streamlit run app.py` or the chain app by running `streamlit run app-chain.py`  
10. Go back to my YouTube channel and like and subscribe 😉...no seriously...please! lol 
11. The comparison app can be started by running `streamlit run app-comparison.py` before you do that though, update the base ggml download path in line 16, e.g. `BASE_PATH = 'C:/Users/User/AppData/Local/nomic.ai/GPT4All/'` and openAI api key on line 18


# Other References 🔗
<p>-<a href="https://github.com/nomic-ai/gpt4all/tree/main">GPT4AllReference
</a>: mainly used to determine how to install the GPT4All library and references. Doco was changing frequently, at the time of coding this was the most up to date example of getting it running.</p>

# Who, When, Why?
👨🏾‍💻 Author: Nick Renotte <br />
📅 Version: 1.x<br />
📜 License: This project is licensed under the MIT License </br>

