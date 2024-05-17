from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama



# def get_ollama_llm():
#     llm = Ollama(base_url="http://localhost:11434", model="llama2:7b-chat")
#     return llm

def get_ollama_chat_llm():
    llm_chat = ChatOllama(model="llama2:7b-chat",temperature=0.01,cache=True,verbose=True)
    return llm_chat


def get_ollama_code_llm():
    llm = Ollama(base_url="http://localhost:11434", model="codellama:7b")
    return llm

def get_ollama_python_code_llm():
    llm = Ollama(base_url="http://localhost:11434", model="codellama:7b-python")
    return llm