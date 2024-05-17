from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# import Chain
import models

def normal_Chain():
    llm = models.get_ollama_chat_llm()
    memory = ConversationBufferMemory(memory_key="history")
    # Create a ConversationChain object
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
    )
    result = conversation_chain.invoke(input="Hi")
    print(result)
    result = conversation_chain.invoke(input="My name is Varaprasad")
    print(result)
    result = conversation_chain.invoke(input="Can you tell me my name once pls ?")
    print(result)

def get_noraml_chain(llm_model):
    # llm = models.get_ollama_chat_llm()
    memory = ConversationBufferMemory(memory_key="history")
    # Create a ConversationChain object
    conversation_chain = ConversationChain(
        llm=llm_model,
        memory=memory,
    )
    return conversation_chain


if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    normal_Chain()