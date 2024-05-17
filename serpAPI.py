from langchain_community.utilities import SerpAPIWrapper
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
# import Chain
import models


def internet_search(user_query):
    llm_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""Your a AI Search Assistent that use the Content give And answer to User Query within 50 words!
        if query Can not be answer by content give hostely respond as "I don't Know"
        """),
        HumanMessagePromptTemplate.from_template("{context}\nQuery:{query}")    
    ])
    search = SerpAPIWrapper()
    chat_llm = models.get_ollama_chat_llm()
    llm_chain = LLMChain(prompt=llm_template , llm=chat_llm)
    context = search.run(user_query)
    print("CONTENT IS \n\n")
    print(context)
    print("\n\n")
    result = llm_chain.run({"query":user_query , "context":context})
    print(result)



def get_serpapi_chain(llm_model):
    llm_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""Your a AI Search Assistent that use the Content give And answer to User Query within 50 words!
        if query Can not be answer by content give hostely respond as "I don't Know"
        """),
        HumanMessagePromptTemplate.from_template("{context}\nQuery:{query}")    
    ])
    # chat_llm = models.get_ollama_chat_llm()
    llm_chain = LLMChain(prompt=llm_template , llm=llm_model)
    return llm_chain

def get_serpapi_content(user_query):
    search = SerpAPIWrapper()
    context = search.run(user_query)
    return context

if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    print("loaded env varibles ")
    internet_search("compare the wheather b/w bangalore and tadepallgudem  ?")