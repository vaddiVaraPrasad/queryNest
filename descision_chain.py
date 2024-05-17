from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
import normal_llm_chain
import serpAPI
import youtube
import models
import pprint
import re

def remove_special_chars(text):
  """
  Removes special characters from a string and returns the remaining text with commas replacing the special characters.

  Args:
      text: The input string.

  Returns:
      The string with special characters replaced by commas.
  """
  pattern = r"[^\w\s]"  # Matches characters except letters, numbers, and whitespace
  res = re.sub(pattern, " ", text)
  print(res)
  return res



def get_decision(user_query):
    llm_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""I am an AI Decision Maker Assistant that can help you with various tasks. Here's what I can do:

        * **YOUTUBE:** If your query is a YouTube URL or relates to YouTube content, I can provide a concise answer as "YOUTUBE".
        * **LLM:** If your question can be directly answered by LLM/me / programming coding related questions of python , LLms , Java , Java Script etc ,and any query is related to Assitance /clarification /AI responce/  . without needing external information and general information like wishes and normal general conversation , I'll return an "LLM" response.
        * **INTERNET:** If your query requires searching the internet or using current information, when user ask for Any Realtime or latest information , I'll provide an "INTERNET" response based on my findings.

        Example 
                                                  
        Query : "https://www.youtube.com/watch?v=3XL9vS8mUJk&t=1698s"
        Answer : "YOUTUBE"
        
        Query : "Who is the current prime minister of India"
        Answer : "INTERNET"
                                                  
        Query : "hi , how you doing "
        Answer : "LLM"

        your are ONLY ALLOWED to give only 3 types of responces as LLM , YOUTUBE , INTERNET                             
        """),
        HumanMessagePromptTemplate.from_template("Query:{query}")    
    ])
    chat_llm = models.get_ollama_chat_llm()
    llm_chain = LLMChain(prompt=llm_template , llm=chat_llm)
    result = llm_chain.invoke({"query":user_query})
    return result


def get_model_descision(user_query):
    llm_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""I am an AI Decision Maker Assistant that can help you with various tasks. Here's what I can do:

        If a question specifically involves PYTHON , python as a programming language,  any of PYTHON libraries (Flask, LLMs, AI, LangChain,Machine learning , deel learning , image processing , pands , numpy  etc.), or Python frameworks, need to respond only as  "CODE LLAMA2 PYTHON"
        If your question is related to programming languages  (Java,JavaScript, Ruby, C++, etc.), programming coding concepts , datastructues , algorithms ,web development, or any other programming concepts, need to  respond only as  "CODE LLAMA2".
        If your question can be answered using static  knowledge NOT RELATED TO ANY PROGRAMMING LANGUAGE(Python , Java , JavaScript) Only, need to  respond only as "GENERAL LLAMA2".
    
                                                    
        Example                                        
        
        Query : "Python Pandas Tutorial (Part 1): Getting Started with Data Analysis - Installation and Loading Data "
        Answer : "CODE LLAMA2 PYTHON"
        
        Query : "Explain with example about inheritance and oops concepts in java "
        Answer : "CODE LLAMA2"
                                                  
                                                  
        Query : "how to lord kishna ? waht is his role in mahabahrat"
        Answer : "GENERAL LLAMA2"
        
        your are ONLY  ALLOWED to give only 3 types of responces as "GENERAL LLAMA2" ,  "CODE LLAMA2" , "CODE LLAMA2 PYTHON"
        Your are Allowed to give ** ONLY ONE ** of ("GENERAL LLAMA2" ,  "CODE LLAMA2" , "CODE LLAMA2 PYTHON") for a Give Query 
                            
        """),
        HumanMessagePromptTemplate.from_template("Query:{query}")    
    ])
    chat_llm = models.get_ollama_chat_llm()
    llm_chain = LLMChain(prompt=llm_template , llm=chat_llm)
    result = llm_chain.invoke({"query":user_query})
    print("MODEL DECISION")
    print(result)
    return result

def get_model_name_modifed(models_decision):
    model_decision = models_decision["text"]
    res = remove_special_chars(model_decision)
    text = res.lower()  # Convert text to lowercase for case-insensitive matching
    pattern = r"(general\s+llama2)|(code\s+llama2(?:\s+python)?)"  # Regular expression for model names (case-insensitive)
    match = re.search(pattern, text)
    if match:
        model_name = match.group()
        if "general llama2" in model_name:
            # 
            print("GENERAL LLAMA2")
            return "GENERAL LLAMA2"
        elif "code llama2 python" in model_name:
            print("CODE LLAMA2 PYTHON")
            return "CODE LLAMA2 PYTHON"
        else:    
            print("CODE LLAMA2")
            return "CODE LLAMA2"
    else:
        print("GENERAL LLAMA2")
        return "GENERAL LLAMA2"
       

  

def get_models_modified(models_decision):
    model_decision = models_decision["text"]
    res = remove_special_chars(model_decision)
    text = res.lower()  # Convert text to lowercase for case-insensitive matching
    pattern = r"(general\s+llama2)|(code\s+llama2(?:\s+python)?)"  # Regular expression for model names (case-insensitive)
    match = re.search(pattern, text)

    if match:
        model_name = match.group()
        if "general llama2" in model_name:
            llm_general = models.get_ollama_chat_llm()
            print("GENERAL LLAMA2")
            return llm_general
        elif "code llama2 python" in model_name:
            llm_code = models.get_ollama_code_llm()
            print("CODE LLAMA2 PYTHON")
            return llm_code
        else:
            llm_python_code = models.get_ollama_python_code_llm()
            print("CODE LLAMA2")
            return llm_python_code
    else:
        llm_general = models.get_ollama_chat_llm()
        print("GENERAL LLAMA2")
        return llm_general


def get_models(models_decision):
    if "GENERAL LLAMA2" in models_decision["text"]:
        print("GENERAL LLAMA2")
        llm_general = models.get_ollama_chat_llm()
        return llm_general
    
    elif "CODE LLAMA2 PYTHON" in models_decision["text"]:
        llm_code = models.get_ollama_code_llm()
        print("CODE LLAMA2 PYTHON")
        return llm_code
    
    elif "CODE LLAMA2" in models_decision["text"]:
        print("CODE LLAMA2")
        llm_python_code = models.get_ollama_python_code_llm()
        return llm_python_code
   
    else:
        llm_general = models.get_ollama_chat_llm()
        print("GENERAL LLAMA2")
        return llm_general
        

def get_thenical_query(user_query):
    llm_template = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template("""
            I am a large language model (LLM) trained on a massive dataset of code, computer science concepts, Python, data structures, algorithms, competitive coding problems, and various programming languages. I can translate natural language query into technical query.

            I can Only able to translate the Natural query to thenical query 
            
            You are NOT Allowed to give explanation to User query , Supposed to Just translate the query 
            
            You need to return/reponde only Thenical Query to User , no need for extra explantion for it 

           
                                        
        """),
        HumanMessagePromptTemplate.from_template("Query:{query}")    
    ])
    chat_llm = models.get_ollama_chat_llm()
    llm_chain = LLMChain(prompt=llm_template , llm=chat_llm)
    result = llm_chain.invoke({"query":user_query})
    print("THENICAL QUERY IS")
    pprint.pprint(result)
    return result

def main():
    
    # conversational_chain  = normal_llm_chain.get_noraml_chain(llm_model=llm_model)
    # serpapi_chain = serpAPI.get_serpapi_chain(llm_model=llm_model)

    while True:
        user_query = input("Enter the user query, (press q to quit) ?      ")
        if user_query == "q":
            break
    
        



        decision = get_decision(user_query=user_query)
        print(decision)
        decision = decision["text"].split(" ")[-1]
        
        print( f"decision is {decision}")
        if "LLM" in decision:
            # use the noraml LLM chain to answer the question 
            model_decision = get_model_descision(user_query=user_query)
            llm_model = get_models(models_decision=model_decision)
            conversational_chain  = normal_llm_chain.get_noraml_chain(llm_model=llm_model)
            if get_model_name(models_decision=model_decision) == "CODE LLAMA2 PYTHON" or get_model_name(models_decision=model_decision) == "CODE LLAMA2":
                thenical_query = get_thenical_query(user_query)
                
            else:
                result = conversational_chain.invoke(input=user_query)
                pprint.pprint(result)
                pprint.pprint(result["response"])

        elif "INTERNET" in decision:
            content = serpAPI.get_serpapi_content(user_query=user_query)
            print("CONTENT \n")
            total_content  = f"user_query is {user_query}\n\nncontent is {content}\n"
            pprint.pprint(total_content)
            model_decision = get_model_descision(user_query=total_content)
            llm_model = get_models(models_decision=model_decision)
            serpapi_chain = serpAPI.get_serpapi_chain(llm_model=llm_model)
            result = serpapi_chain.invoke({"query":user_query , "context":content})
            # use the serpAPI to answer the question , by first provide the search result and then rest 
            pprint.pprint(result)
            pprint.pprint(result["text"])

        elif "YOUTUBE" in decision:
            # for yoube make it differntly
            documents = youtube.get_youtube_documents(user_query)
            print(f"Summary for {documents[0].metadata['title']}")
            model_decision = get_model_descision(user_query=documents[0].metadata['title'])
            llm_model = get_models(models_decision=model_decision)
            summary = youtube.get_summarization(document=documents[0],llm_model=llm_model)  
            pprint.pprint(summary)
            print("\n\n")
            youtube_chain = youtube.get_youtube_chain(documents=documents,llm_model=llm_model)
            while True:
                question = input("You can start asking question about Youtube Viedo ?? (press q to quit)    \n")
                if question == "q":
                    break
                answer = youtube_chain.invoke({"question":question})
                pprint.pprint(answer["answer"])
                pprint.pprint("\n\n")
        else:
            print("Sry to inform , i can't able to answer the question ")
            
     

def temp():
    while True:
        query = input("enter the query, (press q to quit) ?           ")
        if query == "q":
            break
        result = get_model_descision(user_query=query)
        print(result)
        pprint.pprint(result["text"])


if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    # main()
    remove_special_chars("/nvara /ttableun")
