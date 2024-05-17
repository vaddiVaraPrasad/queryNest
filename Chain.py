from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import Docs
import models



def get_ollama_embedding():
    ollama_embeddings = OllamaEmbeddings()
    ollama_embeddings.show_progress = True
    ollama_embeddings.temperature = 0.1
    ollama_embeddings.model="llama2:7b-chat"
    return ollama_embeddings

def get_Conversational_memory(memory_key):
    message_history = ChatMessageHistory()
    conversational_memory = ConversationBufferWindowMemory(
        memory_key=memory_key,
        k=5,
        output_key="answer",
        chat_memory=message_history,
        return_messages=True
    )
    return conversational_memory

def get_mutiQueryRetriver(documents):
    llm = models.get_ollama_chat_llm()
    print(llm)
    ollama_embeddings = get_ollama_embedding()
    print(ollama_embeddings)
    print("Before Vectores!!!")
    # vector = await cl.make_async(FAISS.from_documents)(
    #     documents, ollama_embeddings
    # )
    vector = FAISS.from_documents(documents, ollama_embeddings)
    print(vector)
    print("after vector")
    retriever = vector.as_retriever(search_kwargs={"k": 3})
    print(retriever)
    multi_query_retriver = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )
    print(multi_query_retriver)
    return multi_query_retriver
    # return retriever

def create_conversational_retrival_chain(filePath,fileName):
    docs = Docs.get_docs(filepath=filePath,fileName=fileName)
    print(docs)

    if docs == []:
        return None
    else:
        print(len(docs))
        print("BEFORE RETRIVAL MULTI QUERY")
        retriver_multi_query = get_mutiQueryRetriver(docs)
        print(f"Done with retriver_multi query {retriver_multi_query}")
        memory = get_Conversational_memory("chat_history")
        chat_llm = models.get_ollama_chat_llm()
        print("Before chain")
        chain = ConversationalRetrievalChain.from_llm(
            chat_llm,
            chain_type="stuff",
            retriever=retriver_multi_query,
            memory=memory,
            return_source_documents=True,
            )
        print(f"after chain chain is {chain}")
        return chain 
