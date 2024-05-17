from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import Chain
import models
import Docs


def convert_seconds_to_minutes_and_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} min, {remaining_seconds} sec"

def check_video_size(length):
    if length < 600:
        return "SMALL"
    else:
        return "LARGE"

# def get_ollama_chat_llm():
#     llm_chat = ChatOllama(model="llama2:7b-chat",temperature=0.1,cache=True,verbose=True)
#     return llm_chat

# def get_ollama_llm():
#     llm = Ollama(base_url="http://localhost:11434", model="llama2:7b-chat")
#     return llm

# def get_ollama_embedding():
#     ollama_embeddings = OllamaEmbeddings()
#     ollama_embeddings.show_progress = True
#     ollama_embeddings.temperature = 0.1
#     ollama_embeddings.model="llama2:7b-chat"
#     return ollama_embeddings

def get_youtube_documents(url):
    loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
    documents = loader.load()
    print(f"Found video  {documents[0].metadata['title']} from {documents[0].metadata['author']}  with {convert_seconds_to_minutes_and_seconds(documents[0].metadata['length'])} ")
    return documents


def get_summarization(document,llm_model):
    # llm = models.get_ollama_chat_llm()
    if check_video_size(document .metadata["length"]) == "SMALL":
        short_chain = load_summarize_chain(llm=llm_model , chain_type="stuff",verbose=True)
        summary = short_chain.run([document])
        return summary
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,)
        docs = text_splitter.split_documents(documents=[document])
        long_chain = load_summarize_chain(llm=llm_model,chain_type="map_reduce",verbose=True)
        summary = long_chain.run(docs)
        return summary 


def chat(documents):
    docs = Docs.get_youtube_docs(document=documents[0])
    ollama_embeddings = Chain.get_ollama_embedding()
    chat_llm = models.get_ollama_chat_llm()
    vector = FAISS.from_documents(docs, ollama_embeddings)
    retriever = vector.as_retriever(search_kwargs={"k": 1})
    multi_query_retriver = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=chat_llm
    )
    memory = Chain.get_Conversational_memory("chat_history")
    chain =  ConversationalRetrievalChain.from_llm(
                chat_llm,
                chain_type="map_reduce",
                retriever=multi_query_retriver,
                memory=memory,
                return_source_documents=True,
    )
    while True:
        question = input("You can start asking question ?? (press q to quit)    \n")
        if question == "q":
            break
        answer = chain.invoke({"question":question})
        print(answer["answer"])
        print("\n\n")


def get_youtube_chain(documents,llm_model):
    docs = Docs.get_youtube_docs(document=documents[0])
    ollama_embeddings = Chain.get_ollama_embedding()
    # chat_llm = models.get_ollama_chat_llm()
    vector = FAISS.from_documents(docs, ollama_embeddings)
    retriever = vector.as_retriever(search_kwargs={"k": 1})
    multi_query_retriver = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm_model
    )
    memory = Chain.get_Conversational_memory("chat_history")
    chain =  ConversationalRetrievalChain.from_llm(
                llm_model,
                chain_type="map_reduce",
                retriever=multi_query_retriver,
                memory=memory,
                return_source_documents=True,
    )
    return chain


def main():
    url = input("enter the url for desired youtube video        ")
    documents = get_youtube_documents(url)
    summary = get_summarization(document=documents[0])
    print(f"Summary for {documents[0].metadata['title']}")
    print(summary)
    print("\n\n")
    chat(documents=documents)


if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    main()

