import os
from dotenv import load_dotenv
from uuid import uuid4
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import chainlit as cl
import Chain
import Docs
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma

set_llm_cache(InMemoryCache())
global_chat_history = []
load_dotenv()

print("ENV's are loaded !")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "ls__cc1f2dc41186415d9625aab9b3ae9d67"

# chain = Chain.create_conversational_retrival_chain(filePath="test_mini.txt" , fileName="text_mini.txt")
# print(chain)
# res = chain.invoke({"question":"what is research are of Dr. Jyothi S Nayak ?"})
# print(res)

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content = "Please upload the file to begin!!",
            accept=["application/pdf" , "text/plain" , "application/py" , "application/md" ,"application/html"],
            max_size_mb=20,
            timeout=30,  
        ).send()
    
    file = files[0]
    print(file)
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docs = Docs.get_docs(fileName=file.name , filepath=file.path)
    if docs == []:
        return None
    else:
        print(len(docs))
        print("BEFORE RETRIVAL MULTI QUERY")
        llm = Chain.get_ollama_llm()
        print(llm)
        ollama_embeddings = Chain.get_ollama_embedding()
        print(ollama_embeddings)
        print("Before Vectores!!!")
        metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))
                     ]
        # vector = await cl.make_async(FAISS.from_documents)(
        #     docs, ollama_embeddings
        # )
        vector =  await cl.make_async(Chroma.from_documents)(
        docs, ollama_embeddings, metadatas=metadatas
         )
        retriever = vector.as_retriever(search_kwargs={"k": 1})
        print(retriever)
        multi_query_retriver = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )
        print(f"Done with retriver_multi query {multi_query_retriver}")
        memory = Chain.get_Conversational_memory("chat_history")
        chat_llm = Chain.get_ollama_chat_llm()
        chain = ConversationalRetrievalChain.from_llm(
            chat_llm,
            chain_type="stuff",
            retriever=multi_query_retriver,
            memory=memory,
            return_source_documents=True,
            )
        print(f"after chain chain is {chain}")
        print("Before chain")
        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
        await msg.update()
        cl.user_session.set("chain",chain)


@cl.on_message
async def main(message: cl.Message):
    print(f"question is ${message.content}")
    chain = cl.user_session.get("chain")
    print(chain)
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke({"question":message.content,
                               "chat_history":global_chat_history
                               }, callbacks=[cb])
    print(res)
    answers = res["answer"]
    print(answer)
    tempCurrentChat = {
        "question":message.content,
        "answer": answers
    }
    global_chat_history.append(tempCurrentChat)

    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for source_idx , source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx + 1}"
            text_elements.append(
                cl.Text(content=source_doc.page_content,name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answers,elements=text_elements).send()



# def main():
#     set_llm_cache(InMemoryCache())
#     print("Staring the app.py scripts ....")
#     chain = Chain.create_conversational_retrival_chain("dummy_scripts/resume_updated_1.pdf")
#     print(chain)
#     res = chain.invoke({"question":"give me the details of projects in give resume ?","chat_history":[]})
#     print(res)
#     load_dotenv()
    


# if __name__ == "__main__":
#     main()