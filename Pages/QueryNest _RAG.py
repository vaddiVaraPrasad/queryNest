import streamlit as st
from dotenv import load_dotenv
import os
import Docs
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
# from htmlTemplate import css , bot_template , user_template
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import models
import Chain


def main():
    st.set_page_config(page_title="Query Nest",page_icon=":books:")

    # st.write(css , unsafe_allow_html=True)

    st.header("Query Nest Document RAG Engine ")

   
    ## this is file type
#     UploadedFile(file_id='375a57af-a4d4-4893-8bbf-ee66b3b5821d', name='sample_text.txt', type='text/plain', size=2351, _file_urls=file_id: "375a57af-a4d4-4893-8bbf-ee66b3b5821d"
# upload_url: "/_stcore/upload_file/b9ba191a-5663-4fdd-b978-584ae6ac2b7f/375a57af-a4d4-4893-8bbf-ee66b3b5821d"
# delete_url: "/_stcore/upload_file/b9ba191a-5663-4fdd-b978-584ae6ac2b7f/375a57af-a4d4-4893-8bbf-ee66b3b5821d"
# )
    if "coversation_chain" not in st.session_state:
        st.session_state.coversation_chain = None

    # st.write(user_template.replace("{{MSG}}","Hello robot"),unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello human"),unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Upload Documents")
        files = st.file_uploader("Upload your PDF's here")
        st.text("")
        st.text("")
        if st.button("Process"):
            with st.spinner("Uploading"):
                temp_file_path = f"files/{files.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(files.getbuffer())
            docs = Docs.get_docs(filepath=temp_file_path,fileName=files.name)
            if docs == []:
                return None
            else:
                print(len(docs))
                print("BEFORE RETRIVAL MULTI QUERY")
                llm = models.get_ollama_chat_llm()
                print(llm)
                ollama_embeddings = Chain.get_ollama_embedding()
                print(ollama_embeddings)
                print("Before Vectores!!!")
                metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))]
            with st.spinner("Embedding"):
                vector = FAISS.from_documents(docs, ollama_embeddings)
                retriever = vector.as_retriever(search_kwargs={"k": 1})
                print(retriever)
                multi_query_retriver = MultiQueryRetriever.from_llm(
                    retriever=retriever, llm=llm
                )
                print(f"Done with retriver_multi query {multi_query_retriver}")
                memory = Chain.get_Conversational_memory("chat_history")
                chat_llm = models.get_ollama_chat_llm()
                st.session_state.coversation_chain = ConversationalRetrievalChain.from_llm(
                    chat_llm,
                    chain_type="stuff",
                    retriever=multi_query_retriver,
                    memory=memory,
                    return_source_documents=True,
                    )
                
            st.text("You can Ask query regarding file")

    if st.session_state.coversation_chain == None:
        st.write("Upload file to chat !")
    else:
        question = st.text_input("Ask your question about document!",key="RAG")
        if question:
            with st.spinner("Collection Info"):
                responce = st.session_state.coversation_chain.invoke({"question":question})
                st.write(responce)
                answer = responce["answer"]

                st.subheader(answer)


if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    main()