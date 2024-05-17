import streamlit as st
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import descision_chain
import normal_llm_chain
import serpAPI
import youtube
import models


def main():
    st.set_page_config(page_title="Query Nest",page_icon=":youtube:")

    st.header("Query Nest Youtube Search Engine ")

    if "youtube_chain" not in st.session_state:
        st.session_state.youtube_chain = None
    # if "youtube_model" not in st.session_state:
    #     st.session_state.youtube_model = None

    col1 , col2 = st.columns(2)

    with col1:
        if st.session_state.youtube_chain == None:
            user_query = st.text_input("Get Info about your query ?",key="QueryNest Youtube")
            if user_query:
                with st.spinner("Deciding Content Type"):
                    decision =  descision_chain.get_decision(user_query=user_query)
                    des = descision_chain.remove_special_chars(decision["text"])
                    print(des)
                    st.text(f"We dicided query is related to {des}")
                
                with st.spinner("Deciding on Kind of Model & tring to answer Your query"):
                    if "LLM" in des:
                        model_decision = descision_chain.get_model_descision(user_query=user_query)
                        llm_model =descision_chain.get_models_modified(models_decision=model_decision)
                        llm_model_name = descision_chain.get_model_name_modifed(models_decision=model_decision)
                        st.write(f"{llm_model_name} Model is Going to use for this query")
                        conversational_chain  = normal_llm_chain.get_noraml_chain(llm_model=llm_model)
                        # if descision_chain.get_model_name_modifed(models_decision=model_decision) == "CODE LLAMA2 PYTHON" or descision_chain.get_model_name_modifed(models_decision=model_decision) == "CODE LLAMA2":
                        #     thenical_query = descision_chain.get_thenical_query(user_query)
                        #     st.write(f"Thenical query is \n{thenical_query['text']} ")
                        #     result = conversational_chain.invoke(input=thenical_query["text"])
                        #     st.write(result)
                        #     answer = result["response"]
                        #     st.subheader(answer)
                        # else:
                            # result = conversational_chain.invoke(input=user_query)
                            # st.write(result)
                            # answer = result["response"]
                            # st.subheader(answer)
                        result = conversational_chain.invoke(input=user_query)
                        st.write(result)
                        answer = result["response"]
                        st.subheader(answer)
                        
                    
                    elif "INTERNET" in des:
                        content = serpAPI.get_serpapi_content(user_query=user_query)
                        total_content  = f"User Query --->  {user_query}\n\nContent is {content[0]}\n"
                        st.write(total_content)
                        model_decision = descision_chain.get_model_descision(user_query=total_content)
                        llm_model = descision_chain.get_models_modified(models_decision=model_decision)
                        llm_model_name = descision_chain.get_model_name_modifed(models_decision=model_decision)
                        st.write(f"{llm_model_name} Model is Going to use for this query")
                        serpapi_chain = serpAPI.get_serpapi_chain(llm_model=llm_model)
                        # if descision_chain.get_model_name_modifed(models_decision=model_decision) == "CODE LLAMA2 PYTHON" or descision_chain.get_model_name_modifed(models_decision=model_decision) == "CODE LLAMA2":
                        #     thenical_query = descision_chain.get_thenical_query(user_query)
                        #     st.write(f"Thenical query is \n{thenical_query['text']} ")
                        #     result = serpapi_chain.invoke({"query":thenical_query['text'] , "context":content})
                        #     st.write(result)
                        #     answer = result["text"]
                        #     st.subheader(answer)
                        # else:
                        #     result = serpapi_chain.invoke({"query":user_query , "context":content})
                        #     st.write(result)
                        #     answer = result["text"]
                        #     st.subheader(answer)
                        result = serpapi_chain.invoke({"query":user_query , "context":content})
                        st.write(result)
                        answer = result["text"]
                        st.subheader(answer)

                    elif "YOUTUBE" in des:
                        documents = youtube.get_youtube_documents(user_query)
                        st.write(f"Found video  {documents[0].metadata['title']} from {documents[0].metadata['author']}  with {youtube.convert_seconds_to_minutes_and_seconds(documents[0].metadata['length'])} ")
                        model_decision =  descision_chain.get_model_descision(user_query=documents[0].metadata['title'])
                        llm_model = descision_chain.get_models_modified(models_decision=model_decision)
                        llm_model_name = descision_chain.get_model_name_modifed(models_decision=model_decision)
                        # st.session_state.youtube_model = llm_model_name
                        st.write(f"{llm_model_name} Model is Going to use for this query")
                        summary = youtube.get_summarization(document=documents[0],llm_model=llm_model)  
                        st.subheader("Summary..")
                        st.write(summary)
                        st.write()
                        st.write()
                        with st.spinner("Embedding the docs"):
                            st.session_state.youtube_chain = youtube.get_youtube_chain(documents=documents,llm_model=llm_model)
                        st.write()
                        st.write("You can start asking questions about youtube viedo")
                        st.write()

                    else:
                        st.write("Sry to say we can't be proccess your question at the moment pls come back after a while !!")

    with col2:
        if st.session_state.youtube_chain == None:
            st.write("Try giving Youtube Url , to start Conversations")
        else:
            youtube_query = st.text_input("Ask question about youtubr viedo")
            if youtube_query:
                with st.spinner("Collection info viedo to answer your query"):
                    # if st.session_state.youtube_model  == "CODE LLAMA2 PYTHON" or st.session_state.youtube_model  == "CODE LLAMA2":
                    #     thenical_query = descision_chain.get_thenical_query(user_query)
                    #     st.write(f"Thenical query is \n{thenical_query['text']} ")
                    #     responce = st.session_state.youtube_chain.invoke({"question":thenical_query['text']})
                    #     st.write(responce)
                    #     st.subheader(responce["answer"])
                    # else:
                    #     responce = st.session_state.youtube_chain.invoke({"question":youtube_query})
                    #     st.write(responce)
                    #     st.subheader(responce["answer"])
                    responce = st.session_state.youtube_chain.invoke({"question":youtube_query})
                    st.write(responce)
                    st.subheader(responce["answer"])






if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    main()