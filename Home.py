import streamlit as st
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache


def main():
    st.set_page_config(page_title="Query Nest",page_icon=":books:")

    st.header("Query Nest :books:")
    
    st.text("Which makes search easy and use our RAG engine to query document , or youtube video ")


if __name__ == "__main__":
    set_llm_cache(InMemoryCache())
    load_dotenv()
    main()
