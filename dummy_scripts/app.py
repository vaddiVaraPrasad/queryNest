from typing import List
import PyPDF2
from io import BytesIO
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
#from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_community.llms import Ollama

from langchain_community.chat_models import ChatOllama

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    # AskFileResponse(id='e75313d4-50cc-4e2c-83dc-343a23cd95d3', name='7sem_TechnicalResearchPaper_Report_Format_30-12-2022.pdf', 
        # path='/Users/varaprasadvaddi/Desktop/final Year Project/.files/ec201740-334b-4d1f-92b8-0b98887e48a7/e75313d4-50cc-4e2c-83dc-343a23cd95d3.pdf', size=1879703, type='application/pdf')
    file = files[0]
    print(file)

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
        
    #pdf_stream = BytesIO(content)
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="llama2:7b-chat")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama2:7b-chat"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    print(f"question is ${message.content}")
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    print(chain)
    

    res = chain.invoke(message.content)
    print(res)
    answer = res["answer"]
    print(answer)
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()