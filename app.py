import os
import openai
import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
load_dotenv()
#api_key = os.environ.get('OPEN_API_KEY')
def main():
    st.header('Chat with PDF')

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 20,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open (f'{store_name}.pkl', 'rb') as f:
                Vectorstore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            Vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f'{store_name}.pkl', 'wb') as f:
                pickle.dump(Vectorstore, f)

        query = st.text_input("Ask Questions related to the PDF file: ")

        if query:
            docs = Vectorstore.similarity_search(query=query, k=3)
            llm = ChatOpenAI(model='gpt-3.5-turbo')
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            response = ''
            with get_openai_callback()  as cb:
                response += chain.run(input_documents = docs, question=query)
                st.write(response)

if __name__=="__main__":
    main()