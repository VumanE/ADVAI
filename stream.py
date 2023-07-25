import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


with st.sidebar:
    st.title("PDF Chat")
    st.markdown('''
    ## About
    This app is for ADV 
    It can chat with files 
    ''')
    add_vertical_space(10)
    st.write('By Evan')

load_dotenv()

def main():
    st.header("Chat with PDF")


    pdf = st.file_uploader("Upload your pdf", type="pdf")
    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        

        embeddings = OpenAIEmbeddings()

        VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1","rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded from the disk")

        else: 
            with open(f"{store_name}.pk1","wb") as f:
                pickle.dump(VectorStore, f)

            st.write("Embedding completed")
        query = st.text_input("Ask Questions about your PDF: ")
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature= 0)
            chain = load_qa_chain(llm=llm, chain_type="map_reduce")
            
            response = chain.run(input_documents=docs, question = query)
            
            st.write(response)
        


if __name__ == "__main__":
    main()