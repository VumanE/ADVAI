#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate, LLMChain
import glob
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from PyPDF2 import PdfReader
from langchain.chains import AnalyzeDocumentChain
from langchain.document_loaders import PyPDFDirectoryLoader


# In[2]:


import os 
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature = 0.5)



# In[3]:


loader = PyPDFDirectoryLoader("/Users/evanvu/Downloads/PDFs")
retrieval_doc = loader.load()


# In[4]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(retrieval_doc) #split


# In[5]:


persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)


# In[ ]:


vectordb.persist() #Not going to lie this was a recommended step from online I have no clue what it does. I think it just saves to disk
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)


# In[ ]:


retriever = vectordb.as_retriever() #make receiver


# In[ ]:


qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="map_reduce",
                                  retriever=retriever,
                                  return_source_documents=True)


# In[ ]:


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


# In[ ]:


query = input("Question: ")
llm_response = qa_chain(query)
process_llm_response(llm_response)


# In[ ]:




