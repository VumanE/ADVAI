# ADVAI
For ADV

**At the end are some tips for how to run the code**

These are models that attempted to use LangChain and OPENAI to run queries over different file types. 

What we have:

1. A PDF Retrieval bot that takes in multiple PDFs and can answer queries about them (including summarization)
2. A Streamlit.io (frontend) application that can take in a single PDF and can answer queries about the document.

So far, these models can summarize/retrieve data and run without issues over multiple documents. 

What we still need:

1. A longer, more thorough response during summarization and data retrieval
2. A private way to embed data; OPENAI still stores data even if it's on a local host (through the API).
3. A way to run analysis on the data. 

Some possible ideas/ways to improve:

1. Extend the token limit. This will likely happen anyway as ChatGPT becomes more developed, but token size is one of the most prominent limiting factors
2. Using an LLM other than OpenAI. There are quite a few online, and we would have to explore the privacy/capability of each.
3. Explore embeddings such as HuggingFace more. If tokens become less of an issue, embeddings could produce more accurate 
4. Setting up a private server to run a custom LLM, which would ensure privacy and a tailored LLM.
5. Looking for a new Python Library that can export LLM results to some sort of graphing software such as Seaborn/madplotlib

Some tips for running the code:

1. I used Visual Studios and Jupyter for my code. Any will work, but those are some good ones.
2. Make sure to download Anaconda or find ways to set up an environment
3. There are a TON of different packages you need, and even I'm not sure of all of them. As you go, you **will** encounter errors that occur because of a lack of a package. Just use the Anaconda Terminal or Anaconda to download the packages needed.

