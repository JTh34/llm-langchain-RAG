import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import tempfile
import tiktoken

st.title("🕵🏼 Question/Answering with your PDF file 🔍")

st.write("""
STEP 1: Upload a PDF file 🗄️

STEP 2: Enter your OpenAI API key [OpenAI](https://platform.openai.com/account) 🔐

STEP 3: Type your question at the bottom 👨🏼‍💻 and click "Run" 🚀
""")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

openai_key = st.text_input("Enter your OpenAI API Key", type="password")

prompt_text = st.text_area("Enter your questions here:")

chain_type_options = ['stuff', 'map_reduce', "refine", "map_rerank"]
chain_type = st.radio("🕹️ Chain type", chain_type_options)

k = st.slider("Number of relevant chunks", 1, 5, 2)

def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})

    return result

if st.button("Run"):
    os.environ["OPENAI_API_KEY"] = openai_key

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            result = qa(file=tmp.name, query=prompt_text, chain_type=chain_type, k=k)
            
            st.write("👨🏼‍💻", prompt_text)
            st.write("🦾", result["result"])
            st.write("Relevant source text:")
            st.write('\n\n********************\n\n'.join(doc.page_content for doc in result["source_documents"]))
