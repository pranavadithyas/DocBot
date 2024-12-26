from langchain.chains import RetrievalQA

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from flask import Flask, request, render_template, jsonify

from configparser import ConfigParser

from langchain_google_genai import GoogleGenerativeAI
import os
import configparser

from langchain.embeddings import SentenceTransformerEmbeddings
import sentence_transformers

# config_object = ConfigParser()
# config_object.read("config.ini")
# openai_config = config_object["openai"]
# open_ai_key = openai_config['key']

global db_chain

app = Flask(__name__)

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/set_params_session',methods=["GET","POST"])
def set_params_session():
    global dataDirectory
    dataDirectory = request.form["dataDirectory"]
    global docsearch
    docsearch = gen_and_store_embeddings()
    global qa_chain
    qa_chain = generate_qa_chain()
    
    return "True"

def document_loader():
    loader = DirectoryLoader(dataDirectory)
    documents = loader.load()
    return documents
def split_documents():
    documents = document_loader()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len)
    texts = text_splitter.split_documents(documents)
    return texts
def gen_and_store_embeddings():
    texts = split_documents()
    embedding_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

    class SentenceTransformersEmbeddings:
        def embed_documents(self, texts):
            return embedding_model.encode(texts, convert_to_tensor=True).tolist()

        def embed_query(self, text):
            return embedding_model.encode([text], convert_to_tensor=True).tolist()[0]

    # Use Sentence-Transformers embeddings
    embeddings = SentenceTransformersEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch

def generate_qa_chain():
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    # Retrieve the API key from the config file
    api_key = config.get("google_api", "key")
    
    # Set the API key in the environment variable if not already set
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = api_key
    model_name= "gemini-pro"
    llm = GoogleGenerativeAI(model=model_name)
    retriever=docsearch.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, verbose=True)
    return qa_chain

@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    # matching_documents = docsearch.similarity_search(query)
    # response = qa_chain.run(input_documents = matching_documents,question=query)
    response = qa_chain.run(query)
   
    return response

if __name__ == "__main__":
    app.run(debug=True)
