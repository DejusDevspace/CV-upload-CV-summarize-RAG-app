from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
from dotenv import load_dotenv

# Loads environment variables
load_dotenv()



def main():
    pass


if __name__ == '__main__':
    main()
