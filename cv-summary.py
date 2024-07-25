from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import io
from dotenv import load_dotenv

# Loads environment variables
load_dotenv()


# Function to process .pdf files
def process_pdf(pdf_file: io.IOBase):
    pass

def process_docx(docx_file: io.IOBase):
    pass


def main():
    # Title of the streamlit page
    st.title('CV Summary Generator')
    # File uploader to upload cv in either docx or pdf formats
    uploaded_file = st.file_uploader('Upload CV', type=['docx', 'pdf'])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1] #(docx/pdf)
        if file_type == 'docx':
            pass
        elif file_type == 'pdf':
            pass
        else:
            st.error('Unsupported file format! Please upload a .pdf or .docx file.')


if __name__ == '__main__':
    main()
