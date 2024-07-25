from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import io
import time
from dotenv import load_dotenv

# Loads environment variables
load_dotenv()


# Function to process .pdf files
def process_pdf(pdf_file: io.IOBase):
    pass

def process_docx(docx_file: io.IOBase):
    pass


def stream_data(text: str):
    for word in text.split(' '):
        yield word + ' '
        time.sleep(0.07)


def main():
    # Title of the streamlit page
    st.title('CV Summary Generator')
    # File uploader to upload cv in either docx or pdf formats
    uploaded_file = st.file_uploader('Upload CV', type=['docx', 'pdf'])
    
    # Summarize button
    if st.button('Summarize'):
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1] #(docx/pdf)
            if file_type == 'docx':
                st.write_stream(stream_data('I am a boy that is testing lorem ipsum kinda shit here'))
                # st.text_st('Docx file!')
            elif file_type == 'pdf':
                st.text('PDF File!')
            else:
                st.error('Unsupported file format! Please upload a .pdf or .docx file.')


if __name__ == '__main__':
    main()
