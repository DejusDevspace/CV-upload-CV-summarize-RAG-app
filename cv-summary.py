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
    text = ''
    pdf_loader = PyPDFLoader(pdf_file)

    for page in pdf_loader.load():
        text += page.page_content

    # Replace tab spaces with single spaces 
    text = text.replace('\t', ' ')

    # Splitting the document into chunks of texts
    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        chunk_size=1000,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    # Create documents from a list of texts
    texts = text_splitter.create_documents([text])

    return texts
    

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

            st.subheader('File Details:', divider='grey')
            st.write(f'File Name: {uploaded_file.name}')
            st.write(f'File Type: {'Word (.docx)' if file_type == 'docx' else 'PDF (.pdf)'}')

            if file_type == 'docx':
                ouput = process_docx(uploaded_file)
            elif file_type == 'pdf':
                process_pdf(uploaded_file)
            else:
                st.error('Unsupported file format! Please upload a .pdf or .docx file.')
    
    # st.write_stream(stream_data(text=ouput))


if __name__ == '__main__':
    main()
