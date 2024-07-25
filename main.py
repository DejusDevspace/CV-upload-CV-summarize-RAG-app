from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.word_document import Docx2txtLoader
import streamlit as st
import io
import time
from typing import List
import tempfile
import os
from dotenv import load_dotenv

# Loads environment variables
load_dotenv()


# Function to process .pdf files
def process_pdf(pdf_file: str) -> List:
    text = ''
    pdf_loader = PyPDFLoader(pdf_file)

    for page in pdf_loader.load():
        text += page.page_content

    # Replace tab spaces with single spaces 
    text = text.replace('\t', ' ')

    # Splitting the document into chunks of texts
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )

    # Create documents from list of texts
    texts = text_splitter.create_documents([text])

    return texts


# Function to process .docx files
def process_docx(docx_file: str) -> List:
    text = ''
    docx_loader = Docx2txtLoader(docx_file)

    # Load Documents and split into chunks
    text = docx_loader.load_and_split()

    return text


# Function to get temporary path of uploaded files
def load_file(file: io.IOBase, suffix: str) -> str:
    # Save uploaded file to a temporary file and get the path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file.getbuffer())
        file_path = tmp_file.name
    return file_path


def stream_data(text: str):
    for word in text.split(' '):
        yield word + ' '
        time.sleep(0.1)


def main():
    # Title of the streamlit page
    st.title('CV Summary Generator')
    # File uploader to upload cv in either docx or pdf formats
    uploaded_file = st.file_uploader('Upload CV', type=['docx', 'pdf'])

    texts = ''
    # Summarize button
    if st.button('Summarize'):
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]  # (docx/pdf)

            st.subheader('File Details:', divider='grey')
            st.write(f'File Name: {uploaded_file.name}')
            st.write(f'File Type: {'Word (.docx)' if file_type == 'docx' else 'PDF (.pdf)'}')

            # Get the temporary path to the file
            file_path = load_file(uploaded_file, suffix=f'.{file_type}')

            try:
                if file_type == 'docx':
                    texts = process_docx(file_path)
                elif file_type == 'pdf':
                    texts = process_pdf(file_path)
                else:
                    st.error('Unsupported file format! Please upload a .pdf or .docx file.')
            finally:
                # Remove the temporary file after getting the data from it
                os.remove(file_path)

            # LLM object
            llm = GoogleGenerativeAI(model='gemini-1.5-pro', temperature=0)

            # Prompt template
            prompt_template = """You are given a resume to summarize.
            Write a verbose detail of the following:
            {text}

            Details:
            """
            # Prompt
            prompt = PromptTemplate.from_template(prompt_template)

            # Refine chain prompt template
            refine_template = (
                "Your job is to produce a final summary\n"
                "We have provided an existing summary up to a certain point: {existing_answer}\n"
                "We want a refined version of the existing summary (if needed) based on the additional context below\n"
                "------------\n"
                "{text}\n"
                "------------\n"
                "Given the new context, refine the original summary into the following sections ONLY:\n"
                "Note: Do NOT provide a title for the summary, just start from the parameters below:\n"
                "Name: \n\n"
                "Email: \n\n"
                "Key Skills: \n\n"
                "Last Company: \n\n"
                "Experience Summary: \n\n"

                "Each parameter above should be printed in bold and larger in font than the details\n"
                "If the provided context is not useful, return the original summary\n"
                "If any of the sections are not available from the context, say it is not available in the document\n"
                "For example, if Last Company is not available, you would write in the Last Company section:\n"
                "Last Company: Not available\n"
            )

            # Refine chain prompt
            refine_prompt = PromptTemplate.from_template(refine_template)

            # Summarize chain
            chain = load_summarize_chain(
                llm=llm,
                chain_type='refine',
                question_prompt=prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True,
                input_key='input_documents',
                output_key='output_text',
            )

            # Display loading summary while summary is being generated
            with st.spinner('Loading summary...'):
                result = chain.invoke({"input_documents": texts}, return_only_outputs=True)

            # Resume summary the chain output key
            summary = result['output_text']

            st.subheader('Resume Summary:', divider='grey')
            st.write_stream(stream_data(summary))


if __name__ == '__main__':
    main()
