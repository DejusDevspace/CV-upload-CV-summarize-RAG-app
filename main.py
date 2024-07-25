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
def process_pdf(pdf_file):
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
    

def process_docx(docx_file):
    texts = ''
    docx_loader = Docx2txtLoader(docx_file)

    # Load Documents and split into chunks
    texts = docx_loader.load_and_split()

    return texts


def stream_data(text: str):
    for word in text.split(' '):
        yield word + ' '
        time.sleep(0.07)


def main():
    # Title of the streamlit page
    st.title('CV Summary Generator')
    # File uploader to upload cv in either docx or pdf formats
    uploaded_file = st.file_uploader('Upload CV', type=['docx', 'pdf'])
    
    texts = ''
    # Summarize button
    if st.button('Summarize'):
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1] #(docx/pdf)

            st.subheader('File Details:', divider='grey')
            st.write(f'File Name: {uploaded_file.name}')
            st.write(f'File Type: {'Word (.docx)' if file_type == 'docx' else 'PDF (.pdf)'}')

            if file_type == 'docx':
                texts = process_docx(uploaded_file.name)
            elif file_type == 'pdf':
                texts = process_pdf(uploaded_file.name)
            else:
                st.error('Unsupported file format! Please upload a .pdf or .docx file.')
    
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
           "Given the new context, refine the original summary into the following sections:"
           "Name: \n"
           "Email: \n"
           "Key Skills: \n"
           "Last Company: \n"
           "Experience Summary: \n"

           "If the provided context is not useful, return the original summary\n"
           "If any of the sections are not retrievable from the context, say it is not available in the document\n"
           "For example, if Last Company is not available, you would write in the Last Company section:\n"
           "Last Company: Not avaialalbe\n"
       )
    refine_prompt = PromptTemplate.from_template(refine_template)

    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key='input_documents',
        output_key='output_text',
    )

    result = chain.invoke({"input_documents": texts}, return_only_outputs=True)

    # st.subheader('Resume Summary:')
    st.text_area('Summary:', st.write_stream(stream_data(result['output_text'])))


if __name__ == '__main__':
    main()
