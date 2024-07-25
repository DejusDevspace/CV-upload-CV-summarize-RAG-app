# Resume Summary Generator 
This repository contains a simple RAG application with a basic streamlit interface that allows a user to upload
a resume, and get a summary of the uploaded resume.

It uses Google's gemini as the LLM for the RAG application.

## Installation
To run the app locally, 
1. Clone the repository:
    ```sh
      git clone https://github.com/DejusDevspace/CV-upload-CV-summarize-RAG-app.git
    ```
2. Navigate to the project directory:
    ```sh
      cd CV-upload-CV-summarize-RAG-app
    ```
3. Install the required packages:
    ```sh
      pip install -r requirements.txt
    ````
4. Run the streamlit app:
    ```sh
      streamlit run main.py
    ```