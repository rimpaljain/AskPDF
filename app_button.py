from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def get_pdf_text(pdf):
    # extract the text
    text = ""
    if pdf is not None:
      for pdf_file in pdf:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunk_text(text):
   # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def get_conversation_chain(knowledge_base, user_question, pdf): 
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = knowledge_base.similarity_search(user_question)
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
    # st.write(pdf)
    st.write(response)
    st.write(docs)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    knowledge_base = ""
    pdf=""

    pdf = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)
    st.write(pdf)
    if st.button("OK"):
            with st.spinner("Processing your PDFs..."):

                # Get PDF Text
                text = get_pdf_text(pdf)

                # Get Text Chunks
                chunks = get_chunk_text(text)
     
                # Create Vector Store
                knowledge_base = get_knowledge_base(chunks)
                st.write("DONE")

    user_question = st.text_input("Ask a question about your PDF:")
    if user_question and knowledge_base:
        get_conversation_chain(knowledge_base, user_question, pdf)
                
if __name__ == '__main__':
    main()
