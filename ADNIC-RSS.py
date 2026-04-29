import streamlit as st
import pdfplumber
from Models import get_HF_embeddings, get_gpt3_embeddings, get_doc2vec_embeddings, cosine
import re
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

# Function to clean and preprocess text
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)   # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

# Function to extract and clean text from PDF
def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += clean_text(text)
    return data

# Function to check keyword relevance
def keyword_check(resume_text, keywords):
    matches = [keyword for keyword in keywords if keyword.lower() in resume_text.lower()]
    return len(matches) / len(keywords) * 100 if keywords else 0

# Streamlit App
st.set_page_config(page_title="Applicant Tracking System", layout="wide", initial_sidebar_state="expanded")

# Add the logo to the sidebar
logo = Image.open("CDSlogo.png")
with st.sidebar:
    st.image(logo, width=200)

# Main Title
st.title("Resume Screening System")

# Sidebar for embedding options
embedding_flag = 'HuggingFace-BERT'
with st.sidebar:
    st.markdown('**Select the embedding model to use**')
    embedding_flag = st.selectbox(
        'Embedding Options',
        ['HuggingFace-BERT', 'HuggingFace-MPNet', 'HuggingFace-MiniLM', 'GPT-3.5', 'Doc2Vec'],
        label_visibility="collapsed"
    )

# Tabbed layout for UI
tab1, tab2 = st.tabs(["**Home**", "**Results**"])

# Tab Home: Input for resumes and job description
with tab1:
    st.header("Upload Resumes and Job Description")
    
    uploaded_files = st.file_uploader('**Upload resumes in PDF format:**', type="pdf", accept_multiple_files=True)
    JD = st.text_area("**Enter the job description:**")
    mandatory_keywords = st.text_input("**Enter mandatory keywords (comma-separated):**")
    comp_pressed = st.button("Compare!")
    
    if comp_pressed and uploaded_files and JD:
        mandatory_keywords_list = [kw.strip() for kw in mandatory_keywords.split(',')]
        uploaded_file_texts = [extract_pdf_data(file) for file in uploaded_files]

        # Generate embeddings based on selected model
        if embedding_flag == 'HuggingFace-BERT':
            model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
            JD_embeddings = get_HF_embeddings([JD], model_name)
            resume_embeddings = [get_HF_embeddings([text], model_name) for text in uploaded_file_texts]
        elif embedding_flag == 'HuggingFace-MPNet':
            model_name = 'sentence-transformers/all-mpnet-base-v2'
            JD_embeddings = get_HF_embeddings([JD], model_name)
            resume_embeddings = [get_HF_embeddings([text], model_name) for text in uploaded_file_texts]
        elif embedding_flag == 'HuggingFace-MiniLM':
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            JD_embeddings = get_HF_embeddings([JD], model_name)
            resume_embeddings = [get_HF_embeddings([text], model_name) for text in uploaded_file_texts]
        elif embedding_flag == 'GPT-3.5':
            JD_embeddings = get_gpt3_embeddings([JD])
            resume_embeddings = [get_gpt3_embeddings([text]) for text in uploaded_file_texts]
        elif embedding_flag == 'Doc2Vec':
            JD_embeddings, resume_embeddings = get_doc2vec_embeddings(JD, uploaded_file_texts)
        else:
            st.error("Invalid embedding model selected!")
            sys.exit(1)

        # Calculate similarity scores
        similarity_scores = cosine(resume_embeddings, JD_embeddings)
        
        # Check keyword relevance
        keyword_matches = [keyword_check(resume_text, mandatory_keywords_list) for resume_text in uploaded_file_texts]
        
        # Combine scores
        combined_scores = [0.5 * sim + 0.5 * kw for sim, kw in zip(similarity_scores, keyword_matches)]

# Tab Results: Display scores
with tab2:
    st.header("Results")
    if comp_pressed and uploaded_files and JD:
        result_data = [
            {"Resume Name": file.name,
             "Keyword Match (%)": kw,
             "Suitability Score (%)": score,
             "Icon": "🟢" if score > 60 else "🟡" if 40 <= score <= 60 else "🔴"}
            for file, kw, score in zip(uploaded_files, keyword_matches, combined_scores)
        ]
        result_df = pd.DataFrame(result_data)
        st.table(result_df)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=result_df["Resume Name"], y=result_df["Keyword Match (%)"], name="Keyword Match (%)"))
        fig.add_trace(go.Bar(x=result_df["Resume Name"], y=result_df["Suitability Score (%)"], name="Suitability Score (%)"))
        st.plotly_chart(fig)
