import requests
from PIL import Image
import streamlit as st
import pdfplumber
from Models import get_HF_embeddings, cosine
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
st.set_page_config(page_title="Applicant Tracking System", layout="wide")

# Add the logo and description to the sidebar
logo = Image.open("CDSlogo.png")
with st.sidebar:
    st.markdown("""
    **AI and NLP-powered tool** for automated candidate ranking and filtering using 3 AI models: BERT, MPNet, and MiniLM.

    - **BERT:** Processes text bidirectionally for improved resume parsing and job matching.
    - **MPNet:** Advanced model that enhances pre-training for better resume evaluation.
    - **MiniLM:** AI model optimized for speed and efficiency while maintaining high accuracy.
    """)

# Main Title
st.image(logo, width=200)
st.title("Resume Screening System")

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
        mandatory_keywords_list = [keyword.strip() for keyword in mandatory_keywords.split(',')]
        uploaded_file_texts = [extract_pdf_data(file) for file in uploaded_files]

        # Generate embeddings and calculate similarity scores for each model
        models = {
            'BERT': 'sentence-transformers/bert-base-nli-mean-tokens',
            'MPNet': 'sentence-transformers/all-mpnet-base-v2',
            'MiniLM': 'sentence-transformers/all-MiniLM-L6-v2'
        }

        all_scores = {}
        for model_name, model_path in models.items():
            JD_embeddings = get_HF_embeddings([JD], model_path)
            resume_embeddings = [get_HF_embeddings([text], model_path) for text in uploaded_file_texts]
            similarity_scores = cosine(resume_embeddings, JD_embeddings)
            all_scores[model_name] = similarity_scores

        # Check keyword relevance
        keyword_matches = [keyword_check(resume_text, mandatory_keywords_list) for resume_text in uploaded_file_texts]

        # Combine scores and calculate averages
        combined_scores = {}
        for model_name in models:
            combined_scores[model_name] = [
                0.5 * sim + 0.5 * kw for sim, kw in zip(all_scores[model_name], keyword_matches)
            ]

        average_scores = [
            sum(scores) / len(models) for scores in zip(*combined_scores.values())
        ]

# Tab Results: Display scores
with tab2:
    st.header("Results")
    if comp_pressed and uploaded_files and JD:
        result_data = []
        for i, file in enumerate(uploaded_files):
            result_entry = {
                "Resume Name": f"{'🟢' if average_scores[i] > 60 else '🟡' if 40 <= average_scores[i] <= 60 else '🔴'} {file.name}",
                "Keyword Match (%)": keyword_matches[i],
            }
            for model_name in models:
                result_entry[f"{model_name} Score (%)"] = combined_scores[model_name][i]
            result_entry["Average Score (%)"] = average_scores[i]
            result_data.append(result_entry)

        result_df = pd.DataFrame(result_data)
        st.table(result_df)

        # Graph showing the average score with percentage annotations
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
        fig = go.Figure()
        for idx, row in result_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Resume Name"]],
                y=[row["Average Score (%)"]],
                name=row["Resume Name"],
                marker_color=colors[idx % len(colors)],
                text=[f"{row['Average Score (%)']:.2f}%"],  # Add score percentage as text
                textposition='outside',  # Position text above the bars
                width=0.5  # Thin bar width
            ))
        fig.update_layout(
            barmode='group',
            showlegend=False,
            xaxis_title="Resumes",
            yaxis_title="Average Score (%)",
            
        )
        st.plotly_chart(fig)
 
