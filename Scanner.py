import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare
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
    st.markdown('**Select the embedding model to use**')
    flag = st.selectbox(
        'Embedding Options',
        ['HuggingFace-BERT', 'Doc2Vec'],
        label_visibility="collapsed"
    )

# Main Title
st.title("Resume Screening System")

# Tabbed layout for UI
tab1, tab2 = st.tabs(["**Home**", "**Results**"])

# Tab Home: Input for resumes and job description
with tab1:
    st.header("Upload Resumes and Job Description")
    
    # Resume file uploader
    uploaded_files = st.file_uploader(
        '**Upload resumes in PDF format:**',
        type="pdf",
        accept_multiple_files=True
    )
    
    # Job description text input
    JD = st.text_area("**Enter the job description:**")
    
    # Mandatory keywords input
    mandatory_keywords = st.text_input(
        "**Enter mandatory keywords (comma-separated):**"
    )
    
    # Compare button
    comp_pressed = st.button("Compare!")
    
    # Process the uploaded files when button is pressed
    if comp_pressed and uploaded_files and JD:
        mandatory_keywords_list = [kw.strip() for kw in mandatory_keywords.split(',')]
        uploaded_file_texts = [extract_pdf_data(file) for file in uploaded_files]
        
        # Perform comparison
        scores = compare(uploaded_file_texts, JD, flag)
        
        # Check keyword relevance
        keyword_matches = [
            keyword_check(resume_text, mandatory_keywords_list) for resume_text in uploaded_file_texts
        ]
        
        # Combine scores and keyword matches
        combined_scores = [
            0.5 * float(similarity) + 0.5 * float(keywords)
            for similarity, keywords in zip(scores, keyword_matches)
        ]

# Tab Results: Display the scores
with tab2:
    st.header("Results")
    if comp_pressed and uploaded_files and JD:
        # Prepare data for display
        result_data = [
            {
                "Resume Name": file.name,
                "Keyword Match Percentage (%)": keyword_match,
                "Suitability Score (%)": combined_score,
                "Icon": (
                    "🟢" if combined_score > 60 else
                    "🟡" if 40 <= combined_score <= 60 else
                    "🔴"
                ),
            }
            for file, keyword_match, combined_score in zip(uploaded_files, keyword_matches, combined_scores)
        ]
        
        # Convert to DataFrame for charting
        result_df = pd.DataFrame(result_data)
        
        # Display results as a table
        st.subheader("Results Table")
        st.write("The table below shows the analysis for the uploaded resumes:")
        result_df["Resume Name"] = result_df["Icon"] + " " + result_df["Resume Name"]
        st.table(result_df[["Resume Name", "Keyword Match Percentage (%)", "Suitability Score (%)"]])

        # Display grouped bar chart
        st.subheader("Results Chart")
        st.write("The chart below visualizes the keyword match percentages and suitability scores:")

        if not result_df.empty:
            # Create grouped bar chart
            fig = go.Figure()

            # Add Keyword Match Percentage bars
            fig.add_trace(
                go.Bar(
                    x=result_df["Resume Name"],
                    y=result_df["Keyword Match Percentage (%)"],
                    name="Keyword Match Percentage (%)",
                    marker=dict(color='rgb(66, 135, 245)'),  # Custom color
                    text=[f"{value:.1f}%" for value in result_df["Keyword Match Percentage (%)"]],
                    textposition="auto",
                )
            )

            # Add Suitability Score bars
            fig.add_trace(
                go.Bar(
                    x=result_df["Resume Name"],
                    y=result_df["Suitability Score (%)"],
                    name="Suitability Score (%)",
                    marker=dict(color='rgb(48, 191, 78)'),  # Custom color for green
                    text=[f"{value:.1f}%" for value in result_df["Suitability Score (%)"]],
                    textposition="auto",
                    textfont=dict(color="white")
                    
                )
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Resume Name",
                yaxis_title="Percentage (%)",
                barmode="group",  # Group bars side-by-side
                template="plotly_white",
                legend=dict(title="Metrics"),
                xaxis_tickangle=45,  # Rotate x-axis labels for better readability
            )

            st.plotly_chart(fig)
