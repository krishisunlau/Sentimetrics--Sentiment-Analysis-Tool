import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
import io

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_KEY = "******"

def configure_api():
    """Configures the Google Gemini AI API."""
    genai.configure(api_key=API_KEY)

def generate_synthetic_data():
    """Generates a synthetic dataset of employee feedback using Gemini AI."""
    configure_api()

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }
    )

    chat_session = model.start_chat(history=[])

    prompt = """
    Generate 45 realistic employee feedback entries, formatted as:
    - Type: (Email, Chat, Survey Response)
    - Content: (Actual feedback message)
   
    Examples:
    1. Type: Email, Content: "I'm feeling overwhelmed with workload and need support."
    2. Type: Chat, Content: "Great job on the project! The team worked really well together."
    """

    response = chat_session.send_message(prompt)
   
    data_lines = response.text.split("\n")
    dataset = []
   
    for line in data_lines:
        if "Type:" in line and "Content:" in line:
            type_part = line.split("Type:")[1].split(",")[0].strip()
            content_part = line.split("Content:")[1].strip()
            dataset.append({"Type": type_part, "Content": content_part})
   
    return pd.DataFrame(dataset)

def save_dataset(df):
    csv = df.to_csv(index=False)
    return csv

def analyze_feedback(reviews, prompt):
    """Generates an AI response based on employee feedback and user-selected prompt."""
    configure_api()
   
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    )

    chat_session = model.start_chat(history=[])
   
    full_prompt = f"{prompt}\n\n" + "\n\n".join(reviews)
   
    response = chat_session.send_message(full_prompt)
   
    return response.text

predefined_prompts = [
    "<Choose a prompt>",
    "Summarize employee morale trends.",
    "Identify signs of burnout and stress.",
    "Detect positive feedback and motivational themes.",
    "Analyze changes in tone over time.",
    "Highlight common complaints and areas for improvement.",
    "Detect any changes in satisfaction over time.",
    "Detect any issues on workplace fairness.",
    "Analyze patterns in employee data.",
    "Assess employee thoughts on company values.",
    "Analyze feedback on leadership and trust.",
    "Analyze employee motivation over time.",
    "Give advice on how to boost company morale.",
    "Provide a general analysis on the employee data.",
    "Display only the percentage of employee data was positive, neutral, and negative.",
    "Conduct a statistical analysis on employee data."

]

st.title("Sentimetrics")

st.subheader("Analyze Employee Sentiments")

if st.button("Generate Synthetic Employee Data"):
    with st.spinner("Generating employee dataset..."):
        df = generate_synthetic_data()
        csv_data = save_dataset(df)
        st.success("Dataset generated successfully!")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="employee_data.csv",
            mime="text/csv"
        )

uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

st.subheader("Choose a Prompt or Enter Your Own:")
selected_prompt = st.selectbox("Select a predefined prompt:", predefined_prompts)
custom_prompt = st.text_area("Or enter your own prompt:")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
   
    if 'Content' in df.columns:
        reviews = df['Content'].tolist()

        final_prompt = custom_prompt if custom_prompt.strip() else selected_prompt

        if st.button("Analyze"):
            with st.spinner("Analyzing employee data..."):
                result = analyze_feedback(reviews, final_prompt)
           
            st.subheader("Analysis Report")
            st.write(result)
    else:
        st.error("CSV must contain a 'Content' column.")
