import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# 1. Konfigurasi API Google Generative AI
genai.configure(api_key="AIzaSyCdpJymEcoLDx4Wrm-9WVa0YNg1QuAANzM")

# Membuat custom LLM untuk LangChain
class GoogleGenerativeAI:
    def __init__(self, model_name):
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

# 2. Inisialisasi model Google Generative AI
model = GoogleGenerativeAI(model_name="gemini-1.5-flash")

# 3. Membaca Data dari CSV
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# 4. Membuat Vector Database
def create_vector_db(dataframe, column_name):
    # Konversi kolom teks menjadi embedding
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    texts = dataframe[column_name].tolist()
    vector_db = FAISS.from_texts(texts, embeddings)
    return vector_db

# 5. Streamlit Interface
def main():
    st.title("")
    uploaded_file = st.file_uploader("MYPROject/lets_go_chicken_faq.csv", type="csv")
    
    if uploaded_file:
        # Membaca file CSV
        data = load_data(uploaded_file)
        st.write("Data CSV:")
        st.write(data.head())

        # Pastikan ada kolom teks yang relevan
        if "content" not in data.columns:
            st.error("CSV harus memiliki kolom 'content'.")
            return

        # Membuat vector database
        st.write("Membangun vector database...")
        vector_db = create_vector_db(data, "content")
        st.success("Vector database berhasil dibuat.")

        # Prompt untuk model generative AI
        prompt = st.text_input("Masukkan prompt Anda:", 
                               "di mana lokasi let's go chicken?")
        
        if st.button("Generate"):
            # Menggunakan model untuk menghasilkan hasil
            response = model.generate(prompt)
            st.subheader("Hasil Generasi:")
            st.write(response)
            
            # Pencarian di vector database
            st.write("Hasil pencarian yang relevan dari database:")
            results = vector_db.similarity_search(prompt, k=3)
            for i, result in enumerate(results):
                st.write(f"Result {i + 1}: {result.page_content}")

if __name__ == "__main__":
    main()
