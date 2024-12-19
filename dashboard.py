import streamlit as st
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Ensure nltk vader_lexicon is downloaded
try:
    from nltk.data import find
    find('sentiment/vader_lexicon.zip')
except LookupError:
    download('vader_lexicon')

# Function to clean text
def clean_twitter_text(text):
    if isinstance(text, str):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    return ""

# Stopword removal using Sastrawi
# Additional stop words
more_stop_words = [
    # Stopwords khusus WiFi
    "wifi", "internet", "iconnet", "provider", "modem", "koneksi", 
    "signal", "router", "paket", "layanan", "bandwidth", "mbps", 
    "gb", "speed", "ping", "ip", "langganan", "jaringan",

    # Stopwords umum
    "hi", "halo", "bro", "guys", "loh", "deh", "kok", "nih", 
    "dong", "aja", "ya", "kan", "oh", "banget", "nggak", "ngga","iconnet", "lah"]  # Ganti dengan kata-kata yang ingin ditambahkan
# Combine default stop words with additional stop words
stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)
# Create a new dictionary and stop word remover
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

def remove_stopwords(text):
    return stop_words_remover_new.remove(text)

# LDA Topic Modeling
def lda_topic_modeling(texts, n_topics=5, n_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    dt_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dt_matrix)
    
    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda_model.components_):
        topic_words = [words[i] for i in topic.argsort()[-n_words:]]
        topics.append(f"Topic {idx + 1}: " + ", ".join(topic_words))
    return topics

# Streamlit Dashboard
st.title("Dashboard Analisis Sentimen dan Topik")

# File Upload
uploaded_file = st.file_uploader("Upload file data (Excel)", type=["xlsx", "csv"])
if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Asli")
    st.write(df.head())

    # Drop missing values and duplicates
    df = df.drop_duplicates(subset=['full_text'])
    df = df.dropna(subset=['full_text'])
    df = df[df['username'] != 'pln_123']

    st.subheader("Data Setelah Membersihkan Missing Values dan Duplikasi")
    st.write(df.head())

    # Data cleaning
    st.subheader("Data Setelah Cleaning")
    if 'full_text' in df.columns:
        # Apply cleaning
        df['clean_text'] = df['full_text'].apply(clean_twitter_text)
        # Remove stopwords
        df['clean_text'] = df['clean_text'].apply(remove_stopwords)
        st.write(df[['full_text', 'clean_text']].head())
    else:
        st.error("Kolom 'full_text' tidak ditemukan.")
    
    # Sentiment Analysis
    st.subheader("Analisis Sentimen")
    # Contoh kamus sentimen Bahasa Indonesia
    sentiment_dictionary = {
        # Sentimen Positif
        "baik": "positif",
        "bagus": "positif",
        "indah": "positif",
        "senang": "positif",
        "bahagia": "positif",
        "mantap": "positif",
        "keren": "positif",
        "cepat": "positif",
        "stabil": "positif",
        "puas": "positif",
        "lancar": "positif",
        "handal": "positif",
        "murah": "positif",
        "terjangkau": "positif",
        "unggul": "positif",
        "hebat": "positif",
        "efisien": "positif",
        "gratis": "positif",
        "mulus": "positif",
        "menyenangkan": "positif",
        "sukses": "positif",
        "aman": "positif",
        "rekomendasi": "positif",
        "luar biasa": "positif",
        "kualitas": "positif",
        "memuaskan": "positif",
        "solusi": "positif",
        "senyum": "positif",
        "bantu": "positif",
        "ramah": "positif",
        "support": "positif",
        "loyal": "positif",
        "perfect": "positif",

        # Sentimen Negatif
        "buruk": "negatif",
        "trouble": "negatif",
        "jelek": "negatif",
        "sedih": "negatif",
        "marah": "negatif",
        "kecewa": "negatif",
        "lambat": "negatif",
        "putus": "negatif",
        "lemot": "negatif",
        "lelet": "negatif",
        "mahal": "negatif",
        "error": "negatif",
        "tidak stabil": "negatif",
        "gangguan": "negatif",
        "terputus": "negatif",
        "frustasi": "negatif",
        "mengecewakan": "negatif",
        "tidak responsif": "negatif",
        "mati": "negatif",
        "lemotnya": "negatif",
        "masalah": "negatif",
        "payah": "negatif",
        "lambannya": "negatif",
        "penipuan": "negatif",
        "rip": "negatif",
        "sulit": "negatif",
        "kacau": "negatif",
        "kurang": "negatif",
        "tidak puas": "negatif",
        "males": "negatif",
        "hilang": "negatif",
        "parah": "negatif",
        "capek": "negatif",
        "rip": "negatif"}

    # Menentukan sentimen menggunakan kamus sentimen
    def label_sentiment(text):
        positif_count = sum(word in sentiment_dictionary and sentiment_dictionary[word] == "positif" for word in text.split())
        negatif_count = sum(word in sentiment_dictionary and sentiment_dictionary[word] == "negatif" for word in text.split())
        
        if positif_count > negatif_count:
            return "positif"
        elif negatif_count > positif_count:
            return "negatif"
        else:
            return "netral"

    # Menambahkan kolom sentimen ke dataframe
    df["sentiment"] = df["clean_text"].apply(label_sentiment)

    # Menampilkan hasil analisis dalam tabel
    st.write("Hasil Analisis Sentimen")
    st.dataframe(df[['clean_text', 'sentiment']].head())

    # Visualisasi hasil analisis
    st.subheader("Distribusi Sentimen")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    
    # WordCloud untuk Sentimen Positif dan Negatif
    st.subheader("WordCloud Berdasarkan Sentimen")

    # WordCloud untuk kata-kata dengan sentimen positif
    positive_words = " ".join(df[df['sentiment'] == 'positif']['clean_text'])
    wordcloud_positive = WordCloud(width=800, height=400, background_color="white").generate(positive_words)

    st.write("WordCloud untuk Sentimen Positif")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # WordCloud untuk kata-kata dengan sentimen negatif
    negative_words = " ".join(df[df['sentiment'] == 'negatif']['clean_text'])
    wordcloud_negative = WordCloud(width=800, height=400, background_color="white").generate(negative_words)

    st.write("WordCloud untuk Sentimen Negatif")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    
    # LDA Topic Modeling
    st.subheader("Analisis Topik dengan LDA")
    n_topics = st.slider("Jumlah Topik", 2, 10, 3)
    if df['clean_text'].str.len().sum() > 0:
        lda_topics = lda_topic_modeling(df['clean_text'], n_topics=n_topics)
        st.write("Topik yang dihasilkan:")
        for topic in lda_topics:
            st.write(topic)
    else:
        st.error("Data tidak cukup untuk analisis topik.")

else:
    st.info("Silakan upload file data untuk memulai analisis.")
