import streamlit as st
import whisper
import torch
import torch.nn as nn
import os
import io
import numpy as np
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Setting the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
# Loading and defining the Whisper model, and a pre-trained sentence embedding 
whisper_model = whisper.load_model("base")
autoencoder_model = Autoencoder()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Loading the weights already trained of the Autoencoder
autoencoder_model.load_state_dict(torch.load("autoencoder.pt", map_location=torch.device("cpu")))
# Setting to eval mode
autoencoder_model.eval()

# Loading and defining the Bert model
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

tokenizer, bert_model = load_bert_model()


# Defining the language options
language = st.selectbox(
    "Select language for transcription:",
    ["en (English)", "fr (French)", "pt (Portuguese)", "fa (Farsi)", "hi (Hindi)"]
)
language_code = language.split(" ")[0]


# Transcribing the audio
def transcribe(uploaded_file):
    audio_format = "wav"
    # Loading audio file and converting to 16kHz because Whisper requires it
    audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format=audio_format)
    audio = audio.set_channels(1).set_frame_rate(16000)
    # Converting audio to numpy array (normalized)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    result = whisper_model.transcribe(samples, language=language_code)
    return result["text"]

# Summarizing using the Autoencoder
def summarize_autoencoder(transcription, autoencoder):
    sentences = transcription.split(". ")
    if len(sentences) <= 2:
        return transcription

    embeddings = sentence_model.encode(sentences)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        encoded = autoencoder.encoder(embeddings_tensor)
        decoded = autoencoder.decoder(encoded)
        errors = torch.mean((decoded - embeddings_tensor) ** 2, dim=1)
        summary_length = min(10, len(sentences))
        top_indices = errors.argsort()[:summary_length]
        summary = " ".join([sentences[i] for i in sorted(top_indices)])

    return summary

# Getting the sentence embedding from BERT 
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

# Summarizing the transcription using BERT
def summarize_with_bert(transcription, summary_length=5, plot=False):
    sentences = transcription.split(". ")
    if len(sentences) <= summary_length:
        return transcription

    sentence_embeddings = [get_sentence_embedding(sent) for sent in sentences]
    sentence_embeddings = np.stack(sentence_embeddings)

    document_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = cosine_similarity([document_embedding], sentence_embeddings)[0]
    top_indices = similarities.argsort()[-summary_length:][::-1]
    selected_sentences = [sentences[i] for i in sorted(top_indices)]

    return ". ".join(selected_sentences), similarities.mean()
    

def main():
    # Setting the Streamlit Interface
    st.title("Healthcare Assistant")
    st.subheader(":blue[Physiotherapy and Occupational Therapy]")

    uploaded_audio = st.file_uploader("**Upload your audio file:**", type=["wav"])
    transcript = None
    summary = None

    left, middle, right = st.columns(3)

    # Transcription and Summarization buttons 
    transcription_option = left.selectbox(label="**Transcription 🏋️‍♀️**", options=["None", "Whisper"])
    summarization_option = middle.selectbox(label="**Summarization 🏋️‍♀️**", options=["None", "Autoencoder", "BERT"])

    if uploaded_audio is not None:
        with st.spinner("Processing..."):
            if transcription_option == "Whisper":
                transcript = transcribe(uploaded_audio)
                st.subheader("Transcript:")
                st.write(transcript)

            if summarization_option in ["Autoencoder", "BERT"]:
                if transcript is None:
                    transcript = transcribe(uploaded_audio)
                if summarization_option == "Autoencoder":
                    summary = summarize_autoencoder(transcript, autoencoder_model)
                    st.subheader("Summary:")                        
                    st.write(summary)
                if summarization_option == "BERT":
                    summary, _ = summarize_with_bert(transcript)
                    st.subheader("Summary:")                        
                    st.write(summary)

    # Download buttons
    if transcript:
        st.download_button("Download Transcript", transcript, file_name="transcript.txt")

    if summary:
        st.download_button("Download Summary", summary, file_name="summary.txt")

    with st.sidebar:
        st.image("logo4.jpg")
        st.markdown("### Contacts:")
        st.markdown("[Manoj Sharma](https://www.linkedin.com/in/manoj-sharma-b46b81aa/)")
        st.markdown("[Omid Moridnejad](https://www.linkedin.com/in/omid-moridnejad-2855a5151/)")
        st.markdown("[Raquel Colares](https://www.linkedin.com/in/raquel-colares-7b1327a0/)")
        st.markdown("[Thuvaarakkesh Ramanathan](https://www.linkedin.com/in/rt-rakesh/)")

if __name__ == "__main__":
    st.sidebar.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #a0a4ac;
        }
    </style>
    """, unsafe_allow_html=True)
    main()

