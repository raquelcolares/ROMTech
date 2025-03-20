####################################################
# Step 1: Importing the Libraries
####################################################
import os
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

####################################################
# Step 2: Loading the data
####################################################
data_directory = "data"
transcript_dir = "transcripts" 
audio_files = []

for file in os.listdir(data_directory):
    if file.endswith(".wav") or file.endswith(".mp3"):
        audio_files.append(os.path.join(data_directory, file))

print(f"Audio files: {audio_files}")


####################################################
# Step 3: Preprocessing the Audios Data
####################################################
def preprocess_audio(audio_path, target_sr=16000):
    """
    Load and preprocess the audio data for the Wav2Vec2 model.
    """
    speech, sr = librosa.load(audio_path, sr=target_sr)
    return speech, sr

####################################################
# Step 4: Loading the Pre-trained Wav2Vec2 Model
####################################################
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


####################################################
# Step 5: Setting the device
####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Passing the model to the device
model.to(device)


####################################################
# Step 5: Transcription Process
####################################################
def transcribe_audio(audio_path):
    """
    Transcribes an audio file using the pre-trained Wav2Vec2 model.
    """
    speech, sr = preprocess_audio(audio_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=sr).input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Processing all audio files
for audio_file in audio_files:
    transcript = transcribe_audio(audio_file)
    transcript_file = os.path.join(transcript_dir, os.path.splitext(os.path.basename(audio_file))[0] + ".txt")
    
    with open(transcript_file, "w") as f:
        f.write(transcript)
    
    print(f"Transcription saved: {transcript_file}")
