####################################################
# Step 1: Importing the Libraries
####################################################
import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


####################################################
# Step 2:Loading the data to use on Wav2Vec2 Model
####################################################
data_directory = "data"
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
# Step 4: Creating the Wav2Vec2 Model
####################################################
# Loading the pre-trained model and processor
# Wav2Vec2ForCTC: Model that predicts the token logits from the input audio data
# Wav2Vec2Processor: Normalizes audio data and tokenizes it into input_ids
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")


####################################################
# Step 5: Setting the device
####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Passing the model to the device
model.to(device)


####################################################
# Step 6: Creating the Dataloader 
####################################################
# Create the dataset
processed_audios = []
for audio_path in audio_files:
    speech, sr = preprocess_audio(audio_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=sr).input_values
    processed_audios.append(input_values.squeeze(0))

processed_audios = [tensor for tensor in processed_audios]
processed_audios = pad_sequence(processed_audios, batch_first=True)

dataset = TensorDataset(processed_audios)

# Create the DataLoader
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


####################################################
# Step 7: Initializing the optimizer and loss function
####################################################
criterion = nn.CTCLoss()
# AdamW: Adam optimizer that incorporates weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001)


####################################################
# step 8: Training loop
####################################################

num_epochs = 2

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0 
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to the device
        input_values = batch[0].to(device)
        # Forward pass
        logits = model(input_values).logits 
        targets = input_values.argmax(dim=-1).unsqueeze(1)  # Ensure targets is 2D
        input_lengths = torch.full((input_values.size(0),), logits.size(1), dtype=torch.long).to(device)
        target_lengths = torch.full((input_values.size(0),), targets.size(1), dtype=torch.long).to(device)
        loss = criterion(logits.transpose(0, 1), targets, input_lengths, target_lengths)
        # Backward pass 
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step() 

        epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss / len(dataloader):.4f}")


####################################################
# Step 9: Evaluating the model
####################################################
with torch.no_grad():  
    logits = model(input_values).logits

# Get the predicted token IDs (logits converted to token indices)
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the token IDs back into text using the processor
transcription = processor.decode(predicted_ids[0])

# Printing the transcription
print(f"Transcription: {transcription}")



####################################################
# Step 10: Saving the model
####################################################
model.save_pretrained("models/wav2vec2_model")
processor.save_pretrained("models/wav2vec2_processor")
