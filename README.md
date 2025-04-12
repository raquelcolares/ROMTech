# ROMTech
## Speech-to-Text Interface for Physiotherapists

### Authors
Manoj Sharma, Omid Moridnejad, Raquel Colares, Thuvaarakkesh Ramanathan

### Introduction
Physiotherapists have the task not only to analyze and execute procedures to the patients, but also register detailed everything during the patient’s evaluation. Documenting all, not only takes time during the consultation, causing them work overload, but also they end up with less time to focus only on the patient. 
Based on this healthcare need, the goal of this project is to create an interface where they can attach an audio that will provide the transcription for them, avoiding them to write or type during the patient’s appointment.  
This project encompasses a speech-to-text transcription task, with deep unsupervised learning models.

### Problem to Solve:
The documentation workload for physiotherapists takes away valuable time from patient care. Our goal is to develop a speech-to-text transcription system using deep unsupervised learning models to automate this process, improving efficiency and reducing administrative burden.

### Data
The development of this project relied on doctor-patient conversation audios from the study "PriMock57: A Dataset Of Primary Care Mock Consultations".  This dataset served as the primary resource for training the speech-to-text model within a healthcare context. While it focuses on general medical dialogues rather than physiotherapy-specific interactions, it still provided a valuable foundation for building and testing the transcription pipeline. At the time of development, no publicly available datasets containing specically physiotherapy consultation audio were identified.

However, for testing purposes, we recorded our own simulated physiotherapy consultations in five languages (English, French, Portuguese, Persian, and Hindi) to evaluate both the performance of the models and the functionality of the Streamlit interface.

### Transcription task

* Whisper model

For this project, we explored different transcription approaches, and OpenAI’s Whisper model stood out as the most effective. Widely adopted in healthcare applications, Whisper has proven capable of handling real-world, noisy audio while maintaining high transcription accuracy. It performed well on medical dialogues and provided robust multilingual support across 99 languages, making it interesting for our purpose and evaluation. Its strong performance, language coverage, and relevance to clinical contexts made Whisper the ideal choice for establishing a reliable transcription for our project.


### Summarization task

* Autoencoder



* BERT




### Visualization
- **Project visualization:** https://

The streamlit can be seen on the link above and also accessing by the following command line on the Anaconda prompt:

`streamlit run physio-app.py`

### Demo



### References 

Korfiatis A.P, Sarac R., Moramarco F., Savkov A. (2022). *PriMock57: A Dataset Of Primary Care Mock Consultations.* Available at: https://arxiv.org/abs/2204.00333 (Accessed: February 2025)

*Sequence Modeling With CTC* Available at: https://distill.pub/2017/ctc/ (Accessed: February 2025)

Platen, P.V. *Fine-Tune Wav2Vec2 for English ASR with Transformers* Available at: https://huggingface.co/blog/fine-tune-wav2vec2-english (Accessed: February 2025)

OpenAI Platform. *Speech to text.* Available at: https://platform.openai.com/docs/guides/speech-to-text (Accessed: February 2025)

PyTorch. *Cosine Similarity.* Available at: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html (Accessed: April 2025)

