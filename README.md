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
The development of this project initially relied on doctor-patient conversation audios from the study "PriMock57: A Dataset Of Primary Care Mock Consultations". This dataset provided a foundational resource for training the speech-to-text model within a healthcare context. However, as it primarily focuses on general medical dialogues, an expansion with physiotherapy-specific data is proposed. This enhancement aims to improve the model’s accuracy and contextual relevance, ensuring more precise transcriptions adapted to the documentation needs of physiotherapists during clinical evaluations.

### Transcription task



### Summarization task




### Visualization
- **Project visualization:** https://

The streamlit can be seen on the link above and also accessing by the following command line on the Anaconda prompt:

`streamlit run physio-app.py`

### Demo



### References 

Korfiatis A.P, Sarac R., Moramarco F., Savkov A. (2022). *PriMock57: A Dataset Of Primary Care Mock Consultations.* Available at: https://arxiv.org/abs/2204.00333 (Accessed: February 2025)


https://distill.pub/2017/ctc/ 

https://huggingface.co/blog/fine-tune-wav2vec2-english

https://medium.com/@heyamit10/practical-guide-on-fine-tuning-wav2vec2-7c343d5d7f3b

https://platform.openai.com/docs/guides/speech-to-text

