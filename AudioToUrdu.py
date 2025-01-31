import streamlit as st
import whisper
from deep_translator import GoogleTranslator
import textwrap
import os

def transcribe_audio(audio_path):
    model = whisper.load_model("turbo")
    result = model.transcribe(audio_path)
    return result["text"], result

def detect_language(model, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)

def translate_to_urdu(text, chunk_size=1000):
    translator = GoogleTranslator(source='en', target='ur')
    chunks = textwrap.wrap(text, chunk_size)
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return " ".join(translated_chunks)

st.title("Whisper AI Transcription & Urdu Translation")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    audio_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(audio_path, format="audio/mp3")

    with st.spinner("Transcribing..."):
        model = whisper.load_model("turbo")
        transcribed_text, result = transcribe_audio(audio_path)
        detected_lang = detect_language(model, audio_path)

    st.subheader("Transcribed Text:")
    st.text_area("", transcribed_text, height=150)
    st.write(f"**Detected Language:** {detected_lang}")

    with st.spinner("Translating to Urdu..."):
        urdu_translation = translate_to_urdu(transcribed_text)

    st.subheader("Urdu Translation:")
    st.text_area("", urdu_translation, height=150)

    with open("urdu_translation.txt", "w", encoding="utf-8") as file:
        file.write(urdu_translation)

    st.success("Translation saved as 'urdu_translation.txt'")

    os.remove(audio_path)
