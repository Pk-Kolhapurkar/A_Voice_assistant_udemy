import streamlit as st
import pyttsx3
import speech_recognition as sr
from playsound import playsound
import random
import datetime
import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from modules import commands_answers, load_agenda

# Initial settings
sns.set()
commands = commands_answers.commands
answers = commands_answers.answers
my_name = 'Bob'

# Paths for browser
chrome_path = 'open -a /Applications/Google\ Chrome.app %s'  # MacOS
# chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'  # Windows
# chrome_path = '/usr/bin/google-chrome %s'  # Linux

# Load model
MODEL_TYPES = ['EMOTION']
def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = list(['calm', 'happy', 'fear', 'nervous', 'neutral', 'disgust', 'surprise', 'sad'])
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

loaded_model = load_model_by_name('EMOTION')

# Functions
def search(sentence):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + sentence)

def predict_sound(AUDIO, SAMPLE_RATE, plot=True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr=SAMPLE_RATE)
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end=True, pad_value=0)
    for i, data in enumerate(splitted_audio_data.numpy()):
        mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)[:, :, np.newaxis]
        predictions = loaded_model[0].predict(mfccs_scaled_features)
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            st.pyplot(plt)

        predictions = predictions.argmax(axis=1)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)

    count_results = [[results.count(x), x] for x in set(results)]
    return max(count_results)

def play_music_youtube(emotion):
    play = False
    if emotion == 'sad' or emotion == 'fear':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=k32IPg4dbz0&ab_channel=Amelhorm%C3%BAsicainstrumental')
        play = True
    if emotion == 'nervous' or emotion == 'surprise':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=pWjmpSD-ph0&ab_channel=CassioToledo')
        play = True
    return play

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 90)  # number of words per second
    engine.setProperty('volume', 1)  # min: 0, max: 1
    engine.say(text)
    engine.runAndWait()

def listen_microphone():
    microphone = sr.Recognizer()
    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source, duration=0.8)
        st.write('Listening...')
        audio = microphone.listen(source)
        with open('recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
    try:
        sentence = microphone.recognize_google(audio, language='en-US')
        st.write('You said: ' + sentence)
    except sr.UnknownValueError:
        sentence = ''
        st.write('Not understood')
    return sentence

def test_models():
    audio_source = 'recordings/speech.wav'
    prediction = predict_sound(audio_source, loaded_model[2], plot=False)
    return prediction

# Streamlit UI
st.title("Virtual Assistant")
st.write("This assistant can perform tasks based on your voice commands.")

if st.button("Activate Assistant"):
    result = listen_microphone()

    if my_name.lower() in result.lower():
        result = str(result.split(my_name + ' ')[1])
        result = result.lower()

        if result in commands[0]:
            speak('I will read my list of functionalities: ' + answers[0])

        elif result in commands[3]:
            speak('It is now ' + datetime.datetime.now().strftime('%H:%M'))

        elif result in commands[4]:
            date = datetime.date.today().strftime('%d/%B/%Y').split('/')
            speak('Today is ' + date[0] + ' of ' + date[1])

        elif result in commands[1]:
            speak('Please, tell me the activity!')
            result = listen_microphone()
            annotation = open('annotation.txt', mode='a+', encoding='utf-8')
            annotation.write(result + '\n')
            annotation.close()
            speak(''.join(random.sample(answers[1], k=1)))
            speak('Want me to read the notes?')
            result = listen_microphone()
            if result == 'yes' or result == 'sure':
                with open('annotation.txt') as file_source:
                    lines = file_source.readlines()
                    for line in lines:
                        speak(line)
            else:
                speak('Ok!')

        elif result in commands[2]:
            speak(''.join(random.sample(answers[2], k=1)))
            result = listen_microphone()
            search(result)

        elif result in commands[6]:
            if load_agenda.load_agenda():
                speak('These are the events for today:')
                for i in range(len(load_agenda.load_agenda()[1])):
                    speak(load_agenda.load_agenda()[1][i] + ' ' + load_agenda.load_agenda()[0][i] + ' schedule for ' + str(load_agenda.load_agenda()[2][i]))
            else:
                speak('There are no events for today considering the current time!')

        elif result in commands[5]:
            st.write('Emotion analysis mode activated!')
            analyse = test_models()
            st.write(f'I heard {analyse} in your voice!')
            play_music_youtube(analyse[1])

        elif result == 'turn off':
            speak(''.join(random.sample(answers[4], k=1)))
            st.write("Assistant turned off.")
