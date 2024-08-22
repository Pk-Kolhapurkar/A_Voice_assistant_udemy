import pyttsx3
import speech_recognition as sr
from playsound import playsound
import random
import datetime
hour = datetime.datetime.now().strftime('%H:%M')
#print(hour)
date = datetime.date.today().strftime('%d/%B/%Y')
#print(date)
date = date.split('/')
#print(date)
import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from modules import commands_answers, load_agenda
commands = commands_answers.commands
answers = commands_answers.answers
#print(commands)
#print(answers)

my_name = 'Bob'

# MacOS
chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
# Windows
#chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
# Linux
# chrome_path = '/usr/bin/google-chrome %s'

def search(sentence):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + sentence)

#search('python programming language')

MODEL_TYPES = ['EMOTION']
def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = list(['calm', 'happy', 'fear', 'nervous', 'neutral', 'disgust', 'surprise', 'sad'])
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

#print(load_model_by_name('EMOTION'))
#print(load_model_by_name('EMOTION')[0].summary())

model_type = 'EMOTION'
loaded_model = load_model_by_name(model_type)

def predict_sound(AUDIO, SAMPLE_RATE, plot = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr = SAMPLE_RATE)
    #print(wav_data.shape)
    #print(sample_rate)
    #print(wav_data)
    # ' librosa ' -> 'librosa'
    # https://librosa.org/doc/main/generated/librosa.effects.trim.html
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end = True, pad_value = 0)
    for i, data in enumerate(splitted_audio_data.numpy()):
        #print('Audio split: ', i)
        #print(data.shape)
        #print(data)
        # Mel frequency: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
        # PCA
        mfccs_features = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc=40)
        #print(mfccs_features.shape)
        #print(mfccs_features)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        #print(mfccs_scaled_features.shape)
        mfccs_scaled_features = mfccs_scaled_features[:, :, np.newaxis]
        # batch
        #print(mfccs_scaled_features.shape)
        predictions = loaded_model[0].predict(mfccs_scaled_features)
        #print(predictions)
        #print(predictions.sum())
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            plt.show()

        predictions = predictions.argmax(axis = 1)
        #print(predictions)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)
        #print(results)

        result_str = 'PART ' + str(i) + ': ' + str(predictions).upper()
        #print(result_str)

    count_results = [[results.count(x), x] for x in set(results)]
    #print(count_results)

    #print(max(count_results))
    return max(count_results)

#playsound('sad.wav')
#predict_sound('sad.wav', loaded_model[2], plot=False)

def play_music_youtube(emotion):
    play = False
    if emotion == 'sad' or emotion == 'fear':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=k32IPg4dbz0&ab_channel=Amelhorm%C3%BAsicainstrumental')
        play = True
    if emotion == 'nervous' or emotion == 'surprise':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=pWjmpSD-ph0&ab_channel=CassioToledo')
        play = True
    return play

#play_music_youtube('sad')
#play_music_youtube('surprise')
#emotion = predict_sound('sad.wav', loaded_model[2], plot=False)
#print(emotion)
#play_music_youtube(emotion[1])

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 90) # number of words per second
    engine.setProperty('volume', 1) # min: 0, max: 1
    engine.say(text)
    engine.runAndWait()

#speak("Testing the Assistant's Speech Synthesizer")

def listen_microphone():
    microphone = sr.Recognizer()
    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source, duration=0.8)
        print('Listening: ')
        audio = microphone.listen(source)
        with open('recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
    try:
        # https://pypi.org/project/SpeechRecognition/
        sentence = microphone.recognize_google(audio, language='en-US')
        print('You said: ' + sentence)
    except sr.UnknownValueError:
        sentence = ''
        print('Not understood')
    return sentence

#playsound('recordings/speech.wav')
#listen_microphone()

def test_models():
    audio_source = '/Users/jonesgranatyr/Documents/Ensino/IA Expert/Cursos/Virtual assistent/virtual_assistant/recordings/speech.wav'
    prediction = predict_sound(audio_source, loaded_model[2], plot = False)
    return prediction

#print(test_models())

playing = False
mode_control = False
print('[INFO] Ready to start!')
playsound('n1.mp3')

while (1):
    result = listen_microphone()

    if my_name in result:
        result = str(result.split(my_name + ' ')[1])
        result = result.lower()
        #print('The assistant has been activacted!')
        #print('After processing: ', result)

        if result in commands[0]:
            playsound('n2.mp3')
            speak('I will read my list of functionalities: ' + answers[0])

        if result in commands[3]:
            playsound('n2.mp3')
            speak('It is now ' + datetime.datetime.now().strftime('%H:%M'))

        if result in commands[4]:
            playsound('n2.mp3')
            speak('Today is ' + date[0] + ' of ' + date[1])

        if result in commands[1]:
            playsound('n2.mp3')
            speak('Please, tell me the activity!')
            result = listen_microphone()
            annotation = open('annotation.txt', mode='a+', encoding='utf-8')
            annotation.write(result + '\n')
            annotation.close()
            speak(''.join(random.sample(answers[1], k = 1)))
            speak('Want me to read the notes?')
            result = listen_microphone()
            if result == 'yes' or result == 'sure':
                with open('annotation.txt') as file_source:
                    lines = file_source.readlines()
                    for line in lines:
                        speak(line)
            else:
                speak('Ok!')

        if result in commands[2]:
            playsound('n2.mp3')
            speak(''.join(random.sample(answers[2], k = 1)))
            result = listen_microphone()
            search(result)

        if result in commands[6]:
            playsound('n2.mp3')
            if load_agenda.load_agenda():
                speak('These are the events for today:')
                for i in range(len(load_agenda.load_agenda()[1])):
                    speak(load_agenda.load_agenda()[1][i] + ' ' + load_agenda.load_agenda()[0][i] + ' schedule for ' + str(load_agenda.load_agenda()[2][i]))
            else:
                speak('There are not events for today considering the current time!')

        if result in commands[5]:
            mode_control = True
            playsound('n1.mp3')
            speak('Emotion analysis mode has been activacted!')

        if mode_control:
            analyse = test_models()
            print(f'I heard {analyse} in your voice!')
            if not playing:
                playing = play_music_youtube(analyse[1])

        if result == 'turn off':
            playsound('n2.mp3')
            speak(''.join(random.sample(answers[4], k = 1)))
            break
    else:
        playsound('n3.mp3')



















