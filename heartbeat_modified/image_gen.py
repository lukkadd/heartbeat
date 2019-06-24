import librosa
import librosa.display
import os, fnmatch
import numpy as np
import matplotlib.pyplot as plt

def load_audio_mfccs(folder, file_names, duration=12, sr=16000):
    data = []
    for file_name in file_names:
        try:
            sound_file=folder+file_name
            print ("load file ",sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load( sound_file, res_type='kaiser_fast') 
            dur = librosa.get_duration(y=X, sr=sr)
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file)
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)
    return data

def load_audio(folder, file_names, duration=12, sr=16000):
    data = []
    i = 0
    for file_name in file_names:
        try:
            sound_file=folder+file_name
            print ("load file ",sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load( sound_file, res_type='kaiser_fast') 
            dur = librosa.get_duration(y=X, sr=sr)
            data.append([file_name,X,sr])
            i = i + 1
            if(i == 20):
                return data
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
    return data

def gen_waveplot(data, path):
    for i in range(0,len(data)):
        plt.figure(figsize=(10, 5))
        librosa.display.waveplot(data[i][1],sr=data[i][2],x_axis='time')
        plt.title("Audio signal amplitude")
        plt.savefig(path+data[i][0].replace('.wav','.jpg'))
        plt.close()

def gen_spectograms(data, path):
    for i in range(0,len(data)):
        plt.figure(figsize=(10, 5))
        X = librosa.stft(data[i][1])
        Xb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xb,sr=data[i][2],x_axis='time',y_axis='hz')
        plt.ylim(0,2000)
        plt.title("Spectrogram")
        plt.savefig("./spectograms/"+data[i][0].replace('.wav','.jpg'))
        plt.close()

  
# ************************* MAIN PROGRAM *************************

AUDIO_DIR = './set_b/'

file_names = fnmatch.filter(os.listdir(AUDIO_DIR), '*.wav')

sound_data = load_audio(folder=AUDIO_DIR,file_names=file_names)

gen_waveplot(sound_data,"./waveplots/")

gen_spectograms(sound_data,"./spectograms/")













