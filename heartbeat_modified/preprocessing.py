import pandas as pd
import librosa as lb
import librosa.display as lbd
import matplotlib.pyplot as plt

df = pd.read_csv("set_b.csv")
df = df.drop("sublabel", axis=1)
df = df.replace(["extrastole", "murmur"], "abnormal")
for index, row in df.iterrows():
    row[1] = row[1].replace('Btraining_','')

df.to_csv("pre-set_b.csv")

# y, sr = lb.load("./set_b/extrastole__127_1306764300147_C2.wav")

# plt.figure(figsize=(10, 5))
# lbd.waveplot(y,sr=sr,x_axis='time')
# plt.title("Waveplot")
# plt.show()

# X = lb.stft(y)
# Xdb = lb.amplitude_to_db(abs(X))
# plt.figure(figsize=(12, 5))
# lbd.specshow(Xdb, sr=sr ,x_axis='time', y_axis='hz')
# plt.ylim(0,2000)
# plt.title("Spectrogram")
# plt.colorbar()
# plt.show()
