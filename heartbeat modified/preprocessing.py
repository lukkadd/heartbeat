import pandas as pd
import librosa as lb
import librosa.display as lbd
import matplotlib.pyplot as plt

df = pd.read_csv("set_b.csv")
df = df.drop("sublabel", axis=1)
df = df.replace(["extrastole","murmur"],"abnormal")
df.to_csv("pre-set_b.csv")

y, sr = lb.load("./set_b/extrastole__127_1306764300147_C2.wav",duration=4)
dur = lb.get_duration(y)

X = lb.stft(y)
Xdb = lb.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
lbd.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')
plt.colorbar()
plt.show()

print("duration: ", dur)

plt.figure(figsize=(16,3))
lbd.waveplot(y,sr=sr)
plt.show()
