import librosa
import soundfile as sf
import numpy as np
import glob
import os
from icecream import ic
# file_name = "/Users/klvijeth/Documents/ML-project/toy_split"
features = ["eeg"] + ["mel"]
sr = 64

all_pths = glob.glob("/home/kunal/eeg_data/derivatives/split_data/train/*.npy")
for i in all_pths:
    eeg = np.load(i)
    # print(i)
    segment_dur_secs = [5,10,30,60] # 5 seconds
    for dur in segment_dur_secs:
        new_pth = i.replace("split_data", f"downsample_{dur}")
        segment_len = dur * sr
        split = []
        if("env" not in new_pth):
            for s in range(0, len(eeg), segment_len):
                t = eeg[s: s + segment_len]
                path = new_pth.split(".")[0].replace(new_pth.split("/")[-1].split(".")[0], "")
                # fn = path+new_pth.split("/")[-1].split(".")[0]+"_-_"+str(s//64)
                new_file_name = new_pth.split("/")[-1].split("_-_")
                new_file_name.insert(3, str(s//64))
                new_file_name = "_-_".join(new_file_name)
                fn = path + new_file_name
                if(len(t)==segment_len):
                    np.save(fn, t)
