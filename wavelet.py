# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.gridspec as gridspec

import matplotlib.cm as cm

import pandas as pd
import os
import glob

all_list = glob.glob(r"C:\Users\sallyyeh\Desktop\Bearing1_1/*.csv") #資料夾路徑

index = 1
all_df = pd.DataFrame()
for i in all_list:
    acc_one = pd.read_csv(i,header=None)
    acc_one["id"] = index
    acc_one.columns = ["Hour","Minute","second","u-Second","Horiz_acc","vert_accel","id"]
    index += 1
    all_df = pd.concat([all_df, acc_one],axis=0)


fs=2560 # 一秒幾筆資料
s_length= 99 # 總共幾筆資料
x = np.arange(0, s_length, 1/fs) 
y = all_df['Horiz_acc'] #放入變數
xsl = x.T[::10] 


m=6 #morlet中的波數

max_freq = 1000 #最大頻率 
list1 = range(1,max_freq, 10) #morlet的頻率

print("start") #開始轉換
for freq in list1:
    mw4=signal.morlet(int(fs/freq*m*2), m,1.0, complete=True)
    corr_tmp=(np.correlate(y,mw4)/sum(abs(mw4)))
    
    sup_first=np.zeros(int(len(mw4)/2))
    sup_end=np.zeros(fs*s_length-len(corr_tmp)-int(len(mw4)/2))
    #print (freq,fs*s_length,-len(corr_tmp),-int(len(mw4)/2),fs*s_length-len(corr_tmp)-int(len(mw4)/2))
    corr_tmp=np.append(sup_first,corr_tmp)
    corr_tmp=np.append(corr_tmp,sup_end)
    if freq==list1[0]:
      corr_stack=abs(corr_tmp[::1])
    else:
      corr_stack=np.vstack((corr_stack, abs(corr_tmp[::1])))

print("end")

fig =plt.figure(figsize=(8,5))

gs = gridspec.GridSpec(3,2)
ax1 = fig.add_subplot(gs[0,:])
plt.plot(x,y,linewidth=0.5)
plt.title("sample_waveform_2")
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.grid(True)
ax1.axis('tight')
plt.xlim([0,s_length])

ax2 = fig.add_subplot(gs[1:,0])
im = ax2.imshow(corr_stack,extent=[0,s_length,0,max_freq],interpolation='none',cmap='jet',origin='lower')
#print (xmin,xmax,ymin,ymax)
plt.title("Spectrogram(Wavelet)")
plt.xlabel("Time[s]")
plt.ylabel("Frequency[Hz]")
plt.grid(linestyle='--', linewidth=0.5)

print(corr_stack.max())
cmlb= range(0, int(corr_stack.max())+1, int(corr_stack.max()/10.0))
#[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0] #color bar 的標值(可調整但不影響圖,只影響標值)
#cmlb=[0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
#cmlb=[0,0.005,0.01,0.015,0.02,0.025,0.03]

cbar = fig.colorbar(im, ticks=cmlb,orientation='horizontal')
cbar.ax.set_xticklabels(cmlb) 
ax2.axis('tight')
plt.xlim([0,s_length])
plt.ylim([0,max_freq])


ax3 = fig.add_subplot(gs[1:,1])
NFFT = 2560*s_length  # 2560*資料筆數
Fs = fs # the sampling frequency
plt.specgram(y,NFFT=NFFT, Fs=Fs, noverlap=1000,cmap='jet',scale='linear')
plt.grid(linestyle='--', linewidth=0.5)

plt.title("spectrogram(FFT)")
plt.xlabel("Time[s]")
plt.ylabel("Frequency[Hz]")

plt.colorbar(orientation='horizontal')

ax3.axis('tight')
plt.xlim([0,s_length])
plt.ylim([0,max_freq])
fig.tight_layout()

print(corr_stack.shape) #output
print (corr_stack[:,25600]) #印出定時間的資料(以第25600筆為例)