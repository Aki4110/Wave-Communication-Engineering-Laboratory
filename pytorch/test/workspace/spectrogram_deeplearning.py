##################################################################
##################################################################
#
# Function of Spectrogram_deeplearning
# for the Analysis of Radiowave Data
# Using AlexNet or ResNet18
#
# ラジオ波観測データの深層学習による解析
#
# 関数：spectrogram_deeplearning,
#       スペクトログラム生成
#       AlexNet または ResNet18 により深層学習、検証、判別、
#       グラフ表示
#
##################################################################
##################################################################

#######################################################################################################################################
# !!! Attention Re. PyTorch and CUDA Library Installation !!!
#
# If you encounter an error message of
# "Graphics Device with CUDA capability sm_86 is not compatible with the current PyTorch installation."
# then new PyTorch Version of 1.8.1+cu111 is required.
#
# Uninstall old torch and torchvision with:
#
#    pip3 uninstall torch torchvision torchaudio
#
# Install newer version with:
#
#    pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#
# or even newer version with:
#
#    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#
# Then it should work on newer GPUs with cuda-11.0 or cuda-11.1.
#######################################################################################################################################

#
# プログラム中止コマンドは sys.exit()
# input() で一時停止し、Enterで再開できる
# input("Press Enter to Continue...")
#

#
# Variable Acronims, 変数略称の意味:
# TDExt_ = TimeDataExtracted_ = _timedata_ = RadiowaveDataExtracted_ : 観測データから均一に抽出された時間データ（別プログラムにより生成済み）
# SG_ = Spectrogram_ = _spectrogram_                                 : スペクトログラムまたはスカログラム
# TR_ = Train_ = _train_            : 学習
# VR_ = Valid_ = _valid_            : 検証
# JG_ = Judge_ = _judge_            : 判定
# Tm  = time                        : 実時刻        Unixtime (UTC)  [秒]
# Id  = indx                        : インデックス  Index 整数      [1]
# FT                                : Fourier Transform     フーリエ変換（プロセス）
# FI                                : Fourier Integral      フーリエ積分（区間）
#

#
# Libraries/Modules to import: For install "pip3 install xxxxxx"
# 'インポート'はファイルの先頭に置く 'import' must be placed at the top of files
#
import sys
#import pathlib
import os
from datetime import datetime    # "datetime" オブジェクトによる時刻計算（from ... import ... とすれば、使用するときに datetime.datetime.xxx とせずに、datetime.xxx とすることができる）
import time                      # 現在のunixtime取得 = time.time()
import matplotlib.pyplot as plt  # Matlabライクなプロット
import japanize_matplotlib       # プロット日本語化 (pip3 install japanize-matplotlib)
import numpy as np               # 配列・計算にはnumpyを使うようにする
#import math                     # mathは使わないようにする
import pickle                    # 変数ダンプとロード
import shutil                    # シェルユーティリティ
import glob
#import pandas
import csv

#
# 画像処理用ライブラリ
#
from PIL import Image

#
# Deep-Learning Libraries/Modules
#
import torch                     # PyTorch (pip3 install torch) (torch-1.8.1+cu102)
import torchvision               # Torch Vision (pip3 install torchvision) (torchvision-0.9.1+cu102)

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
#from   torch.utils.data import DataLoader
#from   torch.utils.data import TensorDataset
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
#from   torchvision  import datasets, models
import torchvision.transforms as transforms
import torchvision.models as models  # Network Models (ResNet etc.) 
from   torchsummary import summary  # Torch Summary (pip3 install torchsummary) (torchsummary-1.5.1)
from   sklearn.model_selection import train_test_split  # scikit-learn (pip3 install scikit-learn)


#####################################################################################################################################################
## Start of the Function 関数開始 ###################################################################################################################
#####################################################################################################################################################
##
## ラジオ波観測データ読み込み  
## 全期間スペクトログラム（短時間フーリエ変換データ）の生成
## 異常時データリストの読み込み
## 正常時データリストの読み込みまたは異常時データリストの補集合を計算
## 正常区間、異常区間のそれぞれのスペクトログラム/スカログラム画像生成、保存
## 正常時データ生成 (正常 = Normal)
## 異常時データ生成 (異常 = Abnrml)
## Train 学習（訓練）& Validation 検証
## Judge 判定（判別分析）
## 判別結果のグラフ表示と保存
##
#####################################################################################################################################################
#####################################################################################################################################################
def spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                        generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                        perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                        perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                        plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                        Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                        Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq):

    #####################################################
    #####################################################
    # 解析用一時ファイル保存用ディレクトリ名とパスの生成
    #####################################################
    #####################################################
    Miscellaneous_Directory     = "Miscellaneous_" + City_DR_No_BCLoc_Freq                              # 一時ファイル保存用ディレクトリ
    Miscellaneous_DirectoryPath = os.path.join(Spectrogram_DeepLearning_Path, Miscellaneous_Directory)  # 一時ファイル保存用ディレクトリパス
    # ディレクトリがなければ作成
    if not os.path.isdir(Miscellaneous_DirectoryPath): os.makedirs(Miscellaneous_DirectoryPath)  # 1行if文 [if 条件: 処理]   1行if else文 [処理い if 条件 else 処理ろ]

    #############################
    #############################
    ## ラジオ波観測データ読み込み
    #############################
    #############################
    # 観測データから平均リサンプルした全期間の抽出データを'RadiowaveData_ExtractedFile'ファイルから読み込み
    # ファイル指定
    #RadiowaveData_ExtractedFile = "RadiowaveData_Extracted_WholePeriod_Toyama_NE_5_Niigata_82p3MHz.txt"  # Example of the File Name 例
    RadiowaveData_ExtractedFile  = "RadiowaveData_Extracted_WholePeriod_" + City_DR_No_BCLoc_Freq + ".txt"
    RadiowaveData_ExtractedFile_Path = os.path.join(ExtractedTimedata_Path, RadiowaveData_ExtractedFile)
    print("RadiowaveData_ExtractedFile_Path = ",RadiowaveData_ExtractedFile_Path)

    # 観測データ読み込み (Read_RadiowaveData_ExtractedFile)
    print('Opening File and Reading RadiowaveData Extracted WholePeriod Timedata ファイル読み込み中 (takes about 2 sec)...')
    # Open and Read Extracted Timedata File
    fileID = open(RadiowaveData_ExtractedFile_Path,'r')  # ファイル情報取得
    ##########################################################################################################################
    # ファイル情報取得：データ行数カウント count、時間ステップ計算 timestep、データ開始時刻 starttime、終了時刻 endtime の取得
    ##########################################################################################################################
    count = 0
    lines = fileID.readlines()  # 行数取得
    print("len(lines) = ",len(lines))
    #while feof(fileID) ~= 1 # Matlab（feofはファイル末尾のとき1を返す）
    for tmp_line in lines: # １行ずつ読み込み
        #tmp_line = fgetl(fileID); # Matlab １行読み込み
        #print(tmp_line, end="")  # １行表示した後、改行は１度だけにする
        #if not( (tmp_line == "") | (tmp_line(1:1) == "#") ) # Matlab（空行または先頭が#の行）でない場合のみカウント
        if not( (tmp_line == "") or (tmp_line[0] == "#") ):  #（空行または先頭が#の行）でない場合のみカウント
            count = count + 1
            #print("count = ",count)
            if count == 1:
                tmp_unixtime = float(tmp_line[19:30])  # unixtime (UTC)[s]は20~30文字目、第１番目の時刻を取得
                starttime = tmp_unixtime  # unixtime (UTC)[s]を保存：開始時刻
                #print("starttime = ",starttime)
            # end
            if count == 2:
                tmp_unixtime = float(tmp_line[19:30])  # unixtime (UTC)[s]は20~30文字目、第２番目の時刻を取得
                nexttime = tmp_unixtime  # unixtime (UTC)[s]を保存
                #print("nexttime = ",nexttime)
            # end
            #if count == 4:  # チェック用
            #    break
        # end
    # end
    tmp_unixtime = float(tmp_line[19:30])  # unixtime (UTC)[s]は20~30文字目、最終番目の時刻を取得
    endtime = tmp_unixtime  # unixtime (UTC)[s]を保存：終了時刻
    # TimeStep 自動計算
    print("count = ",count)
    timestep1 = nexttime - starttime               # 第１、２番目のデータから時間ステップを計算
    timestep2 = (endtime - starttime)/(count - 1)  # 第１、最終番目のデータから時間ステップを計算
    print("timestep1 = ", timestep1)
    print("timestep2 = ", timestep2)
    eps = 1.0e-14  # 微小値定義
    print("abs(timestep1 - timestep2) = ", abs(timestep1 - timestep2))
    #input("Press Enter to Continue...")
    if abs(timestep1 - timestep2) < eps:  # 実数値の比較には注意が必要、差が微小値以下であれば等しい
        RadiowaveDataExtractedTimeStep = timestep1  # 時間ステップ [s]
    # end
    print("RadiowaveDataExtractedTimeStep  = ",RadiowaveDataExtractedTimeStep)
    #input("Press Enter to Continue...")
    

    # 観測データ開始時刻
    RadiowaveDataExtractedStartTime        = starttime  # unixtime (UTC)[s]
    print("RadiowaveDataExtractedStartTime = ",RadiowaveDataExtractedStartTime)
    # 観測データ修了時刻
    RadiowaveDataExtractedEndTime          = endtime  # unixtime (UTC)[s]
    print("RadiowaveDataExtractedEndTime   = ",RadiowaveDataExtractedEndTime)
    # RadiowaveDataExtracted データ総数（最大数）
    RadiowaveDataCountMax                  = count
    print("count = ",count)
    print("RadiowaveDataCountMax           = ",RadiowaveDataCountMax)

    # 配列準備
    #RadiowaveDataExtracted = [2, RadiowaveDataCountMax]; #RadiowaveData の unixtime(JUTC) vs. dB値, デフォルトは横長の行列 (2行 x count列)
    RadiowaveDataExtracted_utime = [0] * RadiowaveDataCountMax  # unixtime 時刻を格納する配列
    RadiowaveDataExtracted_tdata = [0] * RadiowaveDataCountMax  # dB   電波強度を格納する配列
    #print("RadiowaveDataExtracted_utime = ",RadiowaveDataExtracted_utime)
    #print("RadiowaveDataExtracted_tdata = ",RadiowaveDataExtracted_tdata)

    # Rewind （ファイル巻き戻し、ファイル指定子を先頭へ）and Read data
    #frewind(fileID); # Matlab ファイルの先頭へ戻る 
    fileID.seek(0)    # Python ファイルの先頭へ戻る
    #######################################
    # RadiowaveData データ再読み込み（本番）
    #######################################
    count = 0
    lines = fileID.readlines()  # 行数取得
    #print("len(lines) = ",len(lines))
    #while count <= RadiowaveDataCountMax-1 #（ループ中で１加算するため、１小さい数まで繰り返す）
    for tmp_line in lines: # １行ずつ読み込み
        #tmp_line = fgetl(fileID); #１行読み込み: fgetl or fgets
        #print(tmp_line, end="")  # １行表示した後、改行は１度だけにする
        if not( (tmp_line == "") or (tmp_line[0] == "#") ):  #（空行または先頭が#の行）でない場合はデータ読み込み
            count = count + 1
            #print("count = ",count)
            tmp_unixtime = int(tmp_line[19:30])   # unixtime (UTC)[s]は20~30文字目
            #print("tmp_unixtime = ",tmp_unixtime)
            RadiowaveDataExtracted_utime[count-1] = tmp_unixtime #unixtime (UTC)[s]を保存
            tmp_dB = float(tmp_line[31:37])       # 観測値[dB]は32~37文字目 （正確には dBm ）
            #print("tmp_dB = ",tmp_dB)
            tmp_mW = 10.0**(tmp_dB/10.0)          # 観測値[mW]に変換
            RadiowaveDataExtracted_tdata[count-1] = tmp_mW  # 観測値[dB]ではなくPower[mW]値を取り込み############################################## dB -> mW 変換
            #if count == 4:  # チェック用
            #    break
        # end
    # end
    fileID.close()
    #RadiowaveDataExtracted = RadiowaveDataExtracted'; #RadiowaveData の unixtime (UTC) vs. dB, Transpose: 縦長の行列へ変換 (count行 x 2列)

    #RadiowaveDataExtractedSize      = size(RadiowaveDataExtracted,1);
    RadiowaveDataExtractedSize       = RadiowaveDataCountMax
    #RadiowaveDataExtractedStartTime = RadiowaveDataExtracted(1,1)
    #RadiowaveDataExtractedEndTime   = RadiowaveDataExtracted(RadiowaveDataExtractedSize,1)
    #RadiowaveDataExtractedTimeStep  = 120.0 #[s]
    #RadiowaveDataExtractedTimeStep  = (RadiowaveDataExtractedEndTime - RadiowaveDataExtractedStartTime)/(RadiowaveDataExtractedSize - 1) #自動設定

    print("RadiowaveDataExtractedSize = ",  RadiowaveDataExtractedSize)
    #print("RadiowaveDataExtracted_utime = ",RadiowaveDataExtracted_utime)
    #print("RadiowaveDataExtracted_tdata = ",   RadiowaveDataExtracted_tdata)

    #################################
    #################################
    # ラジオ波観測データ読み込み完了
    #################################
    #################################



    ###########################################################
    ###########################################################
    # Generate Spectrogram by Moving Fourier Transform 
    # 全期間スペクトログラム（短時間フーリエ変換データ）の生成
    #
    # このプログラム内で全期間通して一気にフーリエ変換する
    # ただし、すでにデータファイルがあればファイルから読み込む
    # （後でPythonによるウェーブレット変換も検討）
    ###########################################################
    ###########################################################
    ###############################################################################################################################################################################
    # Moving Fourier Transform 
    # 全期間のスペクトルデータ（短時間フーリエ変換データ）生成：計算時間＝１系統（2015-2021)あたり２〜３時間必要 (real 136m24.878s on i7-5000)
    #
    # フーリエ変換条件および定数
    #
    # Unixtime [sec]: RadiowaveDataExtracted_utime
    # Timestep [sec]: RadiowaveDataExtractedTimeStep
    # Radiowave [dB]: RadiowaveDataExtracted_tdata
    # DataCount  [1]: RadiowaveDataExtractedSize = RadiowaveDataCountMax
    # WindowSize [1]: window_indx_size
    #
    timestep = RadiowaveDataExtractedTimeStep  # Time Step [sec] = 120 sec 等
    ndata    = RadiowaveDataExtractedSize      # Number of Time Data
    window_time_span = 6.0*3600.0              # [s] Time Window Span [sec] = 6 hour 等############################################################################################
    print("window_time_span 時間ウインドウ幅 = ", window_time_span)
    window_indx_size = int(window_time_span//timestep)  # Time Window Size [1] = 540 等 or 6h/120s = 6*3600/120 = 180 等
    # window_indx_size = 540                            # Time Window Size [1], 540*2min = 1080min = 18h 等 #######################################################################
    spectrogram_timestep = 60.0 * 10.0         # =600 [sec] Time Step for Moving Fouirier Transform = 10 min 等、時間移動フーリエスペクトルは１０分毎に計算
    print("spectrogram_timestep スペクトログラム時間ステップ = ", spectrogram_timestep)
    #
    # work => timedata に変更
    #
    indx_timedata_start = 0                                                         # index ０番目の時間データから開始
    time_timedata_start = float(RadiowaveDataExtracted_utime[indx_timedata_start])  # unixtime(indx_timedata_start)
    indx_timedata_end   = ndata - 1                                                 # index 最後の時間データまで
    time_timedata_end   = float(RadiowaveDataExtracted_utime[indx_timedata_end])    # unixtime(indx_timedata_end)
    indx_ratio_tdpersg  = int(spectrogram_timestep//timestep)                       # 600s/120s = 5 steps of Extracted Timedata
    indx_integral_span  = window_indx_size                                          # フーリエ積分幅数[1]
    time_integral_span  = float(window_indx_size)*timestep                          # フーリエ積分時間[s]
    #
    freq_start = 1.0/time_integral_span  # should be ~0.0001d0  # Start frequency [Hz] 1/540/120s = 1.54320987654321e-05 Hz = 0.0000154 Hz
    freq_end   = 1.0/timestep/2.0        # should be ~0.01d0    # End frequency   [Hz] should be less than Nyquist frequency, 1/120s/2 = 0.00417 Hz
    freq_end   = 0.5*freq_end            # Evaluate lower frequency
    nfreq      = window_indx_size        # Number of frequency points
    ###############################################################################################################################################################################

    # Number of time moving points for the moving Fourier transform 移動時間点数
    spectrogram_maxsize = int((ndata - indx_integral_span - 1) // indx_ratio_tdpersg)  # "//" 切り捨て整数化 (Python3), = (ndata - window_indx_size - 1)/indx_ratio_tdpersg
    print('Number of Points for Moving Fourier Transform フーリエ変換の移動時間点数: spectrogram_maxsize = ',spectrogram_maxsize)

    ######################################
    # Define Arrays (numpy配列を予め生成)
    ######################################
    nwindow_size_array       = np.arange(window_indx_size)    # numpy array of range n = 0:window_indx_size-1    numpy配列を繰り返し計算前に予め作っておく
    indx_integral_span_array = np.arange(indx_integral_span)  # numpy array of range n = 0:indx_integral_span-1  numpy配列を繰り返し計算前に予め作っておく

    # Spectrogram変数 スペクトログラムを格納する変数を初期化
    spectrogram  = np.zeros((spectrogram_maxsize, nfreq), dtype=np.float64)  # spectrogram_maxsize行 nf列の行列を０で初期化、２次元配列：どちらでもOK
    #spectrogram = np.zeros((nfreq, spectrogram_maxsize), dtype=np.float)  # nf行 spectrogram_maxsize列の行列を０で初期化、２次元配列：どちらでもOK
    #indx_conversion_sg2td = [0] * spectrogram_maxsize              # index 各フーリエ変換時点のインデックスを格納、整数型配列
    #indx_conversion_sg2td = np.zeros((spectrogram_maxsize), dtype=np.int64) # 各フーリエ変換時点のインデックスを初期化：forを使わなければ不要
    dummy_spectrum = np.zeros((window_indx_size))  # dummy spectrum array
    # Generate index of moving Fourier transform points 'indx_conversion_sg2td()' 各フーリエ変換時点のインデックスを生成
    # for ns in range(spectrogram_maxsize):  # 0:spectrogram_maxsize-1
    #     #indx_conversion_sg2td[ns] = indx_timedata_start + window_indx_size/2 + 1 + ns*indx_ratio_tdpersg 
    #     indx_conversion_sg2td[ns] = indx_timedata_start + window_indx_size/2 + ns*indx_ratio_tdpersg  # Python3: ０始まり、１小さい数で終わる
    # # end for
    spectrogram_idarray = np.arange(spectrogram_maxsize)  # 0,1,2,..,spectrogram_maxsize-1
    #print('ns = ',ns)
    #indx_conversion_sg2td_tmp = indx_timedata_start + window_indx_size/2 + ns*indx_ratio_tdpersg  # intの演算 -> float
    #indx_conversion_sg2td     = indx_conversion_sg2td_tmp.astype(np.int64)                        # float -> int に変換 ".astype(np....)"を使う
    #indx_conversion_sg2td      = (indx_timedata_start + window_indx_size/indx_ratio_tdpersg/2 + spectrogram_idarray*indx_ratio_tdpersg).astype(np.int64)  # numpy配列をint64に変換: ".astype(np....)"を使う WRONG! 
    #
    # Corrected on 20210318
    indx_conversion_sg2td = (indx_timedata_start + window_indx_size/2 + spectrogram_idarray*indx_ratio_tdpersg).astype(np.int64)  # numpy配列をint64に変換: ".astype(np....)"を使う ################################ Debugged!!
    #
    print('indx_conversion_sg2td = ',indx_conversion_sg2td)
    #time_work = (RadiowaveDataExtracted_utime[indx_conversion_sg2td]).astype(np.float64)  # unixtime(indx_timedata_start) NG
    #time_work = RadiowaveDataExtracted_utime[indx_conversion_sg2td]  # unixtime(indx_timedata_start) NG
    #print('time_work = ',time_work)

    ############################################################
    # Generate time windows 時間ウィンドウ生成
    # nwin = 0:no window / 1:Hamming / 2:Hanning / 3:Blackman
    ############################################################
    timewindow = np.zeros((window_indx_size, 4))  # Declaration needed 宣言/定義必要
    # for n in range(window_indx_size):  # n = 0,window_indx_size-1
    #   timewindow[n,0] = 1.0    # No window
    #   timewindow[n,1] = 0.54 - 0.46*np.cos(2.0*np.pi*float(n)/float(window_indx_size-1)) # Hamming 
    #   timewindow[n,2] = 0.50 - 0.50*np.cos(2.0*np.pi*float(n)/float(window_indx_size-1)) # Hanning
    #   timewindow[n,3] = 0.42 - 0.50*np.cos(2.0*np.pi*float(n)/float(window_indx_size-1)) + 0.08*np.cos(4.0*np.pi*float(n)/float(window_indx_size-1)) #Blackman
    # # end for
    #n = np.arange(window_indx_size)
    timewindow[:,0] = 1.0    # No window
    timewindow[:,1] = 0.54 - 0.46*np.cos(2.0*np.pi*nwindow_size_array/float(window_indx_size-1))  # Hamming 
    timewindow[:,2] = 0.50 - 0.50*np.cos(2.0*np.pi*nwindow_size_array/float(window_indx_size-1))  # Hanning
    timewindow[:,3] = 0.42 - 0.50*np.cos(2.0*np.pi*nwindow_size_array/float(window_indx_size-1)) + 0.08*np.cos(4.0*np.pi*nwindow_size_array/float(window_indx_size-1)) #Blackman
    #####################
    # Check Time Windows
    #####################
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(nwindow_size_array,timewindow[:,0])
    # ax.plot(nwindow_size_array,timewindow[:,1])
    # ax.plot(nwindow_size_array,timewindow[:,2])
    # ax.plot(nwindow_size_array,timewindow[:,3])
    # plt.show()
    # input("Press Enter to Continue...")
    # plt.clf()
    # plt.close()

    ###################################################################################
    # Generate DFT frequency list 周波数リスト freq_list, 角周波数リスト omg_list 生成
    ###################################################################################
    #freq_list  = np.zeros((nfreq))          # Initialize frequency list, Max 'nfreq' points
    #freq_resolution = (freq_end-freq_start)/float(nfreq-1) # Frequency resolution for linear scale [Hz]
    # for m in range(nfreq):  # 0:nfreq-1
    #     #freq_list(m) = freq_start+freq_resolution*dble(m-1)                                                     #Frequency points of linear scale  ##############
    #     freq_list[m]  = freq_start * 10.0**( float(m)/float(nfreq-1)*(np.log10(freq_end) - np.log10(freq_start)) )  # log scale, should be right ####################
    # # end for
    m = np.arange(nfreq)  # 配列で計算する場合：整数型 array range of m = 0:nfreq-1 
    #am = np.arrange(nfreq, dtype='float')  # float型の配列生成 0:nfreq-1 NOT WORK#
    #m_float = m.astype(np.float)  # 整数mをfloatへ変換する場合
    #freq_list = freq_start * 10.0**( m.astype(np.float)/float(nfreq-1)*(np.log10(freq_end) - np.log10(freq_start)) )  # log scale frequency [Hz], mをfloatへ変換: m.astype(np.float)
    freq_list = freq_start * 10.0**( m/float(nfreq-1)*(np.log10(freq_end) - np.log10(freq_start)) )  # log scale frequency [Hz], numpy配列 m は整数型のままで計算可能
    omg_list = 2.0*np.pi*freq_list  # numpy配列 角周波数
    #print('nfreq = ',nfreq)
    #print('m = ',m)
    #print('freq_list = ',freq_list)
    #######################
    # Check Frequency List
    #######################
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(m,freq_list)
    # ax.plot(m,omg_list)
    # plt.show()
    # input("Checking freq_list ....")  # OK
    # plt.clf()
    # plt.close()

    ######################################################################################################################################################
    # Check if pickled spectrogram data file exists, if yes, read the file
    # すでにピクル（バイナリー保存）されたスペクトログラムデータファイル Miscellaneous_DirectoryPath/"SpectrogramDump.pcl" があるか確認し、あれば読み込む
    ######################################################################################################################################################
    #  スペクトログラムを更新せず、            スペクトログラムデータファイルがあり、      そのサイズがゼロより大きく、                     本番解析であれば、スペクトログラムを読み込む
    #if (not update_whole_spectrogram == 1) and (os.path.isfile("SpectrogramDump.pcl")) and (os.path.getsize("SpectrogramDump.pcl") > 0) and (analyze_actual_spectrogram == 1):  
    #  スペクトログラムを更新せず、            スペクトログラムデータファイルがあり、                                                そのサイズがゼロより大きく、                                                               本番解析であれば、スペクトログラムを読み込む
    if (not update_whole_spectrogram == 1) and (os.path.isfile(os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump.pcl"))) and (os.path.getsize(os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump.pcl")) > 0) and (analyze_actual_spectrogram == 1):  # Debuged 20210901
        ############################################
        # スペクトログラムデータの読み込み（ロード）
        ############################################
        # 変数                 数値・長さ                   意味                形状                data-type,データ型
        # nfreq                540 等                       周波数ポイント数    スカラー            numpy.int (=numpy.int64)
        # freq_list            [nfreq]                      周波数              numpy配列（１次元） numpy.float, numpy.ndarray (=numpy.float64)
        # spectrogram_maxsize  約321000                     時間移動点数        スカラー            numpy.int (=numpy.int64)
        # spectrogram          [spectrogram_maxsize, nfreq] スペクトログラム    numpy配列（２次元） numpy.float, numpy.ndarray (=numpy.float64)

        # file load ファイル読み込み
        print('Reading Moving Fourier Transform Data for Whole Period 全期間の時間移動フーリエ変換データ読み込み中 ....')
        #with open('SpectrogramDump.pcl', mode='rb') as fdmp:  # SpectrogramDump.pcl ファイル
        with open(os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump.pcl"), mode='rb') as fdmp:  # SpectrogramDump.pcl ファイル  # Debuged 20210901
            spectrogram = pickle.load(fdmp)                                                             # からスペクトログラムデータをロード
        # with open の後は自動的に close される
    #
    #######################################################################
    # When spectrogram data file does NOT exist, perform Fourier transform
    # スペクトログラムデータファイルがない場合にフーリエ変換を計算
    #######################################################################
    #    スペクトログラム更新フラグが１で、  本番解析の場合、期間全体のスペクトログラムを計算しピクル（保存用バイナリー）ファイルに保存する
    elif (update_whole_spectrogram == 1) and (analyze_actual_spectrogram == 1):  

        #####################################################################################################################
        # ここからフーリエ変換: Moving Fourier Transform for the Whole Timedata 全期間の時間移動フーリエ変換（計算１〜２時間）
        #####################################################################################################################
        print('Performing Moving Fourier Transform for Whole Period 全期間の時間移動フーリエ変換開始 ....')

        for ns in range(spectrogram_maxsize):  # ns = 0,spectrogram_maxsize-1
            #print('ns = ',ns)
            #indx_conversion_sg2td[ns] = indx_timedata_start + window_indx_size/2 + 1 + ns*indx_ratio_tdpersg  # index of moving points
            #indx_conversion_sg2td[ns] = int(indx_timedata_start + window_indx_size/2 + ns*indx_ratio_tdpersg)  # Python3: ０始まり、１小さい数で終わる：インデックスはすべてint型指定必要
            #time_work = float(iunixtime(indx_conversion_sg2td(ns)))                    # Actual time (unixtime, UTC) of the moving points
            #print("indx_conversion_sg2td[ns] = ", indx_conversion_sg2td[ns])
            time_work = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[ns]])  # current working time 現在作業中の時刻 in Unixtime [s] (UTC)
            #print('time_work = ',time_work)
            #For Fourier integral
            #indx_integral_start = int(indx_conversion_sg2td[ns] - indx_integral_span/2)  # = indx_conversion_sg2td(ns) - window_indx_size/2  ################### ERROR! マイナスになる
            indx_integral_start = int(indx_conversion_sg2td[ns] - indx_integral_span/2)
            #print('indx_integral_start = ',indx_integral_start)
            #time_integral_start = float(iunixtime(indx_integral_start))
            time_integral_start = float(RadiowaveDataExtracted_utime[indx_integral_start])
            #print('time_integral_start = ',time_integral_start)
            #indx_integral_end  = indx_integral_start + indx_integral_span - 1  # = indx_integral_start + window_indx_size - 1
            indx_integral_end   = int(indx_integral_start + indx_integral_span)  # = indx_integral_start + window_indx_size （１大きい数）
            #time_integral_end  = float(iunixtime(indx_integral_end))
            time_integral_end   = float(RadiowaveDataExtracted_utime[indx_integral_end])
            #################################
            # Check moving window parameters
            #################################
            if (int(ns/1000)*1000 == ns):
                print('Progress: 進捗状況')
                print('Moving window for Fourier transform')
                print('Current working point of movement ns / spectrogram_maxsize = ', ns, '/', spectrogram_maxsize)
                print('indx_conversion_sg2td[ns], time_work (UTC) [s]             = ', indx_conversion_sg2td[ns], time_work    )  # = dble(iunixtime(indx_conversion_sg2td(ns))) #Actual unixtime (UTC)
                print('indx_integral_start,       time_integral_start             = ', indx_integral_start, time_integral_start)
                print('window_indx_size                                           = ', window_indx_size )
                print('indx_integral_span,        time_integral_span              = ', indx_integral_span,  time_integral_span )
                print('indx_integral_end,         time_integral_end               = ', indx_integral_end,   time_integral_end  )
            # end if

            ###############################################
            # Pick time data section for Fourier transform 
            ###############################################
            #radiowave_data_for_ft = np.zeros(indx_integral_span)
            #for n in range(indx_integral_span):  # n = 0:window_indx_size-1
            #    #radiowave_data_for_ft(n) = radiowave_data(indx_integral_start + n - 1)
            #    radiowave_data_for_ft[n] = RadiowaveDataExtracted_tdata[indx_integral_start + n]
            ## end for
            radiowave_data_for_ft = RadiowaveDataExtracted_tdata[indx_integral_start:indx_integral_start+indx_integral_span]
            #print('radiowave_data_for_ft = ',radiowave_data_for_ft)
            #print('len(radiowave_data_for_ft) = ',len(radiowave_data_for_ft))

            #####################################################################
            # Apply window function to time data (in linear scale of power [mW])
            #####################################################################
            #write (*,'(a)') 'Enter time window: nwin = 0:no window / 1:Hamming / 2:Hanning / 3:Blackman'
            nwin = 1  #3 #2 #1 ## Blackman window '3' has the sharpest resolution of the time variation ################################################# Time Window
            # for n in range(window_indx_size):  # n=0,window_indx_size-1
            #     radiowave_data_for_ft[n] = radiowave_data_for_ft[n]*timewindow[n,nwin]
            # # end for
            radiowave_data_for_ft = radiowave_data_for_ft * timewindow[:,nwin]
            ##################
            # Check Time Data
            ##################
            # t = np.arange(window_indx_size)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(t,radiowave_data_for_ft)
            # plt.show()
            # input()
            # input("Press Enter to Continue...")
            # plt.clf()
            # plt.close()

            ###########################################################
            # Perform Fourier transform by DFT for log scale frequency
            ###########################################################
            #print('Performing Fourier transform by DFT フーリエ変換開始 ....')
            #freq_start = 1.0/time_integral_span   #~0.0001d0 #Start frequency [Hz] 1/512/60s = 3.26e-5 = 0.0000325 Hz, 1/256/60s = 0.0000651 Hz
            #freq_end   = 1.0/timestep/2.0         #~0.01d0   #End frequency   [Hz] should be less than Nyquist frequency, 1/60s/2 = 0.0083 Hz
            #nfreq         = window_indx_size             #Number of frequency points
            #freq_resolution = (freq_end-freq_start)/dble(nfreq-1) #Frequency resolution for linear scale [Hz]

            #WRONG spectrum_data = np.zeros((nfreq,nfreq), dtype=np.complex)  # 複素配列初期化 OK: use "dtype=" WRONG
            spectrum_data = np.zeros((nfreq), dtype=np.complex)  # 複素配列初期化 OK: use "dtype=np.complex"

            #nindx_integral_span_array = np.arange(indx_integral_span)

            for m in range(nfreq):  # m = 0,nfreq-1
                #freq_list(m) = freq_start+freq_resolution*dble(m-1)                 #Frequency points of linear scale  ############################################
                #freq_list(m) = 10.0d0**( log10(freq_start) + dble(m-1)/dble(nfreq-1)*log10(freq_end/freq_start)  ) #log scale, should be right #######################
                #freq_list(m) = freq_start * 10.0d0**( dble(m-1)/dble(nfreq-1)*(log10(freq_end) - log10(freq_start)) ) #log scale, should be right ####################
                #omg = 2.0*np.pi*freq_list[m]
                #r = 0.0
                #x = 0.0
                #total_period = 0.0d0

                #
                # cf. simpler DFT calculation
                #
                # for n in range(indx_integral_span):  # n = 0,indx_integral_span-1  # forループを使用すると非常に遅い
                #     r = r + radiowave_data_for_ft[n]*np.cos(omg*float(n)*timestep)
                #     x = x - radiowave_data_for_ft[n]*np.sin(omg*float(n)*timestep)
                # # end for
                #n = np.arange(indx_integral_span)
                r = np.sum(radiowave_data_for_ft*np.cos(omg_list[m]*indx_integral_span_array*timestep))  # indx_integral_span_array = 0:indx_integral_span-1
                x = np.sum(radiowave_data_for_ft*np.sin(omg_list[m]*indx_integral_span_array*timestep))  # indx_integral_span_array = 0:indx_integral_span-1
                r = r/float(indx_integral_span)  # Real part
                x = x/float(indx_integral_span)  # Imaginary part
                
                # Store complex spectrum
                spectrum_data[m] = complex(r,x)  # Complex spectrum data of "nfreq" elements 複素スペクトルデータ @ 時間インデックス"ns"
                
                ##Store spectrogram
                #spectrogram[ns,m] = 20.0*np.log10(abs(complex(r,x)))  # Spectrogram in dB^2 (Power spectrum density) スペクトログラム: spectrogram_maxsize行 nf列の行列
            # end for # m, End of DFT

            ########################
            # Check "spectrum_data"
            ########################
            # f = np.arange(nfreq)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(f,abs(spectrum_data))  # 絶対値をプロット OK
            # plt.show()
            # input("Press Enter to Continue...")
            # plt.clf()
            # plt.close()

            #print('Spectrum analysis done スペクトル解析終了')
            #input("Press Enter to Continue...")

            #########################################################################
            # Generate Spectrogram for the whole period 全期間のスペクトログラム生成
            #########################################################################
            #for m in range(nfreq):  # m = 0:nfreq-1
            #    #spectrogram[ns, m] = 20.0*np.log10(abs(spectrum_data[m]))
            #    print(spectrum_data[m])
            # end for m
            #m = np.arange(nfreq)
            #print(m)
            #print(abs(spectrum_data))
            #print(spectrogram[:,ns])
            #print(spectrogram[ns,:])
            #print('len(m) = ',len(m))
            #print('len(spectrum_data) = ',len(spectrum_data))
            #print('len(spectrogram[:,ns]) = ',len(spectrogram[:,ns]))
            #print('len(spectrogram[ns,:]) = ',len(spectrogram[ns,:]))
            
            # Generate Spectrogram; ns行 spectrogram_maxsize列 の行列 スペクトログラム
            spectrogram[ns,:] = 20.0 * np.log10(abs(spectrum_data))  # use "np.log10", not "math.log10"
            #print(spectrogram[ns,:])
            #print('len(spectrogram[ns,:]) = ',len(spectrogram[ns,:]))
            
            # Generate Spectrogram: spectrogram_maxsize行 ns列の行列 スペクトログラム（行 列 が逆の場合）
            #spectrogram[:,ns] = 20.0*np.log10(abs(spectrum_data))  
            #print(spectrogram[:,ns])
            #print('len(spectrogram[:,ns]) = ',len(spectrogram[:,ns]))
            
            # # Check variables
            # print("nfreq, type(nfreq) =", nfreq, type(nfreq) )
            # print("freq_list, type(freq_list) = ", freq_list, type(freq_list) )
            # print("spectrogram_maxsize, type(spectrogram_maxsize) = ", spectrogram_maxsize, type(spectrogram_maxsize) )
            # print("spectrogram, type(spectrogram) = ", spectrogram, type(spectrogram) )
            # input("Press Enter to Continue...")
        # end for ns = 0,1,2,...,spectrogram_maxsize-1

        print('Moving Fourier Transform Completed for Whole Period 全期間の時間移動フーリエ変換終了')
        ###############################################################
        # End of Moving Fourier Transform 
        # 全期間のスペクトルデータ（短時間フーリエ変換データ）生成終了
        ###############################################################

        # デバッグ用テストデータ生成（すべて１のデータ）
        #spectrogram  = np.ones((spectrogram_maxsize, nfreq), dtype=np.float64)  # spectrogram_maxsize行 nf列の行列を１で埋める、２次元配列

        ##############################################################################################################################
        # 既にスペクトログラムデータファイルが存在すれば、念のために名前を変えて保存しておく、保存データ名："SpectrogramDump_old.pcl"
        ##############################################################################################################################
        #if os.path.isfile("SpectrogramDump.pcl"):
        #    os.rename("SpectrogramDump.pcl", "SpectrogramDump_old.pcl") 
        if os.path.isfile(os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump.pcl")):  # Debuged 20210901
            os.rename(os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump.pcl"), os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump_old.pcl"))  # Debuged 20210901
        #end if

        ############################################################################
        # スペクトログラムデータを保存（ダンプ）保存データ名："SpectrogramDump.pcl"
        ############################################################################
        # 変数          数値・長さ      意味                形状                data-type,データ型
        # nfreq            540 etc         周波数ポイント数    スカラー            numpy.int (=numpy.int64)
        # freq_list     [nfreq]            周波数              numpy配列（１次元） numpy.float, numpy.ndarray (=numpy.float64)
        # spectrogram_maxsize     約321000        時間移動点数        スカラー            numpy.int (=numpy.int64)
        # spectrogram   [spectrogram_maxsize, nfreq] スペクトログラム    numpy配列（２次元） numpy.float, numpy.ndarray (=numpy.float64)
        # file dump ファイルに保存
        print('Saving Moving Fourier Transform Data for Whole Period 全期間の時間移動フーリエ変換データ保存中 ....')
        with open(os.path.join(Miscellaneous_DirectoryPath,"SpectrogramDump.pcl"), mode='wb') as fdmp:  # SpectrogramDump.pcl ファイル
            pickle.dump(spectrogram, fdmp)                                                              # にスペクトログラムデータをダンプ（ファイルサイズ約１．３GB）
        #
        # with open の後は自動的に close される
    #
    ####################################################
    # デバッグ用: analyze_actual_spectrogram = 0 の場合
    ####################################################
    else:  # デバッグのため、実際のスペクトログラムではなく、テスト用のダミースペクトログラムを生成する

        # テストデータ生成（すべて１）
        #spectrogram  = np.ones((spectrogram_maxsize, nfreq), dtype=np.float64)  # spectrogram_maxsize行 nf列の行列を１で埋める、２次元配列

        # テストデータ生成（仮分布：列ベクトル　＊　行ベクトル）ただし、チェック必要
        spectrogram  = np.matmul( np.sin(100.0*2.0*np.pi*np.arange(spectrogram_maxsize)/(spectrogram_maxsize-1)).reshape(spectrogram_maxsize, 1), np.cos(0.25*2.0*np.pi*np.arange(nfreq)/(nfreq-1)).reshape(1,nfreq) )  # 0:1 の正弦・余弦分布　チェック必要
        print("spectrogram.shape = ", spectrogram.shape)

        # チェック必要
    #
    # end if isfile("SpectrogramDump.pcl")
    ####################################################

    # Check array
    print("spectrogram, type(spectrogram) = ", spectrogram, type(spectrogram))
    #input("Press Enter to Continue...")

    ###############################################
    ###############################################
    # End of Generating / Reading Spectrogram Data
    # スペクトログラムデータ生成または読み込み完了
    ###############################################
    ###############################################



    ###############################
    ###############################
    ## 異常時データリストの読み込み（常に読み込む）
    ###############################
    ###############################
    # List of the Abnormal state of the RadiowaveData time data
    RadiowaveData_ListAbnrmlFile = "RadiowaveData_List_Abnrml_" + City_DR_No_BCLoc_Freq + ".txt"
    RadiowaveData_ListAbnrmlFile_Path = os.path.join(ListNormalAbnrml_Path, RadiowaveData_ListAbnrmlFile)
    print("RadiowaveData_ListAbnrmlFile_Path = ",RadiowaveData_ListAbnrmlFile_Path)

    # 学習期間の開始日と終了日に相当するunixtime時刻を生成
    TrainStartingDate = TrainStartingDate0  # 学習開始年月日をストリング配列で指定（冒頭の解析条件設定値から取り込み）'yyyymmdd'
    TrainEndingDate   = TrainEndingDate0    # 学習終了年月日をストリング配列で指定（冒頭の解析条件設定値から取り込み）'yyyymmdd'
    TrainStartingTime = int(datetime(int(TrainStartingDate[0:4]), int(TrainStartingDate[4:6]), int(TrainStartingDate[6:8]), 0, 0, 0).timestamp())  # 整数指定の DateTime から Unixtime へ変換
    TrainEndingTime   = int(datetime(int(TrainEndingDate[0:4]),   int(TrainEndingDate[4:6]),   int(TrainEndingDate[6:8]),   0, 0, 0).timestamp())  # 整数指定の DateTime から Unixtime へ変換

    #----------------------------------------------------------------------------------------------------------------------
    # 学習・検証のための画像データを新たに作成する、または作成しない---に関わらず、異常データリストのファイルは常に読み込む（変更202304、あとで削除）
    #if generate_train_images == 1: # 新たな画像データの作成: 作成する=1, 作成しない=0(1以外)
    #----------------------------------------------------------------------------------------------------------------------
    # Open file
    fileID = open(RadiowaveData_ListAbnrmlFile_Path,'r')  # ファイル情報取得
    #
    lines = fileID.readlines()  # 行数取得
    #print("len(lines) = ",len(lines))
    AbnrmlStartingTime = [0]*len(lines)  # 若干多めの配列初期化 ０始まり 正常時データの開始時刻(unixtime, UTC), デフォルトは行ベクトル?
    AbnrmlEndingTime   = [0]*len(lines)  # 若干多めの配列初期化 ０始まり 正常時データの終了時刻(unixtime, UTC), デフォルトは行ベクトル?
    #AbnrmlStartingTime = []  # 空配列定義だと代入時にエラー 正常時データの開始時刻(unixtime, UTC), デフォルトは行ベクトル?　===> 注：空配列定義のあとは .append() を使用して要素を追加していく
    #AbnrmlEndingTime   = []  # 空配列定義だと代入時にエラー 正常時データの終了時刻(unixtime, UTC), デフォルトは行ベクトル?　===> 注：空配列定義のあとは .append() を使用して要素を追加していく
    count = 0
    for tmp_line in lines: # １行ずつ読み込み
        #print(tmp_line, end="")  # １行表示した後、改行は１度だけにする

        if not ( (tmp_line == "") or (tmp_line[0] == "#") or (tmp_line[0] == "\n") ):  # (空行または先頭が#または先頭が「改行」) でない場合
            #count = count + 1  # 異常データ数を正確にカウント　ここではしない10/27
            #print("count = ",count)

            # Date-Timeをunixtimeに変換

            ##########
            # 開始時刻
            ##########
            tmp_time = tmp_line[0:11]  # 年月日_時刻 (JST) (1~11文字目を取り出し) string型
            #print("tmp_time = ",tmp_time, " yyyymmdd_hh (JST)")
            year = int(tmp_line[0:4])  # string型を整数型に変換
            mon  = int(tmp_line[4:6])
            day  = int(tmp_line[6:8])
            hour = int(tmp_line[9:11])
            #print("year,mon,day,hour = ",year,mon,day,hour)
            # 年月日_時刻 datetime (JST) を unixtime (UTC) に変換
            dt = datetime(year, mon, day, 0, 0, 0)  # (JST) （注意: datetime関数は"24"時を扱えないのでhour分は別に計算する） timezone はデフォルトで local = Asia/Tokyo が使用される
            #startingtime = int(time.mktime(dt.timetuple())) + hour*3600  # unixtime (UTC), 9*3600 引く必要なし（古い書式）
            #print("startingtime = ",startingtime, " (unixtime, UTC)")
            startingtime = int(dt.timestamp()) + hour*3600  # unixtime (UTC), timestamp() では 9*3600 引く必要なし（注意: timestamp()はfloat型, Python3のint型は無制限）
            #print("startingtime = ",startingtime, " (unixtime, UTC)")
            ###############################################################################################################################################################
            # もし、開始時刻 startingtime が 学習終了年月日 TrainEndingTime を過ぎていればなにもせずに break する 2021/10/27
            ###############################################################################################################################################################
            if TrainEndingTime < startingtime:
                break
            # end if
            # そうでなければ
            count = count + 1  # 異常データ数をカウント
            #print("count = ",count)
            AbnrmlStartingTime[count-1] = startingtime
            #print("AbnrmlStartingTime[count-1] = ", AbnrmlStartingTime[count-1], " (unixtime, UTC)")

            ##########
            # 終了時刻
            ##########
            tmp_time = tmp_line[12:23]  # 年月日_時刻 (JST) (1~11文字目を取り出し) string型
            #print("tmp_time = ",tmp_time, " yyyymmdd_hh (JST)")
            year = int(tmp_line[12:16])  # string型を整数型に変換
            mon  = int(tmp_line[16:18])
            day  = int(tmp_line[18:20])
            hour = int(tmp_line[21:23])
            #print("year,mon,day,hour = ",year,mon,day,hour)
            # 年月日_時刻 datetime (JST) を unixtime (UTC) に変換
            dt = datetime(year, mon, day, 0, 0, 0)  # (JST) （注意: datetime関数は"24"時を扱えないのでhour分は別に計算する） timezone はデフォルトで local = Asia/Tokyo が使用される
            #endingtime = int(time.mktime(dt.timetuple())) + hour*3600  # unixtime (UTC), 9*3600 引く必要なし（古い書式）
            endingtime = int(dt.timestamp()) + hour*3600  # unixtime (UTC), timestamp() では 9*3600 引く必要なし（注意: timestamp()はfloat型, Python3のint型は無制限）
            #print("endingtime = ",endingtime, " (unixtime, UTC)")
            ###############################################################################################################################################################
            # もし、開始時刻 startingtime と終了時刻 endingtime の間に学習終了年月日 TrainEndingTime がくれば AbnrmlEndingTime[count-1] = TrainEndingTime として break する
            ###############################################################################################################################################################
            if (startingtime < TrainEndingTime) & (TrainEndingTime < endingtime):  # if TrainEndingTime < endingtime: だけで良いはず
                AbnrmlEndingTime[count-1] = TrainEndingTime
                break
            # end if
            # そうでなければ
            AbnrmlEndingTime[count-1] = endingtime
            #print("AbnrmlEndingTime[count-1] = ", AbnrmlEndingTime[count-1], " (unixtime, UTC)")

            #if count == 2:  # チェック用
            #    break
        # end if
    # end for

    #print(AbnrmlStartingTime)  # unixtime (UTC) 行ベクトル をfor文の後にチェック
    #print(AbnrmlEndingTime)    # unixtime (UTC) 行ベクトル をfor文の後にチェック
    fileID.close()
    #AbnrmlStartingTimeSize = count
    #AbnrmlEndingTimeSize   = count
    AbnrmlListSize         = count
    print("AbnrmlListSize = ",AbnrmlListSize)

    ############################################################################################################################################################
    ############################################################################################################################################################
    # Check Order of the Abnormal State List 異常リストの時系列順をチェック
    ############################################################################################################################################################
    ############################################################################################################################################################
    error_frag = 0  # デフォルトで０、時系列順に誤りがあれば１にするフラグ
    for i in range(AbnrmlListSize-2):  # i = 0, AbnrmlListSize-2 番目まで
        print ("i, AbnrmlStartingTime[i], AbnrmlEndingTime[i] = ", i, AbnrmlStartingTime[i], AbnrmlEndingTime[i])
        if AbnrmlStartingTime[i] > AbnrmlEndingTime[i]:
            error_frag = 1  # 時系列順に誤りを発見すれば一旦停止する
            input("!!! Error: Starting Time > Ending Time; Press Enter to Continue")
        # end if    
        if AbnrmlEndingTime[i] > AbnrmlStartingTime[i+1]:
            error_frag = 1  # 時系列順に誤りを発見すれば一旦停止する
            input("!!! Error: Ending Time > Next Starting Time; Press Enter to Continue")
        # end if
    # end for
    i = AbnrmlListSize-1  # リストの最終エントリー: i = AbnrmlListSize-1 番目
    print ("i, AbnrmlStartingTime[i], AbnrmlEndingTime[i] = ", i, AbnrmlStartingTime[i], AbnrmlEndingTime[i])
    if AbnrmlStartingTime[i] > AbnrmlEndingTime[i]:
        error_frag = 1  # 時系列順に誤りを発見すれば一旦停止する:q
        input("!!! Error: Starting Time > Ending Time; Press Enter to Continue")
    # end if
    if error_frag == 1:
        input("Abnormal State List Has an Error in the Chronological Order. Stop Here to Check and Correct Manually.")  # 時系列順に誤りを発見すれば一旦停止する
    # end if
    ############################################################################################################################################################
    #input("Abnormal State List is in Correct Chronological Order, Press Enter to Continue")
    ############################################################################################################################################################
    #--------------------------------------------
    # end if 学習用画像データを作成する場合の終了（あとで削除）
    #--------------------------------------------



    ###########################################
    ###########################################
    # 地震データベース (EQ_Database) の読み込み（常に読み込む）
    ###########################################
    ###########################################

    # 地震データベースファイル選択
    #EQ_Database_Toyama_Path   = "/home/keisoku/Dropbox/RadiowaveData_Plots_v5.5f_TY12M0SI1_SZ6M0SI1/EQdatabase_Toyama/EQdatabase_Toyama_e.csv"
    #EQ_Database_Shizuoka_Path = "/home/keisoku/Dropbox/RadiowaveData_Plots_v5.5f_TY12M0SI1_SZ6M0SI1/EQdatabase_Shizuoka/EQdatabase_Shizuoka_e.csv"
    EQ_Database_Toyama_Path   = os.path.join(Databox_Path, "RadiowaveData_Plots_v5.5f_TY12M0SI1_SZ6M0SI1/EQdatabase_Toyama/EQdatabase_Toyama_e.csv")
    EQ_Database_Shizuoka_Path = os.path.join(Databox_Path, "RadiowaveData_Plots_v5.5f_TY12M0SI1_SZ6M0SI1/EQdatabase_Shizuoka/EQdatabase_Shizuoka_e.csv")
    print("EQ_Database_Toyama_Path   = ", EQ_Database_Toyama_Path)
    print("EQ_Database_Shizuoka_Path = ", EQ_Database_Shizuoka_Path)
    # 参考
    # EQdatabase_Toyama_e.csv フォーマット例 
    # 1,2015/01/01 06:12:38.0,福島県沖,Fukushima Offshore(1; 3.4),37°06.2′N,141°09.9′E,28km,M,3.4,1,1,2015/01/01 06:12
    # 4,2015/01/01 22:57:14.4,苫小牧沖,Tomakomai Offshore(1; 4.5),42°35.0′N,141°50.2′E,30km,M,4.5,4,1,2015/01/02 02:01
    # ...
    # EQdatabase_Shizuoka_e.csv フォーマット例
    # 1301,2023/04/25 21:59:58.1,和歌山県北部,Wakayama North(2; 2.5),34°01.6′N,135°15.2′E,5km,M,2.5,2,2,2023/04/25 21:59
    # 1302,2023/04/30 18:52:32.2,岐阜県美濃東部,Gifu Mino East(1; 3.7),35°38.5′N,137°11.3′E,10km,M,3.7,3,1,2023/04/30 18:52
    #

    #----------------------------------
    # 富山より北東12県検索地震データ OK
    #----------------------------------
    #EQ_Database_Toyama_ListSize         # Number of EQs 地震数
    EQ_Database_Toyama_Dates        = [] # Strings, 文字データ：日時
    EQ_Database_Toyama_Epicenters_j = [] # Strings, 文字データ：震央（日本語）
    EQ_Database_Toyama_Epicenters_e = [] # Strings, 文字データ：震央（英語）+(SI, Mag)
    EQ_Database_Toyama_Uxtimes      = [] # Numerics, 数値データ int: Unixtime
    EQ_Database_Toyama_Mags         = [] # Numerics, 数値データ float: Mag
    EQ_Database_Toyama_MaxSIs       = [] # Numerics, 数値データ int: Max_SI, Seismic Intensity, 最大震度
    EQ_Database_Toyama_LocalSIs     = [] # Numerics, 数値データ int; Local_SI, 検索地域の震度
    count = 0
    with open(EQ_Database_Toyama_Path) as f:
        reader = csv.reader(f)
        # 行ごとの処理
        for row in reader:
            #print()
            #print('count = ',count)
            #例 1,2015/01/01 06:12:38.0,福島県沖,Fukushima Offshore(1; 3.4),37°06.2′N,141°09.9′E,28km,M,3.4,1,1,2015/01/01 06:12
            #print('row = ',row)
            #print('row[0] = ',row[0]) # １列目：番号（０始まりに注意）
            eq_strdatetime = row[1]    # ２列目：地震発生の日時（文字列, str）
            #print('type(eq_strdatetime)          = ',type(eq_strdatetime))
            #print('地震発生の日時 eq_strdatetime = ',eq_strdatetime)
            eq_datetime = datetime.strptime(eq_strdatetime, '%Y/%m/%d %H:%M:%S.%f')  # JST を文字列strで読み込み、datetimeオブジェクトに変換
            #eq_uxtime  = int(eq_datetime.strftime('%s'))  # datetimeオブジェクトをUnixtime (UTC, str) 「秒」に変換し、さらに整数に変換
            eq_uxtime   = int(eq_datetime.timestamp())     # datetimeオブジェクトをUnixtime (UTC, float) 「秒」に変換し、さらに整数に変換
            #print('type(eq_uxtime) = ',type(eq_uxtime))
            #print('地震発生の日時 eq_uxtime = ',eq_uxtime)
            eq_epicenter_j = row[2]   # ３列目：震央（日本語）
            eq_epicenter_e = row[3]   # ４列目：震央（英語）+(SI, Mag)
            if row[8] == '不明':      # ９列目：マグニチュード
                eq_mag      = 0.0
                eq_max_si   = 0
                eq_local_si = 0
            else:
                eq_mag      = float(row[8])      #   ９列目：マグニチュード 
                eq_max_si   = int(row[9][0:1])   # １０列目：１文字目のみ取り出し、強弱(Lo,Hi)は無視：最大震度 Max SI
                eq_local_si = int(row[10][0:1])  # １１列目：１文字目のみ取り出し、強弱(Lo,Hi)は無視：検索地域の震度 Searched Local SI
            #end if
            # 地震データを辞書ではなくリストにする（ただし、０始まり）
            EQ_Database_Toyama_ListSize = count + 1         # Number of EQs 地震数保存
            EQ_Database_Toyama_Dates.append(eq_strdatetime)
            EQ_Database_Toyama_Epicenters_j.append(eq_epicenter_j)
            EQ_Database_Toyama_Epicenters_e.append(eq_epicenter_e)
            EQ_Database_Toyama_Uxtimes.append(eq_uxtime)
            EQ_Database_Toyama_Mags.append(eq_mag)
            EQ_Database_Toyama_MaxSIs.append(eq_max_si)
            EQ_Database_Toyama_LocalSIs.append(eq_local_si)
            # print('EQ_Database_Toyama_ListSize            = ',EQ_Database_Toyama_ListSize)
            # print('EQ_Database_Toyama_Dates[count]        = ',EQ_Database_Toyama_Dates[count])
            # print('EQ_Database_Toyama_Epicenters_j[count] = ',EQ_Database_Toyama_Epicenters_j[count])
            # print('EQ_Database_Toyama_Epicenters_e[count] = ',EQ_Database_Toyama_Epicenters_e[count])
            # print('EQ_Database_Toyama_Uxtimes[count]      = ',EQ_Database_Toyama_Uxtimes[count])
            # print('EQ_Database_Toyama_Mags[count]         = ',EQ_Database_Toyama_Mags[count])
            # print('EQ_Database_Toyama_MaxSIs[count]       = ',EQ_Database_Toyama_MaxSIs[count])
            # print('EQ_Database_Toyama_LocalSIs[count]     = ',EQ_Database_Toyama_LocalSIs[count])
            # print()
            # input("Reading EQ Database for Toyama...  Press Enter to Continue...")  #############################################################チェックOK
            count = count + 1
        #end for
    #end with


    #---------------------------------
    # 静岡磐田近辺6県検索地震データ OK
    #---------------------------------
    #EQ_Database_Shizuoka_ListSize         # Number of EQs 地震数
    EQ_Database_Shizuoka_Dates        = [] # Strings, 文字データ：日時
    EQ_Database_Shizuoka_Epicenters_j = [] # Strings, 文字データ：震央（日本語）
    EQ_Database_Shizuoka_Epicenters_e = [] # Strings, 文字データ：震央（英語）+(SI, Mag)
    EQ_Database_Shizuoka_Uxtimes      = [] # Numerics, 数値データ int: Unixtime
    EQ_Database_Shizuoka_Mags         = [] # Numerics, 数値データ float: Mag
    EQ_Database_Shizuoka_MaxSIs       = [] # Numerics, 数値データ int: Max_SI, Seismic Intensity, 最大震度
    EQ_Database_Shizuoka_LocalSIs     = [] # Numerics, 数値データ int; Local_SI, 検索地域の震度
    count = 0
    with open(EQ_Database_Shizuoka_Path) as f:
        reader = csv.reader(f)
        # 行ごとの処理
        for row in reader:
            #print()
            #print('count = ',count)
            #例 1,2015/01/01 06:12:38.0,福島県沖,Fukushima Offshore(1; 3.4),37°06.2′N,141°09.9′E,28km,M,3.4,1,1,2015/01/01 06:12
            #print('row = ',row)
            #print('row[0] = ',row[0]) # １列目：番号（０始まりに注意）
            eq_strdatetime = row[1]   # ２列目：地震発生の日時（文字列）
            #print('type(eq_strdatetime)          = ',type(eq_strdatetime))
            #print('地震発生の日時 eq_strdatetime = ',eq_strdatetime)
            eq_datetime = datetime.strptime(eq_strdatetime, '%Y/%m/%d %H:%M:%S.%f')  # JST を文字列strで読み込み、datetimeオブジェクトに変換
            #eq_uxtime  = int(eq_datetime.strftime('%s'))  # datetimeオブジェクトをUnixtime (UTC, str) 「秒」に変換し、さらに整数に変換
            eq_uxtime   = int(eq_datetime.timestamp())     # datetimeオブジェクトをUnixtime (UTC, float) 「秒」に変換し、さらに整数に変換
            #print('type(eq_uxtime) = ',type(eq_uxtime))
            #print('地震発生の日時 eq_uxtime = ',eq_uxtime)
            eq_epicenter_j = row[2]   # ３列目：震央（日本語）
            eq_epicenter_e = row[3]   # ４列目：震央（英語）+(SI, Mag)
            if row[8] == '不明':      # ９列目：マグニチュード
                eq_mag      = 0.0
                eq_max_si   = 0
                eq_local_si = 0
            else:
                eq_mag      = float(row[8])      #   ９列目：マグニチュード 
                eq_max_si   = int(row[9][0:1])   # １０列目：１文字目のみ取り出し、強弱(Lo,Hi)は無視：最大震度 Max SI
                eq_local_si = int(row[10][0:1])  # １１列目：１文字目のみ取り出し、強弱(Lo,Hi)は無視：検索地域の震度 Searched Local SI
            #end if
            # 地震データを辞書ではなくリストにする（ただし、０始まり）
            EQ_Database_Shizuoka_ListSize = count + 1         # Number of EQs 地震数保存
            EQ_Database_Shizuoka_Dates.append(eq_strdatetime)
            EQ_Database_Shizuoka_Epicenters_j.append(eq_epicenter_j)
            EQ_Database_Shizuoka_Epicenters_e.append(eq_epicenter_e)
            EQ_Database_Shizuoka_Uxtimes.append(eq_uxtime)
            EQ_Database_Shizuoka_Mags.append(eq_mag)
            EQ_Database_Shizuoka_MaxSIs.append(eq_max_si)
            EQ_Database_Shizuoka_LocalSIs.append(eq_local_si)
            # print('EQ_Database_Shizuoka_ListSize            = ',EQ_Database_Shizuoka_ListSize)
            # print('EQ_Database_Shizuoka_Dates[count]        = ',EQ_Database_Shizuoka_Dates[count])
            # print('EQ_Database_Shizuoka_Epicenters_j[count] = ',EQ_Database_Shizuoka_Epicenters_j[count])
            # print('EQ_Database_Shizuoka_Epicenters_e[count] = ',EQ_Database_Shizuoka_Epicenters_e[count])
            # print('EQ_Database_Shizuoka_Uxtimes[count]      = ',EQ_Database_Shizuoka_Uxtimes[count])
            # print('EQ_Database_Shizuoka_Mags[count]         = ',EQ_Database_Shizuoka_Mags[count])
            # print('EQ_Database_Shizuoka_MaxSIs[count]       = ',EQ_Database_Shizuoka_MaxSIs[count])
            # print('EQ_Database_Shizuoka_LocalSIs[count]     = ',EQ_Database_Shizuoka_LocalSIs[count])
            # print()
            # input("Reading EQ Database for Shizuoka...  Press Enter to Continue...")  #############################################################チェックOK
            count = count + 1
        #end for
    #end with


    #-----------------------------------------------------------------
    # 参考：複数のリストをまとめる "zip" : タプルのリストに変換される。（未使用：これを使用して、EQ_Database_... 関係のデータをひとまとめにし、関数作成する予定）
    #-----------------------------------------------------------------
    # リスト3 = list(zip(リスト1,リスト2)
    # Pythonの複数の「リスト型（list）」をまとめて統合するするには「zip」を使います。
    # 「リスト1」と「リスト2」を統合して「リスト3」を作ります。「リスト3」は２次元配列のような形になります。
    # リストの統合は２つだけでなく３つでも４つでも可能です。
    #
    # もしzip()の引数の要素数が異なる場合は、要素数が小さい方に合わせられ残りの要素を捨てられます。
    # タプルは一旦生成した後は変更できない（イミュータブル）。
    #
    # 例1. zipでリスト型（list）をまとめる
    # リスト型（list）
    # adr_no = ['150-0013','150-0021','150-0022']
    # address = ['東京都渋谷区恵比寿','東京都渋谷区恵比寿西','東京都渋谷区恵比寿南']
    # リストを統合
    # adr_list = list(zip(adr_no,address))
    #
    # print(adr_list)
    # [結果] [('150-0013', '東京都渋谷区恵比寿'), ('150-0021', '東京都渋谷区恵比寿西'), ('150-0022', '東京都渋谷区恵比寿南')] # ()内がタプル型
    # adr_list[0][0]
    # [結果] '150-0013'
    # adr_list[0][1]
    # [結果] '東京都渋谷区恵比寿'
    #-----------------------------------------------------------------



    #####################################################################################################################################################
    # 異常時データリストの各要素(i=0〜AbnrmlListSize-1)について、その期間＋３日間以内の最大震度または検索震度または最大マグニチュードを探し出して保存する
    #
    # ＝＞　これをそれぞれの異常信号のクラス分けに利用する
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # 最大震度           : AbnrmlState_MaxSI_Max[i]   整数 int
    # 検索震度           : AbnrmlState_LocalSI_Max[i] 整数 int
    # 最大マグニチュード : AbnrmlState_Mag_Max[i]     実数 float
    #####################################################################################################################################################
    # 全ての異常区間について初期化 Initialization
    AbnrmlState_MaxSI_Max   = [0]*AbnrmlListSize 
    AbnrmlState_LocalSI_Max = [0]*AbnrmlListSize
    AbnrmlState_Mag_Max     = [0]*AbnrmlListSize

    # 観測データによってどの地震データを使用するか選択する
    if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
         City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"  or
         City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
        #-----------------------------------------------
        # 富山より北東１２県の地震データから最大値を探す
        #-----------------------------------------------
        for i in range(AbnrmlListSize):  # i = 0, AbnrmlListSize-1 番目までのすべての異常信号について
            # 最大値を初期化
            maxSI_max   = 0
            localSI_max = 0
            mag_max     = 0.0
            for j in range(EQ_Database_Toyama_ListSize):  # 上の異常信号の区間＋３日間に、地震データベースファイルのj = 0, EQ_Database_Toyama_ListSize-1 番目までの全ての地震から探す
                # j 番目の地震が異常区間内であるときその " j " について
                if all( [ AbnrmlStartingTime[i] < EQ_Database_Toyama_Uxtimes[j], EQ_Database_Toyama_Uxtimes[j] <  AbnrmlEndingTime[i]+int(3*24*60*60) ] ):
                    # 最大震度を探す
                    if (EQ_Database_Toyama_MaxSIs[j] > maxSI_max):
                        maxSI_max = EQ_Database_Toyama_MaxSIs[j]
                    #end if
                    # 最大検索震度を探す
                    if (EQ_Database_Toyama_LocalSIs[j] > localSI_max):
                        localSI_max = EQ_Database_Toyama_LocalSIs[j]
                    #end if
                    # 最大マグニチュードを探す
                    if ( EQ_Database_Toyama_Mags[j] > mag_max):
                        mag_max = EQ_Database_Toyama_Mags[j]
                    #end if 
                #end if
            #end for 全ての j について探し終わったら、
            # 最大値を保存して、
            AbnrmlState_MaxSI_Max[i]   = maxSI_max
            AbnrmlState_LocalSI_Max[i] = localSI_max
            AbnrmlState_Mag_Max[i]     = mag_max
            # 次の異常信号へ行く
        #end for

    # 観測データによってどの地震データを使用するか選択する
    elif(City_DR_No_BCLoc_Freq == "Iwata_NE_11_Yokohama_84p7MHz" or City_DR_No_BCLoc_Freq == "Iwata_NE_16_Shizuoka_88p8MHz" or
         City_DR_No_BCLoc_Freq == "Iwata_NE_4_Shizuoka_79.2MHz"  or City_DR_No_BCLoc_Freq == "Iwata_NW_2_Tsu_78p9MHz"       or
         City_DR_No_BCLoc_Freq == "Iwata_NW_8_Nagoya_82p5MHz"):
        #---------------------------------------------
        # 静岡磐田近辺６県の地震データから最大値を探す
        #---------------------------------------------
        for i in range(AbnrmlListSize):  # i = 0, AbnrmlListSize-1 番目までのすべての異常信号について
            # 最大値を初期化
            maxSI_max   = 0
            localSI_max = 0
            mag_max     = 0.0
            for j in range(EQ_Database_Shizuoka_ListSize):  # 上の異常信号の区間＋３日間に、地震データベースファイルのj = 0, EQ_Database_Shizuoka_ListSize-1 番目までの全ての地震から探す
                # j 番目の地震が異常区間内であるときその " j " について
                if all( [ AbnrmlStartingTime[i] < EQ_Database_Shizuoka_Uxtimes[j], EQ_Database_Shizuoka_Uxtimes[j] <  AbnrmlEndingTime[i]+int(3*24*60*60) ] ):
                    # 最大震度を探す
                    if (EQ_Database_Shizuoka_MaxSIs[j] > maxSI_max):
                        maxSI_max = EQ_Database_Shizuoka_MaxSIs[j]
                    #end if
                    # 最大検索震度を探す
                    if (EQ_Database_Shizuoka_LocalSIs[j] > localSI_max):
                        localSI_max = EQ_Database_Shizuoka_LocalSIs[j]
                    #end if
                    # 最大マグニチュードを探す
                    if ( EQ_Database_Shizuoka_Mags[j] > mag_max):
                        mag_max = EQ_Database_Shizuoka_Mags[j]
                    #end if 
                #end if
            #end for 全ての j について探し終わったら、
            # 最大値を保存して、
            AbnrmlState_MaxSI_Max[i]   = maxSI_max
            AbnrmlState_LocalSI_Max[i] = localSI_max
            AbnrmlState_Mag_Max[i]     = mag_max
            # 次の異常信号へ行く
        #end for
    #
    #end if
    #---------------------------------------------------------
    # それぞれの異常時データに対応する地震データの最大値リストの確認
    #---------------------------------------------------------
    print("それぞれの異常時データに対応する最大震度リスト           AbnrmlState_MaxSI_Max   = ",AbnrmlState_MaxSI_Max)
    print("それぞれの異常時データに対応する検索震度リスト           AbnrmlState_LocalSI_Max = ",AbnrmlState_LocalSI_Max)
    print("それぞれの異常時データに対応する最大マグニチュードリスト AbnrmlState_Mag_Max     = ",AbnrmlState_Mag_Max)


    ######################################################################
    ######################################################################
    ## 正常時データリストの読み込みまたは異常時データリストの補集合を計算
    ######################################################################
    ######################################################################

    # 学習・検証のための画像データを新たに作成する、または作成しない
    if generate_train_images == 1:  # 新たな画像データの作成: 作成する=1, 作成しない=0(1以外)

        if set_normal_state_as_non_abnrml == 1:  # 正常状態を正常状態のリストから読まず、異常状態以外のすべての期間を正常状態とする場合
            #
            # 異常状態のリストの補集合を正常状態のリストとする
            #
            print("Set Normal State as Non-Abnormal State 正常状態を正常状態のリストから読まず、異常状態以外のすべての期間を正常状態とする")

            # 最大の正常区間数 = 異常区間数 + 1 とする
            NormalListSize = AbnrmlListSize + 1  # 配列初期化のために一時的に設定
            # 配列初期化
            NormalStartingTime = [0]*NormalListSize  # 配列初期化 ０始まり 正常時データの開始時刻(unixtime, UTC) 整数として取り扱う
            NormalEndingTime   = [0]*NormalListSize  # 配列初期化 ０始まり 正常時データの終了時刻(unixtime, UTC) 整数として取り扱う

            ## 0番目の区間の開始時刻は0秒としておく（あとで修正必要）
            #NormalStartingTime[0] = 0  # 整数
            
            # 0番目の区間の開始時刻はスペクトログラムの開始時刻とする（あとで修正）
            #Spectrogram_TimeStart        = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])  # Starting Time (unixtime) of "spectrogram" スペクトログラムの最初の時刻
            #print("Spectrogram_TimeStart = ", Spectrogram_TimeStart)
            #NormalStartingTime[0]        = int(Spectrogram_TimeStart)
            #print("NormalStartingTime[0] = ", NormalStartingTime[0])
            
            # 最初（０番目）の区間の開始時刻を冒頭で設定した学習開始年月日とする（学習開始時刻が正常期間にあることを想定しているが、そうでないときは修正必要）
            #TrainStartingDate            = TrainStartingDate0  # 学習開始年月日をストリング配列で指定（冒頭の解析条件設定値から取り込み）'yyyymmdd'
            #print("TrainStartingDate = ", TrainStartingDate)
            NormalStartingTime[0]        = TrainStartingTime  # =int(datetime(int(TrainStartingDate[0:4]), int(TrainStartingDate[4:6]), int(TrainStartingDate[6:8]), 0, 0, 0).timestamp())  # 整数指定の DateTime から Unixtime へ変換
            #print("NormalStartingTime[0] = ", NormalStartingTime[0])
            #TrainEndingDate              = TrainEndingDate0  # 学習終了年月日をストリング配列で指定（冒頭の解析条件設定値から取り込み）'yyyymmdd'

            count = 0
            # 中間の区間の時刻を異常区間と次の異常区間の間にとる
            for i in range(NormalListSize-1):  # i = 0, NormalListSize-2（i+1番目にアクセスするため一つ下のインデックスまでにしておく = インデックスを１つ少なく回す）
            #for i in range(NormalListSize):  # i = 0, NormalListSize-1

                count = count + 1 # 正常データ数を正確にカウント
                #print("count = ",count)
                # i番目の正常区間の終了時刻はi番目の異常区間の開始時刻より1秒前
                #NormalEndingTime[i]     = AbnrmlStartingTime[i] - 1  # 整数, 1秒下げておく
                # i番目の正常区間の終了時刻はi番目の異常区間の開始時刻と同じにする
                #NormalEndingTime[i]      = AbnrmlStartingTime[i]      # unixtime 整数, 1秒下げておくのをやめる（異常区間と正常区間の境界が重なっても良い）(Note: i = count-1)
                # (Note: i = count-1)
                ##########################################################################################################################################
                # もし AbnrmlStartingTime[i] が TrainEndingTime を過ぎていれば NormalEndingTime[i] = TrainEndingTime として break する (デバッグ 20211026)
                ##########################################################################################################################################
                #if TrainEndingTime <= NormalEndingTime[i]:  # TrainEndingTime = int(datetime(int(TrainEndingDate[0:4]), int(TrainEndingDate[4:6]), int(TrainEndingDate[6:8]), 0, 0, 0).timestamp()):  # TrainEndingDate から Unixtime へ変換した時間
                if TrainEndingTime <= AbnrmlStartingTime[i]:
                    NormalEndingTime[i] = TrainEndingTime   # 正常リストの最後のエントリー
                    break
                # end if
                # そうでなければ
                # i番目の正常区間の終了時刻はi番目の異常区間の開始時刻と同じにする
                NormalEndingTime[i] = AbnrmlStartingTime[i]      # unixtime 整数, 1秒下げておくのをやめる（異常区間と正常区間の境界が重なっても良い）(Note: i = count-1)

                ###################################################################################
                # もし AbnrmlEndingTime[i] が TrainEndingTime を過ぎていればなにもせずに break する
                ###################################################################################
                if TrainEndingTime <= AbnrmlEndingTime[i]:
                    break
                # end if
                # そうでなければ
                # i+1番目の正常区間の開始時刻はi番目の異常区間の終了時刻の1秒後
                #NormalStartingTime[i+1] = AbnrmlEndingTime[i]   + 1  # 整数, 1秒上げておく
                # i+1番目の正常区間の開始時刻はi番目の異常区間の終了時刻と同じにする
                NormalStartingTime[i+1]  = AbnrmlEndingTime[i]        # unixtime 整数, 1秒上げておくのをやめる（異常区間と正常区間の境界が重なっても良い）
            #
            #end for
            
            NormalListSize = count  # 実際に数えた正常データ数の値を設定する
            print("NormalListSize = ",NormalListSize)

            ## 最終 (NormalListSize-1) 番目の終了時刻は現在時刻 (unixtime) としておく（あとで修正必要）
            #NormalEndingTime[NormalListSize-1] = int(time.time())  # 現在のunixtime (s)
            ##
            ## 最終 (NormalListSize-1) 番目の終了時刻はスペクトログラムの最後の時刻とする
            ##
            #spectrogram_maxsize                      = len(spectrogram[:,0])
            #Spectrogram_TimeEnd                      = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[spectrogram_maxsize-1]])  # Ending Time (unixtime) of "spectrogram" スペクトログラムの最後の時刻
            #print("Spectrogram_TimeEnd               = ", Spectrogram_TimeEnd)
            #NormalEndingTime[NormalListSize-1]       = int(Spectrogram_TimeEnd)
            ##
            ## 処理の終了時刻を設定
            ##
            # 最終 (NormalListSize-1) 番目の終了時刻は２０２０年４月１日０時０分０秒とする（その後の期間を判別分析する）
            #NormalEndingTime[NormalListSize-1]       = int(datetime(2020, 4, 1, 0, 0, 0).timestamp())  # 最終 (NormalListSize-1) 番目の終了時刻は 2020年04月01日00時00分00秒
            #
            # 正常リストの最後のエントリーを学習期間終了時刻とする（学習期間の最後の区間が正常区間になる場合）
            #
            # もう少し良い判定方法があれば採用するがとりあえずこれで行く
            #
            if (NormalStartingTime[count] < TrainEndingTime) & (not AbnrmlEndingTime[AbnrmlListSize-1] == TrainEndingTime):  # =int(datetime(int(TrainEndingDate[0:4]), int(TrainEndingDate[4:6]), int(TrainEndingDate[6:8]), 0, 0, 0).timestamp()):  # TrainEndingDate から Unixtime へ変換した時間
                NormalEndingDate        = TrainEndingDate
                NormalListSize = count + 1
                NormalEndingTime[NormalListSize-1] = TrainEndingTime  # =int(datetime(int(NormalEndingDate[0:4]), int(NormalEndingDate[4:6]), int(NormalEndingDate[6:8]), 0, 0, 0).timestamp())  # 整数指定の DateTime から Unixtime へ変換
                print("NormalListSize = ", NormalListSize)   
                print("NormalEndingTime[NormalListSize-1] = ", NormalEndingTime[NormalListSize-1])
            # end if
            #input("Press Enter to Continue")

            ####################################################################################################################################################################################################
            # Check Normal and Abnormal State List 正常、異常リストをチェック
            #for i in range(NormalListSize-1):  # i = 0, NormalListSize-2 番目まで表示（最後がNormal正常区間の場合）
            print("Check Normal and Abnormal State List; 正常、異常リストをチェック")
            print("i, NormalStartingTime[i], NormalEndingTime[i], AbnrmlStartingTime[i], AbnrmlEndingTime[i]")
            for i in range(NormalListSize):    # i = 0, NormalListSize-1 番目まで（最後がAbnrml異常区間の場合）
                print (i, NormalStartingTime[i], NormalEndingTime[i], AbnrmlStartingTime[i], AbnrmlEndingTime[i])
            #end for
            print("NormalListSize = ", NormalListSize)   
            print("AbnrmlListSize = ", AbnrmlListSize)   
            #print (NormalListSize-1, NormalStartingTime[NormalListSize-1], NormalEndingTime[NormalListSize-1], )  # 最後の要素 NormalListSize-1 番目
            #print (NormalListSize, NormalStartingTime[NormalListSize], NormalEndingTime[NormalListSize])  # 最後の要素 NormalListSize 番目（正常区間がなければ初期値 0 のままのはずで、この場合ここは使用しない）
            #####################################################################################################################################################################################################
            #input("Press Enter to Continue")
        #
        #
        else:  # if not set_normal_state_as_non_abnrml == 1:  # 正常状態を正常状態のリストから読み込む

            # List of the Normal state of the RadiowaveData time data 正常状態のリストを指定
            RadiowaveData_ListNormalFile = "RadiowaveData_List_Normal_" + City_DR_No_BCLoc_Freq + ".txt"
            RadiowaveData_ListNormalFile_Path = os.path.join(ListNormalAbnrml_Path, RadiowaveData_ListNormalFile)
            print("RadiowaveData_ListNormalFile_Path = ",RadiowaveData_ListNormalFile_Path)

            # Open file
            fileID = open(RadiowaveData_ListNormalFile_Path,'r')  # ファイル情報取得
            #
            lines = fileID.readlines()  # 行数取得
            #print("len(lines) = ",len(lines))
            NormalStartingTime = [0]*len(lines)  # 若干多めの配列初期化 ０始まり 正常時データの開始時刻(unixtime, UTC), デフォルトは行ベクトル?
            NormalEndingTime   = [0]*len(lines)  # 若干多めの配列初期化 ０始まり 正常時データの終了時刻(unixtime, UTC), デフォルトは行ベクトル?
            #NormalStartingTime = []  # 空配列定義だと代入時にエラー 正常時データの開始時刻(unixtime, UTC), デフォルトは行ベクトル?
            #NormalEndingTime   = []  # 空配列定義だと代入時にエラー 正常時データの終了時刻(unixtime, UTC), デフォルトは行ベクトル?
            count = 0
            for tmp_line in lines: # １行ずつ読み込み
                #print(tmp_line, end="")  # １行表示した後、改行は１度だけにする

                #if not( (tmp_line == "") or (tmp_line[0] == "#") ):  # (空行または先頭が#の行) でない場合
                if not ( (tmp_line == "") or (tmp_line[0] == "#") or (tmp_line[0] == "\n") ):  # (空行または先頭が#または先頭が「改行」) でない場合                
                    #count = count + 1  # データ数を正確にカウント ここではしない10/27
                    #print("count = ",count)

                    # Date-Timeをunixtimeに変換
                    # 開始時刻
                    tmp_time = tmp_line[0:11]  # 年月日_時刻 (JST) (1~11文字目を取り出し) string型
                    #print("tmp_time = ",tmp_time, " yyyymmdd_hh (JST)")
                    year = int(tmp_line[0:4])  # string型を整数型に変換
                    mon  = int(tmp_line[4:6])
                    day  = int(tmp_line[6:8])
                    hour = int(tmp_line[9:11])
                    #print("year,mon,day,hour = ",year,mon,day,hour)
                    # 年月日_時刻 datetime (JST) を unixtime (UTC) に変換
                    dt = datetime(year, mon, day, 0, 0, 0)  # (JST) （注意: datetime関数は"24"時を扱えないのでhour分は別に計算する） timezone はデフォルトで local = Asia/Tokyo が使用される
                    #startingtime = int(time.mktime(dt.timetuple())) + hour*3600  # unixtime (UTC), 9*3600 引く必要なし（古い書式）
                    #print("startingtime = ",startingtime, " (unixtime, UTC)")
                    startingtime = int(dt.timestamp()) + hour*3600  # unixtime (UTC), timestamp() では 9*3600 引く必要なし（注意: timestamp()はfloat型, Python3のint型は無制限）
                    #print("startingtime = ",startingtime, " (unixtime, UTC)")
                    ###############################################################################################################################################################
                    # もし、開始時刻 startingtime が 学習終了年月日 TrainEndingTime を過ぎていればなにもせずに break する 2021/10/27
                    ###############################################################################################################################################################
                    if TrainEndingTime < startingtime:
                        break
                    # end if
                    # そうでなければ
                    count = count + 1  # 正常データ数をカウント
                    #print("count = ",count)
                    NormalStartingTime[count-1] = startingtime
                    #print("NormalStartingTime[count-1] = ", NormalStartingTime[count-1], " (unixtime, UTC)")

                    # 終了時刻
                    tmp_time = tmp_line[12:23]  # 年月日_時刻 (JST) (1~11文字目を取り出し) string型
                    #print("tmp_time = ",tmp_time, " yyyymmdd_hh (JST)")
                    year = int(tmp_line[12:16])  # string型を整数型に変換
                    mon  = int(tmp_line[16:18])
                    day  = int(tmp_line[18:20])
                    hour = int(tmp_line[21:23])
                    #print("year,mon,day,hour = ",year,mon,day,hour)
                    # 年月日_時刻 datetime (JST) を unixtime (UTC) に変換
                    dt = datetime(year, mon, day, 0, 0, 0)  # (JST) （注意: datetime関数は"24"時を扱えないのでhour分は別に計算する） timezone はデフォルトで local = Asia/Tokyo が使用される
                    #endingtime = int(time.mktime(dt.timetuple())) + hour*3600  # unixtime (UTC), 9*3600 引く必要なし（古い書式）
                    endingtime = int(dt.timestamp()) + hour*3600  # unixtime (UTC), timestamp() では 9*3600 引く必要なし（注意: timestamp()はfloat型, Python3のint型は無制限）
                    #print("endingtime = ",endingtime, " (unixtime, UTC)")
                    ###############################################################################################################################################################
                    # もし、開始時刻 startingtime と終了時刻 endingtime の間に学習終了年月日 TrainEndingTime がくれば NormalEndingTime[count-1] = TrainEndingTime として break する
                    ###############################################################################################################################################################
                    if (startingtime < TrainEndingTime) & (TrainEndingTime < endingtime):  # if TrainEndingTime < endingtime: だけで良いはず
                        NormalEndingTime[count-1] = TrainEndingTime
                        break
                    # end if
                    NormalEndingTime[count-1] = endingtime
                    #print("NormalEndingTime[count-1] = ", NormalEndingTime[count-1], " (unixtime, UTC)")

                    #if count == 2:  # チェック用
                    #    break
                    #end if    
                #
                # end if
            #
            # end for
            #print(NormalStartingTime)  # unixtime (UTC) 行ベクトル をfor文の後にチェック
            #print(NormalEndingTime)    # unixtime (UTC) 行ベクトル をfor文の後にチェック
            fileID.close()
            #NormalStartingTimeSize = count
            #NormalEndingTimeSize   = count
            NormalListSize         = count
            #print("NormalListSize = ",NormalListSize)
        #
        # end if 正常状態を正常状態のリストから読む、または、読まずに異常状態以外のすべての期間を正常状態とする場合分けの終了
        print("NormalListSize = ",NormalListSize)
    #
    # end if 学習用画像データを作成する場合の終了



    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # 正常区間、異常区間のそれぞれのスペクトログラム/スカログラム画像生成、保存:
    #
    # 正常時・異常時データ生成とスペクトログラム/スカログラム生成保存
    #
    # 0. 既存の画像ファイルをすべて消去または消去しないを選択
    #
    # 1. 正常時および異常時データリストに従って配列'RadiowaveDataExtracted'から正常時および異常時データを生成、
    # その後、フーリエ変換/ウェーブレット変換によりスペクトログラム/スカログラムを生成、JPG画像保存
    #
    # 2. 正常時および異常時データを少しずつずらして開始時の時刻で名前をつけて保存 
    # 保存場所：
    # Matlab
    # (Spectrogram_DL_Path +
    # "/Ref_Timedata_Toyama_NE_5_Niigata_82p3MHz/Normal/timedata_normal_yyyymmdd_HHMMSS.txt")
    # および
    # "/Ref_Timedata_Toyama_NE_5_Niigata_82p3MHz/Abnrml/timedata_abnrml_yyyymmdd_HHMMSS.txt")
    # JPG画像を生成保存 (--same--.jpg)
    #
    # 3. 対応するスペクトログラム/スカログラムデータを生成して保存 
    # 保存場所：
    # Matlab
    # (Spectrogram_DL_Path + 
    # "/Spectrogram_Train_Toyama_NE_5_Niigata_82p3MHz/Normal/Spectrogram_normal_yyyymmdd_HHMMSS.txt")
    # および
    # "/Spectrogram_Train_Toyama_NE_5_Niigata_82p3MHz/Abnrml/Spectrogram_abnrml_yyyymmdd_HHMMSS.txt")
    # JPG画像を生成保存 (--same--.jpg)
    #
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################


    ###############################################################################################################################################ここから設定項目、ほぼ変更不要
    ###############################################################################################################################################ここから設定項目、ほぼ変更不要
    #
    # 準備
    #
    # 正常、異常時データの区間幅の指定 [回]
    NormalAbnrmlTimeWidthCount  = window_indx_size  # [回](= 6.0 hour / 2 min = 180 等)
    # 正常、異常時データ時間長 [s] = timestep(=120.0 s) * 区間幅の回数（カウント数）
    NormalAbnrmlTimeSpan = RadiowaveDataExtractedTimeStep * NormalAbnrmlTimeWidthCount  #[s] = timestep(120s = 2 min) * 区間幅の回数（カウント数, 360、180回等）
    # データ区間を少しずつずらして学習データ数を増やすための時間シフト （MovingFourierTransformTimeStep ＝３０分の整数倍でなければならない）
    TrainTimeShift  =  spectrogram_timestep * 6.0  # 10分*60秒*6倍 = 3600 [s]  学習用画像データを1時間ずつずらして作成#################################################################################################学習時間シフト  1 h
    #TrainTimeShift =  spectrogram_timestep * 3.0  # 10分*60秒*3倍 = 1800 [s]  学習用画像データを30分ずつずらして作成##################################################################################################学習時間シフト 30 min
    #TrainTimeShift =  spectrogram_timestep * 1.0  # 10分*60秒*1倍 =  600 [s]  学習用画像データを10分ずつずらして作成##################################################################################################学習時間シフト 10 min
    # ウエーブレット変換/フーリエ変換 周波数条件
    SamplingFreq   = 1.0/RadiowaveDataExtractedTimeStep 
    FreqLimitUpper = SamplingFreq/5.0 
    FreqLimitLower = 1.0/NormalAbnrmlTimeSpan 
    ###############################################################################################################################################ここまで設定項目、ほぼ変更不要
    ###############################################################################################################################################ここまで設定項目、ほぼ変更不要

    ####################################################################################################
    # スペクトログラム/スカログラムデータ保存用ディレクトリ名の作成（注意：Pathではない：正常、異常共通）
    ####################################################################################################
    RefRadiowaveData_Directory    = "Ref_Timedata_"          + City_DR_No_BCLoc_Freq  # 参考用時間データ(For Referece)
    RefSpectrogramWaxis_Directory = "Ref_Spectrogram_waxis_" + City_DR_No_BCLoc_Freq  # 比較参考用スペクトログラム/スカログラム画像データ（Reference with Axis）
    SpectrogramTrain_Directory    = "Spectrogram_Train_"     + City_DR_No_BCLoc_Freq  # 学習用スペクトログラム/スカログラム(For Training)
    SpectrogramTrain_Path         = os.path.join(Spectrogram_DeepLearning_Path, SpectrogramTrain_Directory)  # 学習用スペクトログラム/スカログラム格納パス：検証時にも使用

    ###################################
    ###################################
    # 正常時データ生成 (正常 = Normal)
    ###################################
    ###################################

    #####################################
    # 正常時データ保存用ディレクトリ作成
    #####################################
    RefRadiowaveDataNormal_Path    = os.path.join(Spectrogram_DeepLearning_Path, RefRadiowaveData_Directory, "Normal")  # 参考用時間データ(For Referece)
    RefSpectrogramNormalWaxis_Path = os.path.join(Spectrogram_DeepLearning_Path, RefSpectrogramWaxis_Directory,"Normal")  # 比較参考用スペクトログラム/スカログラム画像データ（Reference with Axis）
    SpectrogramTrainNormal_Path    = os.path.join(SpectrogramTrain_Path, "Normal")  # 学習用正常スペクトログラム/スカログラム(For Training)

    # ディレクトリ内の既存のファイルをディレクトリごとすべて一括削除
    # （使用注意：そのディレクトリ内にいる場合はアクセスを失い、後から作成される同名のディレクトリに自動的に移動しない）
    if delete_train_images_before_gen == 1:
        if os.path.isdir(RefRadiowaveDataNormal_Path):
            shutil.rmtree(RefRadiowaveDataNormal_Path)
        if os.path.isdir(RefSpectrogramNormalWaxis_Path):
            shutil.rmtree(RefSpectrogramNormalWaxis_Path)
        if os.path.isdir(SpectrogramTrainNormal_Path):
            shutil.rmtree(SpectrogramTrainNormal_Path)
    # end if

    # 正常時データ保存用ディレクトリがなければ作成
    if not os.path.isdir(RefRadiowaveDataNormal_Path):
        os.makedirs(RefRadiowaveDataNormal_Path)
    if not os.path.isdir(RefSpectrogramNormalWaxis_Path):
        os.makedirs(RefSpectrogramNormalWaxis_Path)
    if not os.path.isdir(SpectrogramTrainNormal_Path):
        os.makedirs(SpectrogramTrainNormal_Path)

    # 学習・検証のための画像データを新たに作成する、または作成しない
    if generate_train_images == 1:  # 新たな画像データの作成: 作成する=1, 作成しない=0(1以外)
        
        ###################
        # 正常時データ生成
        ###################
        # TrainDataListSize = min(NormalListSize, AbnrmlListSize) #小さい方をとってデータ数をそろえる場合（テスト用）
        print('Start Processing RadiowaveData Retreival for Normal State: 正常時データ生成')
        count = 0
        # 正常区間をスイープ    
        print("正常時画像データ生成： 正常時リストのすべてのデータを使用")
        for i in range(NormalListSize):  #正常時データについて繰り返し i = 0:NormalListSize-1 ##########################################################################################Python
        #print("正常時画像データ生成デバッグ：　正常時リストの最初の30データのみ使用")
        #for i in range(30): #正常時データについて繰り返し i = 0:NormalListSize-1 #####################################################################################################デバッグ
            
            # 画像データを少しずつシフトして区間全体に渡って一定時間幅のデータを多数生成
            shiftmax  = NormalEndingTime[i] - NormalStartingTime[i] - NormalAbnrmlTimeSpan  #シフト余剰（最大）時間（秒）
            jshiftmax = int(shiftmax//TrainTimeShift) + 1   # データ区間内での学習区間割り当て可能回数（シフトできなくても１回割り当て可能）'//'は切り捨て除算
            if ((shiftmax > -0.001) and (jshiftmax >= 1)):  # シフト余剰（最大）が正、割り当て可能回数が１回以上ならば画像生成
                for j in range(jshiftmax):  # j = 0:jshiftmax-1 ########################################################################################################################Python
                #for j in range(2): # j = 0:jshiftmax-1 ##############################################################################################################################デバッグ
                    
                    count = count + 1  # スペクトログラム数のカウンター
                    if (int(count/500)*500 == count):
                        print("Number of Normal State Spectrograms for Train 正常時スペクトログラム数 = ", count)
                    # end if

                    ########################################################################
                    # TimedataExtracted の開始時、終了時の Unixtime とそのインデックスを取得
                    ########################################################################
                    #print("NormalStartingTime[i] = ", NormalStartingTime[i])
                    #print("NormalEndingTime[i]   = ", NormalEndingTime[i])
                    #start_uxtime = NormalStartingTime(i) + (j-1)*TrainTimeShift;
                    start_uxtime = int(NormalStartingTime[i] + j*TrainTimeShift)  # Starting Time: 正常時データの開始時 Unixtime
                    #end_uxtime   = start_uxtime + NormalAbnrmlTimePeriodSpectrogram_DeepLearning_Judge_Python
                    #start_index  = int((start_uxtime - RadiowaveDataExtractedStartTime)//RadiowaveDataExtractedTimeStep)+1-1 # 開始時インデックス ０始まり '//'は切り捨て除算（インデックスがMatlabより１小さい）
                    start_index  = int((start_uxtime - RadiowaveDataExtractedStartTime)//RadiowaveDataExtractedTimeStep)+1    # 開始時インデックス ０始まり '//'は切り捨て除算（インデックスをMatlabに合わせておく）
                    end_index    = int(start_index + NormalAbnrmlTimeWidthCount)-1+1                                          # 終了時インデックス ＋１する（開始時インデックスから一定幅後ろへずらした値）

                    ################################################################################
                    # Spectrogram スペクトログラムにおける正常時データの開始時、終了時のインデックス
                    # スペクトログラムの timestep は "spectrogram_timestep"
                    ################################################################################
                    # TrainTimeShift  =  spectrogram_timestep * 1.0  # 30.0*60.0 * 1.0  # =  1800 [s]  30分ずつずらす 
                    #print("indx_conversion_sg2td[0] = ",indx_conversion_sg2td[0])
                    #print("RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]] = ",RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])
                    spectrogram_time_start = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])  # スペクトログラムの最初の Unixtime [s] は Extracted Timedata の最初の時刻に等しくない
                    #print("spectrogram_time_start = ", spectrogram_time_start)
                    TrainSpectrogram_TimeSpan = window_indx_size * timestep  # スペクトログラムの区間幅[sec]は時間窓[sec]と同じに設定する timestep = 120.0 [s]等
                    #print("TrainSpectrogram_TimeSpan = ", TrainSpectrogram_TimeSpan)
                    TrainSpectrogram_IndxSpan = int(TrainSpectrogram_TimeSpan//spectrogram_timestep)   # スペクトログラムの区間幅数[1]は区間幅[sec]を移動フーリエ変換ステップで割ったもの
                    #print("TrainSpectrogram_IndxSpan = ", TrainSpectrogram_IndxSpan)
                    ##################################
                    # Spectrogram の indices 計算結果
                    ##################################
                    TrainSpectrogram_IndxStart = int((start_uxtime - spectrogram_time_start)//spectrogram_timestep)  # スペクトログラム中の正常時プロットの開始時点のインデックス 
                    #print("TrainSpectrogram_IndxStart = ",TrainSpectrogram_IndxStart)
                    TrainSpectrogram_IndxEnd   = TrainSpectrogram_IndxStart + TrainSpectrogram_IndxSpan              # スペクトログラム中の正常時プロットの終了時点のインデックス
                    #print("TrainSpectrogram_IndxEnd   = ",TrainSpectrogram_IndxEnd)

                    ####################################################################################
                    # Extracted Timedata 配列の一部をコピーして正常時の時間区間データを取得（参考のため）
                    ####################################################################################
                    #RadiowaveDataNormal = RadiowaveDataExtracted(start_index:end_index,1:2); # (NormalAbnrmlTimeWidthCount行 x 2列)の配列
                    RadiowaveDataNormal_utime = RadiowaveDataExtracted_utime[start_index:end_index]  # (NormalAbnrmlTimeWidthCount行 x 1列)の配列 Unixtimeを取得
                    RadiowaveDataNormal_tdata = RadiowaveDataExtracted_tdata[start_index:end_index]  # (NormalAbnrmlTimeWidthCount行 x 1列)の配列 dB値ではなくPower[mW]を取得
                    #print("RadiowaveDataNormal_tdata = ", RadiowaveDataNormal_tdata)

                    #######################
                    # 保存用ファイル名生成
                    #######################
                    # unixtimeからdatetimeを生成
                    #filename_datetime = datetime(start_uxtime,'ConvertFrom','posixtime','TimeZone','local','Format','yyyyMMdd_HHmmss'); #datetime変数のまま:文字識別子に注意: Matlab
                    filename_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.fromtimestamp(start_uxtime,tz=None))  # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                    #print(filename_datetime)
                    RadiowaveDataNormal_Fig = "timedata_normal_" + filename_datetime + ".jpg"  # 時間データプロット
                    #print(RadiowaveDataNormal_Fig)
                    RadiowaveDataNormal_Txt = "timedata_normal_" + filename_datetime + ".txt"  # 時間データテキスト
                    #print(RadiowaveDataNormal_Txt)
                    RadiowaveDataNormal_TxtPath = os.path.join(RefRadiowaveDataNormal_Path, RadiowaveDataNormal_Txt)
                    #print(RadiowaveDataNormal_TxtPath)
                    RadiowaveDataNormal_FigPath = os.path.join(RefRadiowaveDataNormal_Path, RadiowaveDataNormal_Fig)
                    #print(RadiowaveDataNormal_FigPath)
                    # Spectrogram image JPGファイル名生成
                    SpectrogramNormalWaxis_Fig = "Spectrogram_waxis_normal_" + filename_datetime + ".jpg"  # スペクトログラム軸ありプロット PNG/JPG ファイル
                    SpectrogramNormal_Fig      = "Spectrogram_normal_" + filename_datetime + ".jpg"        # スペクトログラム軸なしプロット PNG/JPG ファイル <<===== 本番学習用正常スペクトログラムの画像ファイル名！！
                    SpectrogramNormalWaxis_FigPath = os.path.join(RefSpectrogramNormalWaxis_Path, SpectrogramNormalWaxis_Fig)
                    #print(SpectrogramNormalWaxis_FigPath)
                    SpectrogramNormal_FigPath      = os.path.join(SpectrogramTrainNormal_Path, SpectrogramNormal_Fig)  # <<========================================= 本番学習用正常スペクトログラムの画像ファイルパス！！
                    #print(SpectrogramNormal_FigPath)Spectrogram_DeepLearning_Judge_Python

                    # もし同じ名前の「スペクトログラム/スカログラム画像データ」が存在しなければ新たに参考用時間、画像データ、スペクトログラム/スカログラム画像データを作成する、存在すれば新たに作成しない
                    if not os.path.isfile(SpectrogramNormal_FigPath) == 1:

                        if generate_reference_images == 1:  # 参考用画像データ作成する場合
                            ##################################
                            # Timedata （テキストデータ）保存
                            ##################################
                            fileID = open(RadiowaveDataNormal_TxtPath,'w')  # ファイルOPEN
                            #fprintf(fileID, '# unixtime(UTC) dB\n');
                            fileID.write("# Unixtime(UTC)  Power[mW]\n")  # "write"のほうが"print"より速い
                            #for n=1:NormalAbnrmlTimeWidthCount
                            for n in range(NormalAbnrmlTimeWidthCount):
                                tmp_line = "{0:12d} {1:10.3e}\n".format(RadiowaveDataNormal_utime[n],RadiowaveDataNormal_tdata[n])  # 12桁整数 10.3桁指数: Unixtime[s], Power[mW]
                                fileID.write(tmp_line)  # ファイルに書き込み
                            # end for
                            fileID.close

                            ############################
                            # Timedata プロット作成保存
                            ############################
                            t = np.linspace(0,1,len(RadiowaveDataNormal_utime))  # Holizontal axis: time, len(...)-1 と "-1" しないでよい
                            plt.rcParams["font.size"] = 14                       # フォントサイズ
                            fig = plt.figure()                                   # figureオブジェクト定義、デフォルトサイズ 6.4 x 4.8（インチ）
                            #fig = plt.figure(figsize=(8,6))                     # figureオブジェクト定義、サイズ指定（インチ）
                            ax = fig.add_subplot(111)
                            ax.plot(t,RadiowaveDataNormal_tdata)                 # プロット
                            ax.set_xlabel('Time [Samples]')
                            ax.set_ylabel('Radiowave Power [mW]')
                            #ax.set_title('Radiowave Timedata Normal')
                            ax.set_title('Radiowave Timedata Normal 時間データ')  # For Japanize_matplotlib (pip3 install japanize-matplotlib) 日本語対応可能
                            #saveas(gcf,RadiowaveDataNormal_FigPath,'jpg')       # Matlab get current figure: 'jpg' is supported
                            #pause(pausing_time)                                 # Matlab
                            #close                                               # Matlab 現在のFigureをクローズ
                            #fig.savefig(RadiowaveDataNormal_FigPath)  # Python3 matplotlib: 'jpg' is NOT supported, use 'png' instead（文字の一部が隠れる問題あり）
                            fig.savefig(RadiowaveDataNormal_FigPath, bbox_inches="tight")  # Tight bounding box（文字の一部が隠れる問題を回避）
                            #print('Plot Radiowave Timedata for Normal State Saved ラジオ波時間データ（正常）を保存')
                            #plt.show()             # Python3 matplotlib: Display plot, プロット表示確認（ブロック状態：saveしてあるため必ずしも必要ない）############################ Python3 プロット確認（保持）
                            #plt.show(block=False)  # ブロック解除状態（すぐに次の行に移行）########################################################################################## Python3 プロット確認（不保持）
                            #input()                # ブロック解除状態の場合、Enterを押せばプロット終了
                            #input("Press Enter to Continue...")  # ブロック解除状態の場合、Enterを押せばプロット終了
                            plt.cla()   # clear axis ################################################################################################################################# Python3 
                            plt.clf()   # clear figure ############################################################################################################################### Python3 
                            plt.close() # close figure: メモリ開放 (注意：savefigしている場合にメモリ開放されない可能性あり) ######################################################### Python3 プロット終了
                        # end if

                        ############################################################################################################################################
                        ## Generate Spectrogram by WT, ウエーブレット変換によりスカログラムデータ生成 Matlab
                        #signallength = size(RadiowaveDataNormal,1);  # データ数
                        ## The default wavelet used in the cwt filter bank is the analytic Morse (3,60) wavelet.
                        ##filterbank = cwtfilterbank('SignalLength',signallength,'VoicesPerOctave',12); # フィルター設定 Default: Morse(3,60) Wevelet
                        #filterbank = cwtfilterbank('SignalLength',signallength,...
                        #                            'VoicesPerOctave',30,...
                        #                            'TimeBandWidth',100,...
                        #                            'SamplingFrequency',SamplingFreq,...
                        #                            'FrequencyLimits',[FreqLimitLower FreqLimitUpper]);
                        ## Perform Wavelet Transform, ウエーブレット変換
                        #[coeffs,f,coi] = wt(filterbank, RadiowaveDataNormal(:,2)); #Wavelet coefficients, Frequency, Cone of influence: ウェーブレット変換 Matlab
                        ############################################################################################################################################

                        ##################################################################
                        # Extract Spectrogram Data, スペクトログラムデータ取り出し Python3
                        ################################################################## 
                        # Generate 3D plot data: 
                        spectrogram_tmp = spectrogram[TrainSpectrogram_IndxStart:TrainSpectrogram_IndxEnd,:]
                        #print("Extract Spectrogram Data for Normal State: spectrogram_tmp = ",spectrogram_tmp)
                        #print("Extract Spectrogram Data for Normal State: spectrogram_tmp.shape = ",spectrogram_tmp.shape)
                        #input("Extract Spectrogram Data for Normal State: Press Enter to Continue...")

                        if generate_reference_images == 1:  # 参考用画像データ作成する場合
                            #########################################
                            # Spectrogram with axes（参考用プロット）
                            #########################################
                            fig = plt.figure()
                            #plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), vmin=-280, vmax=-190)  # Plot (default: 'viridis')
                            plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), cmap="jet", vmin=-280, vmax=-190)  # Plot in 'jet'
                            plt.title("Plot 2D array")
                            #plt.show() # 表示
                            fig.savefig(SpectrogramNormalWaxis_FigPath, bbox_inches="tight")  # 保存: Tight bounding box（文字の一部が隠れる問題を回避）
                            #print('Plot Radiowave Spectrogram with Axes for Normal State Saved ラジオ波スペクトログラム軸あり（正常）を保存')
                            plt.cla()   # clear axis
                            plt.clf()   # clear figure
                            plt.close()
                        # end if

                        #############################################
                        # Spectrogram without axes with fixed pixels
                        # 学習用画像の生成（軸なし）
                        #############################################
                        fig = plt.figure()
                        #plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), vmin=-280, vmax=-190)  # Plot (default: 'viridis')
                        plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), cmap="jet", vmin=-280, vmax=-190)  # Plot in 'jet'
                        plt.axis("off")
                        #plt.show() # 表示
                        #fig.savefig(SpectrogramNormal_FigPath, bbox_inches="tight")  # 保存: Tight bounding box（文字の一部が隠れる問題を回避）白い余白が生じる
                        fig.savefig(SpectrogramNormal_FigPath, transparent=False, bbox_inches="tight", pad_inches = 0)  # 保存: 透明背景なし, 余白なし
                        #print('Plot Radiowave Spectrogram with Axes for Normal State Saved ラジオ波スペクトログラム軸なし（正常）を保存')
                        plt.cla()   # clear axis
                        plt.clf()   # clear figure
                        plt.close()
                        #img = plt.imread(SpectrogramNormal_FigPath)  # 点検のためイメージ読み込み
                        #print(img.shape)                             # ピクセル数表示                    
                        #########################
                        # Pillowで画像サイズ変換、再保存...効率化可能
                        #########################
                        img = Image.open(SpectrogramNormal_FigPath)  # 画像の読み込み
                        #img_resize = img.resize(size=(227,227), resample=Image.NEAREST)     # リサイズ処理: Matlab AlexNet では 227 X 227, Matlab ではネットワークによってサイズが異なる
                        img_resize = img.resize(size=(224,224), resample=Image.NEAREST)      # リサイズ処理: Python のネットワークは全て 224 x 224
                        img_resize = img_resize.save(SpectrogramNormal_FigPath, quality=95)  # ファイルに保存 <<==================================================== 本番学習用正常スペクトログラムの画像をファイルに保存！！
                        # 画像サイズチェック
                        #img = plt.imread(SpectrogramNormal_FigPath)  # イメージ読み込み
                        #print(img.shape)                             # ピクセル数表示
                    #
                    # end if
                # end for j
            # end if
        # end for i
        print("Generating Train Images for Normal State Completed.")
        print("Total Number of Normal State Spectrograms Generated for Training 全学習用正常時スペクトログラム生成数 = ", count)
    #
    # end if # 正常時画像データを作成する場合の終了




    ###################################
    ###################################
    # 異常時データ生成 (異常 = Abnrml)  ===============================================>> Abnrml: 異常データ全体の集計を行う; AbnLrg, AbnSml の２クラス分け, ディレクトリ作成, それぞれデータを格納
    ###################################
    ###################################

    #####################################
    # 異常時データ保存用ディレクトリ作成  ===========================================================================================================================>> AbnLrg, AbnSml の２クラス用の両方のディレクトリを予め作成しておく
    #####################################

    # ディレクトリ名の生成
    # 元のディレクトリ(あとでコメントアウトする)
    #RefRadiowaveDataAbnrml_Path    = os.path.join(Spectrogram_DeepLearning_Path, RefRadiowaveData_Directory, "Abnrml")    # 参考用異常時間データ(For Reference)
    #RefSpectrogramAbnrmlWaxis_Path = os.path.join(Spectrogram_DeepLearning_Path, RefSpectrogramWaxis_Directory,"Abnrml")  # 比較参考用異常スペクトログラム/スカログラム画像データ（Reference with Axis）
    #SpectrogramTrainAbnrml_Path    = os.path.join(SpectrogramTrain_Path, "Abnrml")                                        # 学習用異常スペクトログラム/スカログラム(For Training)
    # large用
    RefRadiowaveDataAbnLrg_Path    = os.path.join(Spectrogram_DeepLearning_Path, RefRadiowaveData_Directory, "AbnLrg")    # 参考用異常時間データ(For Reference)
    RefSpectrogramAbnLrgWaxis_Path = os.path.join(Spectrogram_DeepLearning_Path, RefSpectrogramWaxis_Directory,"AbnLrg")  # 比較参考用異常スペクトログラム/スカログラム画像データ（Reference with Axis）
    SpectrogramTrainAbnLrg_Path    = os.path.join(SpectrogramTrain_Path, "AbnLrg")                                        # 学習用異常スペクトログラム/スカログラム(For Training)
    # Small用
    RefRadiowaveDataAbnSml_Path    = os.path.join(Spectrogram_DeepLearning_Path, RefRadiowaveData_Directory, "AbnSml")    # 参考用異常時間データ(For Reference)
    RefSpectrogramAbnSmlWaxis_Path = os.path.join(Spectrogram_DeepLearning_Path, RefSpectrogramWaxis_Directory,"AbnSml")  # 比較参考用異常スペクトログラム/スカログラム画像データ（Reference with Axis）
    SpectrogramTrainAbnSml_Path    = os.path.join(SpectrogramTrain_Path, "AbnSml")                                        # 学習用異常スペクトログラム/スカログラム(For Training)

    #################################################################
    # ディレクトリ内の既存のファイルをディレクトリごとすべて一括削除
    #################################################################
    # （使用注意：そのディレクトリ内にいる場合はアクセスを失い、後から作成される同名のディレクトリに自動的に移動しない）
    if delete_train_images_before_gen == 1:
        # 元のディレクトリ(あとでコメントアウトする)
        #if os.path.isdir(RefRadiowaveDataAbnrml_Path):
        #    shutil.rmtree(RefRadiowaveDataAbnrml_Path)
        #if os.path.isdir(RefSpectrogramAbnrmlWaxis_Path):
        #    shutil.rmtree(RefSpectrogramAbnrmlWaxis_Path)
        #if os.path.isdir(SpectrogramTrainAbnrml_Path):
        #    shutil.rmtree(SpectrogramTrainAbnrml_Path)
        # large用
        if os.path.isdir(RefRadiowaveDataAbnLrg_Path):
            shutil.rmtree(RefRadiowaveDataAbnLrg_Path)
        if os.path.isdir(RefSpectrogramAbnLrgWaxis_Path):
            shutil.rmtree(RefSpectrogramAbnLrgWaxis_Path)
        if os.path.isdir(SpectrogramTrainAbnLrg_Path):
            shutil.rmtree(SpectrogramTrainAbnLrg_Path)
        # Small用
        if os.path.isdir(RefRadiowaveDataAbnSml_Path):
            shutil.rmtree(RefRadiowaveDataAbnSml_Path)
        if os.path.isdir(RefSpectrogramAbnSmlWaxis_Path):
            shutil.rmtree(RefSpectrogramAbnSmlWaxis_Path)
        if os.path.isdir(SpectrogramTrainAbnSml_Path):
            shutil.rmtree(SpectrogramTrainAbnSml_Path)
    # end if

    ###############################################
    # 異常時データ保存用ディレクトリがなければ作成
    ###############################################
    # 元のディレクトリ(あとでコメントアウトする)
    #if not os.path.isdir(RefRadiowaveDataAbnrml_Path):
    #    os.makedirs(RefRadiowaveDataAbnrml_Path)
    #if not os.path.isdir(RefSpectrogramAbnrmlWaxis_Path):
    #    os.makedirs(RefSpectrogramAbnrmlWaxis_Path)
    #if not os.path.isdir(SpectrogramTrainAbnrml_Path):
    #    os.makedirs(SpectrogramTrainAbnrml_Path)
    # large用
    if not os.path.isdir(RefRadiowaveDataAbnLrg_Path):
        os.makedirs(RefRadiowaveDataAbnLrg_Path)
    if not os.path.isdir(RefSpectrogramAbnLrgWaxis_Path):
        os.makedirs(RefSpectrogramAbnLrgWaxis_Path)
    if not os.path.isdir(SpectrogramTrainAbnLrg_Path):
        os.makedirs(SpectrogramTrainAbnLrg_Path)
    # Small用
    if not os.path.isdir(RefRadiowaveDataAbnSml_Path):
        os.makedirs(RefRadiowaveDataAbnSml_Path)
    if not os.path.isdir(RefSpectrogramAbnSmlWaxis_Path):
        os.makedirs(RefSpectrogramAbnSmlWaxis_Path)
    if not os.path.isdir(SpectrogramTrainAbnSml_Path):
        os.makedirs(SpectrogramTrainAbnSml_Path)


    # 学習・検証のための画像データを新たに作成する:1、または作成しない:0
    if generate_train_images == 1:  # 新たな画像データの作成: 作成する=1, 作成しない=0(1以外)
        
        ###################
        # 異常時データ生成
        ###################
        # TrainDataListSize = min(NormalListSize, AbnrmlListSize) #小さい方をとってデータ数をそろえる場合（テスト用）
        print('Start Processing RadiowaveData Retreival for Abnrml State: 異常時データ生成')
        count = 0
        # 異常区間をスイープ
        print("異常時画像データ生成： 異常時リストのすべてのデータを使用")    
        for i in range(AbnrmlListSize):  #異常時データについて繰り返し i = 0:AbnrmlListSize-1 ##########################################################################################Python
        #print("異常時画像データ生成デバッグ：　異常時リストの最初の30データのみ使用")
        #for i in range(30): #異常時データについて繰り返し i = 0:AbnrmlListSize-1 #####################################################################################################デバッグ

            # 画像データを少しずつシフトして区間全体に渡って一定時間幅のデータを多数生成
            shiftmax  = AbnrmlEndingTime[i] - AbnrmlStartingTime[i] - NormalAbnrmlTimeSpan  #シフト余剰（最大）時間（秒）
            jshiftmax = int(shiftmax//TrainTimeShift) + 1   # データ区間内での学習区間割り当て可能回数（シフトできなくても１回割り当て可能）'//'は切り捨て除算
            if ((shiftmax > -0.001) and (jshiftmax >= 1)):  # シフト余剰（最大）が正、割り当て可能回数が１回以上ならば画像生成
                for j in range(jshiftmax):  # j = 0:jshiftmax-1 ########################################################################################################################Python
                #for j in range(2): # j = 0:jshiftmax-1 ##############################################################################################################################デバッグ
                    
                    count = count + 1  # スペクトログラム数のカウンター
                    if (int(count/500)*500 == count):
                        print("Number of Abnormal State Spectrograms for Train 異常時スペクトログラム数 = ", count)
                    # end if

                    ########################################################################
                    # TimedataExtracted の開始時、終了時の Unixtime とそのインデックスを取得
                    ########################################################################
                    #print("AbnrmlStartingTime[i] = ", AbnrmlStartingTime[i])
                    #print("AbnrmlEndingTime[i]   = ", AbnrmlEndingTime[i])
                    #start_uxtime = AbnrmlStartingTime(i) + (j-1)*TrainTimeShift;
                    start_uxtime = int(AbnrmlStartingTime[i] + j*TrainTimeShift)  # Startitme Time: 異常時データの開始時 Unixtime
                    #end_uxtime   = start_uxtime + NormalAbnrmlTimePeriodSpectrogram_DeepLearning_Judge_Python
                    #start_index  = int((start_uxtime - RadiowaveDataExtractedStartTime)//RadiowaveDataExtractedTimeStep)+1-1 # 開始時インデックス ０始まり '//'は切り捨て除算（インデックスがMatlabより１小さい）
                    start_index  = int((start_uxtime - RadiowaveDataExtractedStartTime)//RadiowaveDataExtractedTimeStep)+1    # 開始時インデックス ０始まり '//'は切り捨て除算（インデックスをMatlabに合わせておく）
                    end_index    = int(start_index + NormalAbnrmlTimeWidthCount)-1+1                                          # 終了時インデックス ＋１する（開始時インデックスから一定幅後ろへずらした値）

                    ################################################################################
                    # Spectrogram スペクトログラムにおける異常時データの開始時、終了時のインデックス
                    # スペクトログラムの timestep は "spectrogram_timestep"
                    ################################################################################
                    # TrainTimeShift  =  spectrogram_timestep * 1.0  # 30.0*60.0 * 1.0  # =  1800 [s]  30分ずつずらす 
                    #print("indx_conversion_sg2td[0] = ",indx_conversion_sg2td[0])
                    #print("RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]] = ",RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])
                    spectrogram_time_start = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])  # スペクトログラムの最初の Unixtime [s] は Extracted Timedata の最初の時刻に等しくない
                    #print("spectrogram_time_start = ", spectrogram_time_start)
                    TrainSpectrogram_TimeSpan = window_indx_size * timestep  # スペクトログラムの区間幅[sec]は時間窓[sec]と同じに設定する timestep = 120.0 [s]等
                    #print("TrainSpectrogram_TimeSpan = ", TrainSpectrogram_TimeSpan)
                    TrainSpectrogram_IndxSpan = int(TrainSpectrogram_TimeSpan//spectrogram_timestep)   # スペクトログラムの区間幅数[1]は区間幅[sec]を移動フーリエ変換ステップで割ったもの
                    #print("TrainSpectrogram_IndxSpan = ", TrainSpectrogram_IndxSpan)
                    ##################################
                    # Spectrogram の indices 計算結果
                    ##################################
                    TrainSpectrogram_IndxStart = int((start_uxtime - spectrogram_time_start)//spectrogram_timestep)  # スペクトログラム中の異常時プロットの開始時点のインデックス 
                    #print("TrainSpectrogram_IndxStart = ",TrainSpectrogram_IndxStart)
                    TrainSpectrogram_IndxEnd   = TrainSpectrogram_IndxStart + TrainSpectrogram_IndxSpan              # スペクトログラム中の異常時プロットの終了時点のインデックス
                    #print("TrainSpectrogram_IndxEnd   = ",TrainSpectrogram_IndxEnd)

                    ####################################################################################
                    # Extracted Timedata 配列の一部をコピーして異常時の時間区間データを取得（参考のため）
                    ####################################################################################
                    #RadiowaveDataAbnrml = RadiowaveDataExtracted(start_index:end_index,1:2); # (NormalAbnrmlTimeWidthCount行 x 2列)の配列
                    RadiowaveDataAbnrml_utime = RadiowaveDataExtracted_utime[start_index:end_index]  # (NormalAbnrmlTimeWidthCount行 x 1列)の配列 Unixtimeを取得
                    RadiowaveDataAbnrml_tdata = RadiowaveDataExtracted_tdata[start_index:end_index]  # (NormalAbnrmlTimeWidthCount行 x 1列)の配列 dB値ではなくPower[mW]を取得
                    #print("RadiowaveDataAbnrml_tdata = ", RadiowaveDataAbnrml_tdata)

                    #################################
                    # 保存用ファイル名およびパス生成    ==========================================================================================================================================================================>> AbnLrg, AbnSml の２クラス用ファイル、パス作成
                    #################################
                    # unixtimeからdatetimeを生成
                    #filename_datetime = datetime(start_uxtime,'ConvertFrom','posixtime','TimeZone','local','Format','yyyyMMdd_HHmmss'); #datetime変数のまま:文字識別子に注意: Matlab
                    filename_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.fromtimestamp(start_uxtime,tz=None))  # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                    #print(filename_datetime)
                    #
                    # 元のファイル名（あとでコメントアウトする）
                    #
                    # RadiowaveDataAbnrml_Fig = "timedata_abnrml_" + filename_datetime + ".jpg"  # 時間データプロット
                    # #print(RadiowaveDataAbnrml_Fig)
                    # RadiowaveDataAbnrml_Txt = "timedata_abnrml_" + filename_datetime + ".txt"  # 時間データテキスト
                    # #print(RadiowaveDataAbnrml_Txt)
                    # RadiowaveDataAbnrml_TxtPath = os.path.join(RefRadiowaveDataAbnrml_Path, RadiowaveDataAbnrml_Txt)
                    # #print(RadiowaveDataAbnrml_TxtPath)
                    # RadiowaveDataAbnrml_FigPath = os.path.join(RefRadiowaveDataAbnrml_Path, RadiowaveDataAbnrml_Fig)
                    # #print(RadiowaveDataAbnrml_FigPath)
                    # # Spectrogram image JPGファイル名生成
                    # SpectrogramAbnrmlWaxis_Fig = "Spectrogram_waxis_abnrml_" + filename_datetime + ".jpg"  # スペクトログラム軸ありプロット PNG/JPG ファイル
                    # SpectrogramAbnrml_Fig      = "Spectrogram_abnrml_" + filename_datetime + ".jpg"        # スペクトログラム軸なしプロット PNG/JPG ファイル <<===== 本番学習用異常スペクトログラムの画像ファイル名！！ ===========>> AbnLrg, AbnSml の２クラスに変更
                    # SpectrogramAbnrmlWaxis_FigPath = os.path.join(RefSpectrogramAbnrmlWaxis_Path, SpectrogramAbnrmlWaxis_Fig)
                    # #print(SpectrogramAbnrmlWaxis_FigPath)
                    # SpectrogramAbnrml_FigPath      = os.path.join(SpectrogramTrainAbnrml_Path, SpectrogramAbnrml_Fig)  # <<========================================= 本番学習用異常スペクトログラムの画像ファイルパス！！ ===========>> AbnLrg, AbnSml の２クラスに変更
                    # #print(SpectrogramAbnrml_FigPath)
                    #
                    ###############################################################################################################################################################################################################################
                    # ファイル名の一部の AbnLrg または AbnSml の区別分け（Large または Small によってファイル名、パスの両方を区別するか、パスだけ区別する：両方区別するほうがあとでわかりやすいか、パスだけ区別するほうが簡単か ---> 両方区別する）
                    # i: index of abnormal list, 異常信号のインデックス（異常信号のリストの中でi番目）, i = 0, AbnrmlListSize-1 番目までのすべての異常信号
                    # 
                    # 変数名は変えずに、変数の中身を変えること（Large または Small によってファイル名、パスの両方を区別する）
                    ###############################################################################################################################################################################################################################
                    # テスト表示
                    # print("この異常時データに対応する最大震度           AbnrmlState_MaxSI_Max[i]  （整数）= ",AbnrmlState_MaxSI_Max[i])
                    # print("この異常時データに対応する検索震度           AbnrmlState_LocalSI_Max[i]（整数） = ",AbnrmlState_LocalSI_Max[i])
                    # print("この異常時データに対応する最大マグニチュード AbnrmlState_Mag_Max[i]    （実数） = ",AbnrmlState_Mag_Max[i])
                    #
                    
                    # 観測データによってどの基準を使用するか選択する
                    AbnrmlState_LocalSI_Max_Boundary = 4  # 検索震度（整数）で分ける場合の境界のデフォルト値
                    if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
                         City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"  or
                         City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):     # "Toyama_NE_5_Niigata_82p3MHz"の場合：確認済み
                        AbnrmlState_LocalSI_Max_Boundary = 4                                                                                    # 検索震度（整数）で分ける場合 (Niigata: 検索震度３の場合の画像数 Large=8000, Small=3000, 検索震度４の場合 Large=3700, Small=7500)
                    elif(City_DR_No_BCLoc_Freq == "Iwata_NE_11_Yokohama_84p7MHz" or City_DR_No_BCLoc_Freq == "Iwata_NE_16_Shizuoka_88p8MHz" or
                         City_DR_No_BCLoc_Freq == "Iwata_NE_4_Shizuoka_79.2MHz"  or City_DR_No_BCLoc_Freq == "Iwata_NW_2_Tsu_78p9MHz"       or
                         City_DR_No_BCLoc_Freq == "Iwata_NW_8_Nagoya_82p5MHz"):                                                                 # "Iwata_NW_2_Tsu_78p9MHz" の場合：確認中
                        AbnrmlState_LocalSI_Max_Boundary = 3                                                                                    # 検索震度（整数）で分ける場合 (Iwata:   検索震度３の場合の画像数 Large=3000, Small=9000, Normal=19200)
                    # end if

                    if (AbnrmlState_LocalSI_Max[i] >= AbnrmlState_LocalSI_Max_Boundary): # 検索震度（整数）で分ける場合 ================================================================================== 検索震度のしきい値を調整する必要あり!!
                        #
                        # Large
                        #
                        RadiowaveDataAbnrml_Fig = "timedata_abnlrg_" + filename_datetime + ".jpg"  # 時間データプロット
                        #print(RadiowaveDataAbnrml_Fig)
                        RadiowaveDataAbnrml_Txt = "timedata_abnlrg_" + filename_datetime + ".txt"  # 時間データテキスト
                        #print(RadiowaveDataAbnrml_Txt)
                        RadiowaveDataAbnrml_TxtPath = os.path.join(RefRadiowaveDataAbnLrg_Path, RadiowaveDataAbnrml_Txt)
                        #print(RadiowaveDataAbnrml_TxtPath)
                        RadiowaveDataAbnrml_FigPath = os.path.join(RefRadiowaveDataAbnLrg_Path, RadiowaveDataAbnrml_Fig)
                        #print(RadiowaveDataAbnrml_FigPath)
                        # Spectrogram image JPGファイル名生成
                        SpectrogramAbnrmlWaxis_Fig = "Spectrogram_waxis_abnlrg_" + filename_datetime + ".jpg"  # スペクトログラム軸ありプロット PNG/JPG ファイル
                        SpectrogramAbnrml_Fig      = "Spectrogram_abnlrg_" + filename_datetime + ".jpg"        # スペクトログラム軸なしプロット PNG/JPG ファイル <<===== 本番学習用異常スペクトログラムの画像ファイル名！！ ===========>> AbnLrg, AbnSml の２クラスに変更
                        SpectrogramAbnrmlWaxis_FigPath = os.path.join(RefSpectrogramAbnLrgWaxis_Path, SpectrogramAbnrmlWaxis_Fig)
                        #print(SpectrogramAbnrmlWaxis_FigPath)
                        SpectrogramAbnrml_FigPath      = os.path.join(SpectrogramTrainAbnLrg_Path, SpectrogramAbnrml_Fig)  # <<========================================= 本番学習用異常スペクトログラムの画像ファイルパス！！ ===========>> AbnLrg, AbnSml の２クラスに変更
                        #print(SpectrogramAbnrml_FigPath)
                    else:
                        #
                        # Small
                        #
                        RadiowaveDataAbnrml_Fig = "timedata_abnsml_" + filename_datetime + ".jpg"  # 時間データプロット
                        #print(RadiowaveDataAbnrml_Fig)
                        RadiowaveDataAbnrml_Txt = "timedata_abnsml_" + filename_datetime + ".txt"  # 時間データテキスト
                        #print(RadiowaveDataAbnrml_Txt)
                        RadiowaveDataAbnrml_TxtPath = os.path.join(RefRadiowaveDataAbnSml_Path, RadiowaveDataAbnrml_Txt)
                        #print(RadiowaveDataAbnrml_TxtPath)
                        RadiowaveDataAbnrml_FigPath = os.path.join(RefRadiowaveDataAbnSml_Path, RadiowaveDataAbnrml_Fig)
                        #print(RadiowaveDataAbnrml_FigPath)
                        # Spectrogram image JPGファイル名生成
                        SpectrogramAbnrmlWaxis_Fig = "Spectrogram_waxis_abnsml_" + filename_datetime + ".jpg"  # スペクトログラム軸ありプロット PNG/JPG ファイル
                        SpectrogramAbnrml_Fig      = "Spectrogram_abnsml_" + filename_datetime + ".jpg"        # スペクトログラム軸なしプロット PNG/JPG ファイル <<===== 本番学習用異常スペクトログラムの画像ファイル名！！ ===========>> AbnLrg, AbnSml の２クラスに変更
                        SpectrogramAbnrmlWaxis_FigPath = os.path.join(RefSpectrogramAbnSmlWaxis_Path, SpectrogramAbnrmlWaxis_Fig)
                        #print(SpectrogramAbnrmlWaxis_FigPath)
                        SpectrogramAbnrml_FigPath      = os.path.join(SpectrogramTrainAbnSml_Path, SpectrogramAbnrml_Fig)  # <<========================================= 本番学習用異常スペクトログラムの画像ファイルパス！！ ===========>> AbnLrg, AbnSml の２クラスに変更
                        #print(SpectrogramAbnrml_FigPath)
                    # end if


                    #########################
                    # 異常時ファイル生成保存
                    #########################
                    # もし同じ名前の「スペクトログラム/スカログラム画像データ」が存在しなければ新たに参考用時間、画像データ、スペクトログラム/スカログラム画像データを作成する、存在すれば新たに作成しない
                    if not os.path.isfile(SpectrogramAbnrml_FigPath) == 1:

                        if generate_reference_images == 1:  # 参考用画像データ作成する場合
                            ##################################
                            # Timedata （テキストデータ）保存
                            ##################################
                            fileID = open(RadiowaveDataAbnrml_TxtPath,'w')  # ファイルOPEN
                            #fprintf(fileID, '# unixtime(UTC) dB\n');
                            fileID.write("# Unixtime(UTC)  Power[mW]\n")  # "write"のほうが"print"より速い
                            #for n=1:NormalAbnrmlTimeWidthCount
                            for n in range(NormalAbnrmlTimeWidthCount):
                                tmp_line = "{0:12d} {1:10.3e}\n".format(RadiowaveDataAbnrml_utime[n],RadiowaveDataAbnrml_tdata[n])  # 12桁整数 10.3桁指数: Unixtime[s], Power[mW]
                                fileID.write(tmp_line)  # ファイルに書き込み
                            # end for
                            fileID.close

                            ############################
                            # Timedata プロット作成保存
                            ############################
                            t = np.linspace(0,1,len(RadiowaveDataAbnrml_utime))  # Holizontal axis: time, len(...)-1 と "-1" しないでよい
                            plt.rcParams["font.size"] = 14                       # フォントサイズ
                            fig = plt.figure()                                   # figureオブジェクト定義、デフォルトサイズ 6.4 x 4.8（インチ）
                            #fig = plt.figure(figsize=(8,6))                     # figureオブジェクト定義、サイズ指定（インチ）
                            ax = fig.add_subplot(111)
                            ax.plot(t,RadiowaveDataAbnrml_tdata)                 # プロット
                            ax.set_xlabel('Time [Samples]')
                            ax.set_ylabel('Radiowave Power [mW]')
                            #ax.set_title('Radiowave Timedata Abnrml')
                            ax.set_title('Radiowave Timedata Abnrml 時間データ')  # For Japanize_matplotlib (pip3 install japanize-matplotlib) 日本語対応可能
                            #saveas(gcf,RadiowaveDataAbnrml_FigPath,'jpg')       # Matlab get current figure: 'jpg' is supported
                            #pause(pausing_time)                                 # Matlab
                            #close                                               # Matlab 現在のFigureをクローズ
                            #fig.savefig(RadiowaveDataAbnrml_FigPath)  # Python3 matplotlib: 'jpg' is NOT supported, use 'png' instead（文字の一部が隠れる問題あり）
                            fig.savefig(RadiowaveDataAbnrml_FigPath, bbox_inches="tight")  # Tight bounding box（文字の一部が隠れる問題を回避）
                            #print('Plot Radiowave Timedata for Abnrml State Saved ラジオ波時間データ（異常）を保存')
                            #plt.show()             # Python3 matplotlib: Display plot, プロット表示確認（ブロック状態：saveしてあるため必ずしも必要ない）############################ Python3 プロット確認（保持）
                            #plt.show(block=False)  # ブロック解除状態（すぐに次の行に移行）########################################################################################## Python3 プロット確認（不保持）
                            #input()                # ブロック解除状態の場合、Enterを押せばプロット終了
                            #input("Press Enter to Continue...")  # ブロック解除状態の場合、Enterを押せばプロット終了
                            plt.cla()   # clear axis ################################################################################################################################# Python3 
                            plt.clf()   # clear figure ############################################################################################################################### Python3 
                            plt.close() # close figure: メモリ開放 (注意：savefigしている場合にメモリ開放されない可能性あり) ######################################################### Python3 プロット終了
                        # end if

                        ############################################################################################################################################
                        ## Generate Spectrogram by WT, ウエーブレット変換によりスカログラムデータ生成 Matlab
                        #signallength = size(RadiowaveDataAbnrml,1);  # データ数
                        ## The default wavelet used in the cwt filter bank is the analytic Morse (3,60) wavelet.
                        ##filterbank = cwtfilterbank('SignalLength',signallength,'VoicesPerOctave',12); # フィルター設定 Default: Morse(3,60) Wevelet
                        #filterbank = cwtfilterbank('SignalLength',signallength,...
                        #                            'VoicesPerOctave',30,...
                        #                            'TimeBandWidth',100,...
                        #                            'SamplingFrequency',SamplingFreq,...
                        #                            'FrequencyLimits',[FreqLimitLower FreqLimitUpper]);
                        ## Perform Wavelet Transform, ウエーブレット変換
                        #[coeffs,f,coi] = wt(filterbank, RadiowaveDataAbnrml(:,2)); #Wavelet coefficients, Frequency, Cone of influence: ウェーブレット変換 Matlab
                        ############################################################################################################################################

                        ##################################################################
                        # Extract Spectrogram Data, スペクトログラムデータ取り出し Python3
                        ################################################################## 
                        # Generate 3D plot data: 
                        spectrogram_tmp = spectrogram[TrainSpectrogram_IndxStart:TrainSpectrogram_IndxEnd,:]
                        #print("Extract Spectrogram Data for Abnormal State: spectrogram_tmp = ",spectrogram_tmp)
                        #print("Extract Spectrogram Data for Abnormal State: spectrogram_tmp.shape = ",spectrogram_tmp.shape)
                        #input("Extract Spectrogram Data for Abnormal State: Press Enter to Continue...")

                        if generate_reference_images == 1:  # 参考用画像データ作成する場合
                            #########################################
                            # Spectrogram with axes（参考用プロット）
                            #########################################
                            fig = plt.figure()
                            #plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), vmin=-280, vmax=-190)  # Plot (default: 'viridis')
                            plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), cmap="jet", vmin=-280, vmax=-190)  # Plot in 'jet'
                            plt.title("Plot 2D array")
                            #plt.show()  # 表示
                            fig.savefig(SpectrogramAbnrmlWaxis_FigPath, bbox_inches="tight")  # 保存: Tight bounding box（文字の一部が隠れる問題を回避）
                            #print('Plot Radiowave Spectrogram with Axes for Abnrml State Saved ラジオ波スペクトログラム軸あり（異常）を保存')
                            plt.cla()   # clear axis
                            plt.clf()   # clear figure
                            plt.close()
                        # end if

                        #############################################
                        # Spectrogram without axes with fixed pixels
                        # 学習用画像の生成保存（軸なし）
                        #############################################
                        fig = plt.figure()
                        #plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), vmin=-280, vmax=-190)  # Plot (default: 'viridis')
                        plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), cmap="jet", vmin=-280, vmax=-190)  # Plot in 'jet'
                        plt.axis("off")
                        #plt.show() # 表示
                        #fig.savefig(SpectrogramAbnrml_FigPath, bbox_inches="tight")  # 保存: Tight bounding box（文字の一部が隠れる問題を回避）白い余白が生じる
                        fig.savefig(SpectrogramAbnrml_FigPath, transparent=False, bbox_inches="tight", pad_inches = 0)  # 保存: 透明背景なし, 余白なし
                        #print('Plot Radiowave Spectrogram with Axes for Abnrml State Saved ラジオ波スペクトログラム軸なし（異常）を保存')
                        plt.cla()   # clear axis
                        plt.clf()   # clear figure
                        plt.close()
                        #img = plt.imread(SpectrogramAbnrml_FigPath)  # 点検のためイメージ読み込み
                        #print(img.shape)                             # ピクセル数表示                    
                        ####################################################
                        # Pillowで画像サイズ変換、再保存...あとで効率化可能
                        ####################################################
                        img = Image.open(SpectrogramAbnrml_FigPath)  # 画像の読み込み
                        #img_resize = img.resize(size=(227,227), resample=Image.NEAREST)     # リサイズ処理: AlexNet では 227 X 227 (Matlabの場合)？ For Python 224 x 224
                        img_resize = img.resize(size=(224,224), resample=Image.NEAREST)      # リサイズ処理: AlexNet では 227 X 227 (Matlabの場合)？ For Python 224 x 224
                        img_resize = img_resize.save(SpectrogramAbnrml_FigPath, quality=95)  # ファイルに保存 <<======================================================================= 本番学習用異常スペクトログラムの画像をファイルに保存！！
                        # 画像サイズチェック
                        #img = plt.imread(SpectrogramAbnrml_FigPath)  # イメージ読み込み
                        #print(img.shape)                             # ピクセル数表示
                    #
                    # end if
                # end for j
            # end if
        # end for i
        print("Generating Train Images for Abnormal State Completed.")
        print("Total Number of Abnormal State Spectrograms Generated for Training 全学習用異常時スペクトログラム生成数 = ", count)
    #
    # end if # 異常時画像データを作成する場合の終了

    #########################################################
    #########################################################
    # 正常区間、異常区間のスペクトログラム画像生成、保存完了  ====================== 変更ここまで終了、チェックすること
    #########################################################
    #########################################################

    #input(" Check Train Data for 3 Classes ３クラス用の学習データをチェック; Start Training Process 学習プロセス開始, press Enter to Continue ...")





    ######################################
    ######################################
    # Train 学習（訓練）& Validation 検証
    ######################################
    ######################################

    # #################################################################################################################
    # # 参考：PNG画像の表示例　その１ (PIL, チャンネルはRGB順)
    # from PIL import Image
    # im = Image.open("../Spectrogram_Train_Toyama_NE_5_Niigata_82p3MHz/Abnrml/Spectrogram_abnrml_20200126_210000.png")
    # im.show()
    # #################################################################################################################

    # #################################################################################################################
    # # 参考：PNG画像の表示例　その２ (numpy)
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # import numpy as np
    # im = Image.open("../Spectrogram_Train_Toyama_NE_5_Niigata_82p3MHz/Abnrml/Spectrogram_abnrml_20200126_210000.png")
    # im_list = np.asarray(im)
    # plt.imshow(im_list)
    # plt.show()
    ###################################################################################################################

    ###########################################################################
    # スペクトログラムデータを学習(Train)データと検証(Valid, Test)データに分割
    ###########################################################################
    # import PyTorch, Dataset, DataLoader
    #
    # 画像データ変換モジュール torchvision.transforms 使用例
    # import torchvision.transforms as transforms
    # data_transform = transforms.Compose([
    #                  transforms.Resize(256, 256),  # サイズ変換
    #                  transforms.ToTensor(),        # テンソルに変換
    #                  ])
    #
    #
    # または、画像をテンソル化した後に、イメージ画像のデータ変換（標準化など）を行うクラスを定義
    # class ImageTransform():
    #   def __init__(self, mean, std):
    #        self.data_transform = transforms.Compose([
    #            transforms.Resize(256, 256),     # サイズ変換
    #            transforms.ToTensor(),           # テンソルに変換
    #            transforms.Normalize(mean, std)  # 規格化
    #        ])
    #    
    #    def __call__(self, img):
    #        return self.data_transform(img)

    ################################################################################################
    # Image Data Transform 画像データ変換（Pytorch用テンソルに変換）する ImageTransformクラスを定義
    ################################################################################################
    # class ImageTransform():
    #     def __init__(self):
    #         self.data_transform = transforms.Compose([  # 画像データ変換モジュール torchvision.transforms.Compose 使用
    #             transforms.ToTensor(),                  # ToTensor テンソルに変換
    #         ])                                          # （注意：Resizeでサイズ変換するとエラーが出る；サイズは画像生成時に指定）
    #     def __call__(self, img):
    #         return self.data_transform(img)
    #######################################
    # 画像データ変換を簡単に関数として定義
    #######################################
    # transformer = transforms.Compose([  # 画像データ変換モジュール torchvision.transforms.Compose 使用
    #     transforms.ToTensor(),          # ToTensor テンソルに変換
    #     transforms.CenterCrop(224),     # 227 -> 224 にサイズ縮小 CenterCrop(224)  (224, 224ではなく、224と、１回だけ書く)
    # ])                                  # （サイズは画像生成時に指定、全画像再生成に４，５時間必要）

    transformer = transforms.Compose([  # Pytorch 画像データ変換モジュール torchvision.transforms.Compose 使用
        transforms.ToTensor(),          # ToTensor テンソルに変換
    #    transforms.Resize(224),         # 227 -> 224 にサイズ変換 "Resize(224,224)"ではなく224は１回だけ書くかまたは、"Resize((224,224))"と書く)
    ])                                  # （サイズは画像生成時に指定、全画像再生成に４，５時間必要）

    # PILまたはnumpy.ndarray((height x width x channel)で(0~255))
    # を
    # Tensor((channel x height x width)で(0.0~1.0))に変換
    # PILやnumpyでは画像は(height x width x channel)の順番だが、Pytorchでは(channel x height x width)の順に注意
    #
    # ドキュメント
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    #
    # またはディクショナリーで
    # data_transforms = {
    #        'train': transforms.Compose(
    #            [transforms.Resize((256,256)),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225]),
    #             ]),
    #        'valid': transforms.Compose(
    #            [transforms.Resize((256,256)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225]),
    #             ])}
    # 等


    #################################
    # 学習ネットワーク定義 (AlexNet)
    #################################
    # ####################################################################From this part below is no longer necessary.
    # # 全結合の次元を計算し、ネットワークを修正
    # size_check = torch.FloatTensor(10, 3, 224, 224)
    # #size_check = torch.tensor(10, 3, 224, 224)
    # features = nn.Sequential(
    #             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #             nn.ReLU(inplace=True),
    #             nn.MaxPool2d(kernel_size=2, stride=2),
    #             nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #             nn.ReLU(inplace=True),
    #             nn.MaxPool2d(kernel_size=2, stride=2),
    #             nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #             nn.ReLU(inplace=True),
    #             nn.MaxPool2d(kernel_size=2, stride=2),
    #         )
    # #バッチサイズ10, 6×6のフィルターが256枚
    # #10バッチは残して、6×6×256を１次元に落とす=>6×6×256=9216
    # print(features(size_check).size())
    # #バッチ１０の値を軸にして残りの次元を１次元へ落とした場合のTensorの形状をチェックすると9216。
    # print(features(size_check).view(size_check.size(0), -1).size())

    # #fc_sizeを全結合の形状として保持しておく
    # fc_size = features(size_check).view(size_check.size(0), -1).size()[1]
    # print("fc_size = ", fc_size)
    # ##############################################################################Up to here is no longer necessary.
    # 計算結果
    #fc_size =  9216 

    #############################################################################
    # 全結合の入力次元変更後のネットワーク (fc_size = 9216 を折込み済み)
    #print("Network Configuration: AlexNet, ネットワーク構成表示")
    num_classes = 3  # Number of Output Classes, 画像分割のクラス数（３分割問題）================================================================================クラス数を３に設定！！
    #############################################################################

    #################################
    # For Alex Net of depth: 8 layers
    #################################
    class AlexNet(nn.Module):  # 引数 num_classes が必要 ======================================================================================================クラス数設定用 num_classes
        #fc_sizeを引き渡す
        def __init__(self, num_classes):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            #fc_sizeで計算した形状を指定
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(9216, 4096), 
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    #
    #end class

    ##############################################
    # For Residual Net (ResNet) of depth 18 layers
    ##############################################
    class block(nn.Module):
        def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
            """
            残差ブロックを作成するクラス
            Args:
                first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
                first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
                identity_conv : channel数調整用のconv層
                stride : 3×3conv層におけるstide数。sizeを半分にしたいときは2に設定
            """        
            super(block, self).__init__()

            # 1番目のconv層（1×1）
            self.conv1 = nn.Conv2d(
                first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
            self.bn1 = nn.BatchNorm2d(first_conv_out_channels)

            # 2番目のconv層（3×3）
            # パターン3の時はsizeを変更できるようにstrideは可変
            self.conv2 = nn.Conv2d(
                first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn2 = nn.BatchNorm2d(first_conv_out_channels)

            # 3番目のconv層（1×1）
            # output channelはinput channelの4倍になる
            self.conv3 = nn.Conv2d(
                first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(first_conv_out_channels*4)
            self.relu = nn.ReLU()

            # identityのchannel数の調整が必要な場合はconv層（1×1）を用意、不要な場合はNone
            self.identity_conv = identity_conv

        def forward(self, x):

            identity = x.clone()  # 入力を保持する

            x = self.conv1(x)  # 1×1の畳み込み
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)  # 3×3の畳み込み（パターン3の時はstrideが2になるため、ここでsizeが半分になる）
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)  # 1×1の畳み込み
            x = self.bn3(x)

            # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから足す
            if self.identity_conv is not None:
                identity = self.identity_conv(identity)
            x += identity

            x = self.relu(x)

            return x
    #
    #end class

    class ResNet18(nn.Module):
        def __init__(self, block, num_classes):   # ======================================================================================================クラス数設定用 num_classes
            super(ResNet18, self).__init__()

            # conv1はアーキテクチャ通りにベタ打ち
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # conv2_xはサイズの変更は不要のため、strideは1
            self.conv2_x = self._make_layer(block, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=2)

            # conv3_x以降はサイズの変更をする必要があるため、strideは2
            self.conv3_x = self._make_layer(block, 4, res_block_in_channels=256,  first_conv_out_channels=128, stride=4)
            #self.conv4_x = self._make_layer(block, 6, res_block_in_channels=512,  first_conv_out_channels=256, stride=2)
            #self.conv5_x = self._make_layer(block, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512, num_classes)

        def forward(self,x):

            x = self.conv1(x)   # in:(3,224*224)、out:(64,112*112)
            x = self.bn1(x)     # in:(64,112*112)、out:(64,112*112)
            x = self.relu(x)    # in:(64,112*112)、out:(64,112*112)
            x = self.maxpool(x) # in:(64,112*112)、out:(64,56*56)

            x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)
            x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)
            #x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)
            #x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)

            return x

        def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
            layers = []

            # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
            # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
            identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
            layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

            # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
            in_channels = first_conv_out_channels*4

            # channel調整、size調整は発生しないため、identity_convはNone、strideは1
            for i in range(num_res_blocks - 1):
                layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

            return nn.Sequential(*layers)        
    #
    #end class (ResNet18)


    class ResNet50(nn.Module):
        def __init__(self, block, num_classes):   # ======================================================================================================クラス数設定用 num_classes
            super(ResNet50, self).__init__()

            # conv1はアーキテクチャ通りにベタ打ち
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # conv2_xはサイズの変更は不要のため、strideは1
            self.conv2_x = self._make_layer(block, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

            # conv3_x以降はサイズの変更をする必要があるため、strideは2
            self.conv3_x = self._make_layer(block, 4, res_block_in_channels=256,  first_conv_out_channels=128, stride=2)
            self.conv4_x = self._make_layer(block, 6, res_block_in_channels=512,  first_conv_out_channels=256, stride=2)
            self.conv5_x = self._make_layer(block, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*4, num_classes)

        def forward(self,x):

            x = self.conv1(x)   # in:(3,224*224)、out:(64,112*112)
            x = self.bn1(x)     # in:(64,112*112)、out:(64,112*112)
            x = self.relu(x)    # in:(64,112*112)、out:(64,112*112)
            x = self.maxpool(x) # in:(64,112*112)、out:(64,56*56)

            x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)
            x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)
            x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)
            x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)

            return x

        def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
            layers = []

            # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
            # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
            identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
            layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

            # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
            in_channels = first_conv_out_channels*4

            # channel調整、size調整は発生しないため、identity_convはNone、strideは1
            for i in range(num_res_blocks - 1):
                layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

            return nn.Sequential(*layers)
    #
    #end class (ResNet50)


    ###############################
    # Define Device デバイスの定義（冒頭で設定）
    ###############################
    # CUDAドライバーを正常にインストールできていればGPU利用
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 冒頭で設定
    #device = 'cpu'  # 学習にはCPUを利用（cuda がインストールされていないとき）

    #####################################################
    # Training and Validation Process 学習・検証プロセス
    #####################################################
    if perform_train_and_varid == 1:  # 学習、検証をする(=1)またはしない(=0)

        ###########################
        # 画像データ取込みと前処理
        ###########################
        # ImageFolderを使用すれば、フォルダ構成を仕様に合わせれば自動的にラベルと画像データを読み込む事が可能
        # フォルダ構成：
        # フォルダ名＝クラス名（クラス名のフォルダの中に画像を格納すればよい）
        # 例
        # イメージ画像ファイルを格納したディレクトリと画像変換設定を与えるだけ
        # 以下でイメージ画像の取り込み
        # images = torchvision.datasets.ImageFolder(root = "./MNIST_PNG", transform = ImageTransform()) #クラス使用の場合、ラベル付けは格納ディレクトリ名でできている
        ############################################################################
        # 画像データを ImageFolder を使って取込み（Matlab の ImageDataStore に相当）
        ############################################################################
        print("SpectrogramTrain_Path = ",SpectrogramTrain_Path)  # 学習・検証用画像データを格納したパスの確認
        #allImages = torchvision.datasets.ImageFolder(root = "../Spectrogram_Train_Toyama_NE_5_Niigata_82p3MHz", transform = ImageTransform())  # ImageFolder 使用
        #allImages = torchvision.datasets.ImageFolder(root = SpectrogramTrain_Path, transform = ImageTransform())  # ImageTransformクラスは上で定義
        allImages = torchvision.datasets.ImageFolder(root = SpectrogramTrain_Path, transform = transformer)  # すでに定義した関数 transformer を使用

        # または
        # data_dir = 'hymenoptera_data'
        #
        # train_dataset = torchvision.datasets.ImageFolder(root = os.path.join(data_dir, 'train'), transform = data_transforms['train']) # ディクショナリー使用transform
        # valid_dataset = torchvision.datasets.ImageFolder(root = os.path.join(data_dir, 'valid'), transform = data_transforms['valid'])
        #
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True,  num_workers=4)
        # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=5, shuffle=False, num_workers=4)
        #
        # for train in train_loader:
        #     print(train[0].shape)
        #     print(train[0].dtype)
        #     break


        ###############################################
        # 画像データを Train と Valid に一定比率で分割
        ###############################################
        # 層化分割(Stratified Split) https://qiita.com/sin9270/items/c6023453e00bfead9e9f
        # 機械学習をしていると、データセットを学習用データとバリデーション用データに分割することがよくあります。
        # 特に分類問題の場合、クラスラベルを考慮せずランダムに分割してもいいのですが、
        # 分割後のデータのクラスラベルの分布が元データと同じになるように分割するのが望ましいです。
        # このように各クラスの比率を保ったまま分割することを、層化抽出とか層化分割(Stratified Split)と言います。

        # PyTorchでの実装例
        # scikit-learnではsklearn.model_selection.train_test_splitという関数にstratifyオプションを渡すことでStratified Splitを行うことができます。
        # 一方、PyTorchにはそのような仕組みがありません。
        # torch.utils.data.random_splitのような関数を使えばデータセットをランダムに分割することはできますが、
        # ストレートにStratified Splitを行うことはできません。
        # そこで、scikit-learn の train_test_split と組み合わせることで、Stratified Splitを実現します。

        # train_test_splitの第一引数には分割したい配列を渡しますが、Datasetを直接渡すことはできないので、
        # list(range(len(dataset.targets))) で Datasetのインデックス配列([0,1,2,3,...データ数])を生成し、それを代わりに渡しています。
        # そして、このインデックス配列に対するクラスラベル dataset.targets を stratify オプションとして渡すことで、
        # 元データのクラスラベルの比率を保ったまま、インデックス配列を学習用とバリデーション用に分割することができます。

        # データセットをtrainとvalidationに一定比率で分割（インデックス配列として分割、7対3に分割）scikit-learn の train_test_split を使用
        train_indices, valid_indices = train_test_split(list(range(len(allImages.targets))), test_size=0.2, stratify=allImages.targets)
        print("len(train_indices) = ", len(train_indices))
        print("len(valid_indices) = ", len(valid_indices))

        # 分割したのはあくまでインデックス配列なので、そのインデックスを元にデータセットを分割します。
        # Subsetはその名の通りデータのサブセットを作るためのクラスで、元となるDatasetとインデックス配列を渡すことで、
        # インデックスに対応するDatasetを生成できます。
        trainImages = torch.utils.data.Subset(allImages, train_indices)
        validImages = torch.utils.data.Subset(allImages, valid_indices)


        ############################################################
        # データローダーを作成（torch.utils.data.DataLoader を使用）
        ############################################################
        #  ImageFolderで取り込んだイメージ画像のデータを使用してデータローダーを作成
        # データローダーは torch.utils.data.DataLoader に
        # 「ImageFolderで取り込んだイメージ画像データを格納したオブジェクト変数 'allimages' 」、
        # 「1バッチ当たりのデータ数」、「シャッフルするかどうか」等を与えることで作成する

        # １バッチに含む（１度の計算に入力する）画像の枚数：学習時の計算機(CPU or GPU) のメモリ容量により制限
        #batch_size = 64    #
        #batch_size = 128   #
        #batch_size = 256   # Req. GPU Mem >  3 GB
        #batch_size = 512   # Req. GPU Mem >  6 GB
        #batch_size = 1024  # Req. GPU Mem > 12 GB
        #batch_size = 2048   # Req. GPU Mem > 24 GB
        batch_size = batch_size0  # 冒頭で設定

        # Data Loader Parallel Processing データローダの並列処理: num_workers
        # デフォルトでは num_workers=0
        # その結果、ミニバッチの取り出しがSingle processになっています。
        # num_workers=2などに設定することで、multi-process data loadingとなり、処理が高速化されます。
        # CPUのコア数は以下で確認できます。
        # import os
        # os.cpu_count()  # コア数
        # コア数は1GPUに対して2程度？
        #cpu_core = 2  # ０でも２でもほとんど変わらず
        cpu_core  = cpu_core0  # 冒頭で設定
        #cpu_core = 10

        # ImageFolder で取り込んだ画像からデータローダーを作成する
        train_dataloader = torch.utils.data.DataLoader(trainImages, batch_size = batch_size, shuffle = True, num_workers = cpu_core)  # Matlab の imdsTrain に相当
        valid_dataloader = torch.utils.data.DataLoader(validImages, batch_size = batch_size, shuffle = True, num_workers = cpu_core)  # Matlab の imdsValid に相当


        #######
        # 確認
        #######
        # バッチ化されたデータローダーからイメージ画像を取り出し確認
        # データローダから、画像とラベルのテンソルを取り出す
        # for imgs, labels in train_dataloader
        #     print("batch image size: {}".format(imgs.size))
        #     print("image size: {}".format(imgs[0].size()))
        #     print("batch labels size: {}".format(labels.size()))
        #     break  # １回だけforループ利用

        # # 学習用データを確認（イテレーター使用：事前確認のみ、本番では使用しない）
        # imgs, labels = iter(train_dataloader).next()
        # # バッチから取り出した画像のサイズを確認（バッチの画像データはすでにシャッフルされている）
        # print("batch_size = ", batch_size)
        # print("imgs.shape = ",imgs.shape)
        # print("imgs[0].shape = ",imgs[0].shape)
        # print("imgs[batch_size-1].shape = ",imgs[batch_size-1].shape)

        # #バッチから取り出した画像のイメージとラベルを表示 (Label: Abnrml = 0, Normal = 1)
        # pic = transforms.ToPILImage(mode='RGB')(imgs[0])
        # plt.imshow(pic)
        # plt.show()
        # print("labels[0] = ",labels[0].numpy())

        # pic = transforms.ToPILImage(mode='RGB')(imgs[batch_size-1])
        # plt.imshow(pic)
        # plt.show()
        # print("labels[batch_size-1] = ",labels[batch_size-1].numpy())

        # # 検証用データを確認（イテレーター使用：事前確認のみ、本番では使用しない
        # imgs, labels = iter(valid_dataloader).next()
        # # バッチから取り出した画像のサイズを確認（バッチの画像データはすでにシャッフルされている）
        # print("batch_size = ", batch_size)
        # print("imgs.shape = ",imgs.shape)
        # print("imgs[0].shape = ",imgs[0].shape)
        # print("imgs[batch_size-1].shape = ",imgs[batch_size-1].shape)

        # #バッチから取り出した画像のイメージとラベルを表示 (Label: Abnrml = 0, Normal = 1)
        # pic = transforms.ToPILImage(mode='RGB')(imgs[0])
        # plt.imshow(pic)
        # plt.show()
        # print("labels[0] = ",labels[0].numpy())

        # pic = transforms.ToPILImage(mode='RGB')(imgs[batch_size-1])
        # plt.imshow(pic)
        # plt.show()
        # print("labels[batch_size-1] = ",labels[batch_size-1].numpy())


        ###############################
        # Training Conditions 学習条件
        ###############################

        # #######################################################################################################################################################################################
        # # Torch.Hub ネットワーク取得チェック方法
        # #######################################################################################################################################################################################
        # import torch       
        # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                                      # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
        # model = torch.hub.load('rwightman/pytorch-image-models:master', 'resnet18', pretrained=True)  # Load ResNet18 from Torch.Hub
        # print(model)
        # #######################################################################################################################################################################################

        ##################################################################################
        # Select Network Model ネットワークモデルの選択とインスタンス生成（学習プロセス用）
        ##################################################################################
        if networkmodel == 1:                                # AlexNetによるインスタンス生成（参考書を見てこのプログラム内で構築したネットワーク：動作OK）================================================クラス数設定用 num_classes 指定済み
            print("Network Configuration: AlexNet ネットワーク構成")
            model = AlexNet(num_classes).to(device)
        elif networkmodel == 2:                              # ResNet18によるインスタンス生成（参考書を見てこのプログラム内で構築したネットワーク：動作OK）===============================================クラス数設定用 num_classes 指定済み
            print("Network Configuration: ResNet18 (local) ネットワーク構成")
            model = ResNet18(block, num_classes).to(device)
        elif networkmodel == 3:                              # ResNet34によるインスタンス生成（Pytorchから都度ダウンロードする学習済みモデル）
            print("Network Configuration: ResNet34 (torch.hub) ネットワーク構成")
            if device == 'cuda': 
                torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                        # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34',  pretrained=True)  # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
                #model.cuda()                                                                   # GPU(cuda)で使用
                model = model.to(device)                                                        # 自動
                # model.fc.out_features = 2                                                     # 最終層の出力数を２に変更（正常信号と異常信号に分類するため2つのクラスを生成）
                model.fc.out_features = 3                                                       # 最終層の出力数を３に変更（正常信号と異常信号（強、弱）に分類するため3つのクラスを生成）
            else:
                print("Analysis is stopped. Device is not GPU/cuda; specified network model is not applicable in this environment. プログラムを停止します。指定されたネットワークモデルはGPUのない環境では使用できません。")
                sys.exit()
            #end if
        elif networkmodel == 4:                              # ResNet50によるインスタンス生成（参考書を見てこのプログラム内で構築したネットワーク：動作OK）===============================================クラス数設定用 num_classes 指定済み
            print("Network Configuration: ResNet50 (local) ネットワーク構成")
            model = ResNet50(block, num_classes).to(device)
        elif networkmodel == 5:                              # ResNet50によるインスタンス生成（Pytorchから都度ダウンロードする学習済みモデル）
            print("Network Configuration: ResNet50 (torch.hub) ネットワーク構成")
            if device == 'cuda':
                torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                        # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)   # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
                #model.cuda()                                                                   # GPU(cuda)で使用
                model = model.to(device)                                                        # 自動
                # model.fc.out_features = 2                                                     # 最終層の出力数を２に変更（正常信号と異常信号に分類するため2つのクラスを生成）
                model.fc.out_features = 3                                                       # 最終層の出力数を３に変更（正常信号と異常信号（強、弱）に分類するため3つのクラスを生成）
            else:
                print("Analysis is stopped. Device is not GPU/cuda; specified network model is not applicable in this environment. プログラムを停止します。指定されたネットワークモデルはGPUのない環境では使用できません。")
                sys.exit()
            #end if
        elif networkmodel == 6:                              # ResNet152によるインスタンス生成（Pytorchから都度ダウンロードする学習済みモデル）
            print("Network Configuration: ResNet152 (torch.hub) ネットワーク構成")
            
            if device == 'cuda':
                torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                        # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)  # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
                #model.cuda()                                                                   # GPU(cuda)で使用
                model = model.to(device)                                                        # 自動
                # model.fc.out_features = 2                                                     # 最終層の出力数を２に変更（正常信号と異常信号に分類するため2つのクラスを生成）
                model.fc.out_features = 3                                                       # 最終層の出力数を３に変更（正常信号と異常信号（強、弱）に分類するため3つのクラスを生成）
            else:
                print("Analysis is stopped. Device is not GPU/cuda; specified network model is not applicable in this environment. プログラムを停止します。指定されたネットワークモデルはGPUのない環境では使用できません。")
                sys.exit()
            #end if
        # end if
        #
        # Print the Whole Network Structure 全体のネットワーク構造を表示
        print("Network Configuration: ネットワーク構成の確認") 
        print(model)
        #
        print("Network Output Layer: ネットワーク最終層の確認") 
        print(model.fc)



        ##############################
        # Training Method 学習方法設定
        ##############################
        # Learning Rate 学習率
        #LearningRate = 0.005  # 0.01 # 0.005  #  0.01, 0.002 でも同じ傾向 (Validation精度最大約86% @ epoch=130~140) 
        LearningRate = LearningRate0  # 冒頭で設定
        print("Learning Rate 学習率 = {}".format(LearningRate))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LearningRate, momentum=0.9, weight_decay=5.0e-4)  # AlexNet, ResNet18
        #optimizer = optim.SGD(model.parameters(), lr=LearningRate, momentum=0.9, weight_decay=1.0e-3)

        ##############################################
        # AlexNet / ResNet18 Network 構造をテスト表示
        ##############################################
        if networkmodel == 1:
            print("Network Configuration: AlexNet ネットワーク構成")
        elif networkmodel == 2:
            print("Network Configuration: ResNet18 (local) ネットワーク構成")
        elif networkmodel == 3:
            print("Network Configuration: ResNet34 (torch.hub) ネットワーク構成")
        elif networkmodel == 4:
            print("Network Configuration: ResNet50 (local) ネットワーク構成")
        elif networkmodel == 5:
            print("Network Configuration: ResNet50 (torch.hub) ネットワーク構成")
        elif networkmodel == 6:
            print("Network Configuration: ResNet152 (torch.hub) ネットワーク構成")
        # end if
        print(model)


        ####################################
        ####################################
        # Training Iteration 学習の繰り返し
        ####################################
        ####################################
        #if perform_train_and_varid == 1:  # 学習、検証をする(=1)またはしない(=0)

        # Number of Epochs エポック数（１エポックは解析がすべての画像データについて一巡すること）
        # 学習データが良好な場合、LearningRate = 0.01,  num_epochs = 30  程度で収束、valid_acc = 0.992 程度に到達
        #num_epochs = 1000
        num_epochs = num_epochs0  # エポック数（冒頭で設定）
        print("Traning Process Started 学習過程開始 (Max Epock = {}) ...".format(num_epochs))

        train_loss_list = []  # Loss
        train_acc_list  = []  # Accuracy
        valid_loss_list = []  # Loss
        valid_acc_list  = []  # Accuracy

        for epoch in range(num_epochs):
            train_loss = 0
            train_acc  = 0
            valid_loss = 0
            valid_acc  = 0
            
            #train 学習
            #net.train()
            model.train()
            for i, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                #outputs = net(images)   ###ここで RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED（cuda が正しくインストールされていないとき）
                outputs = model(images)  ###ここで RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED（cuda が正しくインストールされていないとき）
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                train_acc += (outputs.max(1)[1] == labels).sum().item()
                loss.backward()
                optimizer.step()
            #end for
            avg_train_loss = train_loss / len(train_dataloader.dataset)
            avg_train_acc  = train_acc  / len(train_dataloader.dataset)
            
            #validate 検証
            #net.eval()
            model.eval()
            with torch.no_grad():
                for images, labels in valid_dataloader:
                    images  = images.to(device)
                    labels  = labels.to(device)
                    #outputs = net(images)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_acc  += (outputs.max(1)[1] == labels).sum().item()
                #end for
            #end with
            avg_valid_loss  = valid_loss / len(valid_dataloader.dataset)
            avg_valid_acc   = valid_acc  / len(valid_dataloader.dataset)
            
            print("Epoch [{}/{}], Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Valid_Loss: {valid_loss:.4f}, Valid_Acc: {valid_acc:.4f}".format(
                   epoch + 1, num_epochs,      train_loss=avg_train_loss,   train_acc=avg_train_acc,     valid_loss=avg_valid_loss,   valid_acc=avg_valid_acc))

            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)
            valid_loss_list.append(avg_valid_loss)
            valid_acc_list.append(avg_valid_acc)
        #
        # end for epoch

        #################
        # 転移学習（まだ）
        #################
        # #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'  # 学習にCPUを利用（cuda がインストールされていないため）
        # #学習済みAlexNetを取得
        # net = models.alexnet(pretrained=True)
        # net = net.to(device)
        # print(net)  # AlexNetの構造を表示
        # パラメータを更新しない設定
        # for params in net.parameters():
        #     params.requires_grad = False
        # net = net.to(device)
        # #最終層の出力数を２に設定、２クラス分類用にする
        # num_features = net.classifier[6].in_features  # 出力層の入力サイズ
        # num_classes = 2
        # net.classifier[6] = nn.Linear(num_features, num_classes).to(device)
        # print(net)
        # # Training Method 学習方法（損失関数、最適化関数）の設定
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # #以上の設定で学習開始
        # .....(PyTorchプログラミング入門 p.113~)

        ###############################
        # 学習経過プロットの表示と保存
        ###############################

        fig = plt.figure()
        plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-',  label='train_loss')
        plt.plot(range(num_epochs), valid_loss_list, color='red',  linestyle='--', label='valid_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss')
        plt.grid()
        fig.savefig(os.path.join(Miscellaneous_DirectoryPath,"TrainResultsLoss.png"), bbox_inches="tight")
        plt.ion() # Plot Interactive ON (in order not to be blocked by the figure window)
        plt.show()
        #plt.show(block=False)  # No use
        plt.pause(0.001)
        #input("Press Enter to continue.")

        fig = plt.figure()
        plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-',  label='train_acc')
        plt.plot(range(num_epochs), valid_acc_list, color='red',  linestyle='--', label='valid_acc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Training and validation accuracy')
        plt.grid()
        fig.savefig(os.path.join(Miscellaneous_DirectoryPath,"TrainResultsAccuracy.png"), bbox_inches="tight")
        plt.ion() # Plot Interactive ON (in order not to be blocked by the figure window)
        plt.show()
        #plt.show(block=False)  # No use
        plt.pause(0.001)
        #input("Press Enter to continue.")


        ################################################################
        # 学習結果のネットワークモデルの保存 (CPU, GPU-CUDA 両方に対応)
        ################################################################
        if device == 'cpu':                                                               # CPUで学習した場合
            model_path = os.path.join(Miscellaneous_DirectoryPath,'TrainedNet_cpu.mdl')   # CPUのmapで保存
            torch.save(model.to('cpu').state_dict(), model_path)
            #torch.save(model.state_dict(), model_path)                                   # こう書くべきか?
            print("Trained Network Model has been stored in 'TrainedNet_cpu.mdl' file. 書き込み完了")
        elif device == 'cuda':                                                            # GPU(cuda)で学習した場合
            model_path = os.path.join(Miscellaneous_DirectoryPath,'TrainedNet_cpu.mdl')   # CPUのmapで保存し、さらに
            torch.save(model.to('cpu').state_dict(), model_path)
            print("Trained Network Model has been stored in 'TrainedNet_cpu.mdl' file. 書き込み完了")
            model_path = os.path.join(Miscellaneous_DirectoryPath,'TrainedNet_cuda.mdl')  # GPU(cuda)のmapでそのまま保存
            torch.save(model.state_dict(), model_path)
            print("Trained Network Model has been stored in 'TrainedNet_cuda.mdl' file. 書き込み完了")
        # end if
        
        ####################################
        # Training Loop Done 学習ループ終了
        ####################################

        ##############################################################################
        # Valid (Test) 検証
        # 検証データを判別分析(classify)して学習精度を検証(Validation Process, Valid)
        ##############################################################################
        # すでに学習と同時進行で検証済み
    #
    # end if  # 学習・検証をする(=1)またはしない(=0)


    #elif perform_train_and_varid == 0:  # 学習、検証をしない場合(=0); 学習結果はファイルから読み込む

    # 学習してもしていなくても以下の部分は判定のために必要
    ####################################################################################################
    # 学習結果の読み込み（もし学習結果のネットワークモデルファイル 'TrainedNet_*.mdl' があれば読み込む）
    ####################################################################################################

    ##################################################################################
    # Select Network Model ネットワークモデルの選択とインスタンス生成（判別プロセス用）
    ##################################################################################
    if networkmodel == 1:
        model = AlexNet(num_classes).to(device)          # AlexNetによるインスタンス生成（参考書を見てこのプログラム内で構築したネットワーク：動作OK）
    elif networkmodel == 2:
        model = ResNet18(block, num_classes).to(device)  # ResNet18によるインスタンス生成（参考書を見てこのプログラム内で構築したネットワーク：動作OK）
    elif networkmodel == 3:                              # ResNet34によるインスタンス生成（Pytorchにある学習済みモデル）
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                          # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34',  pretrained=True)  # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
        # model.cuda()                                                                    # GPU(cuda)で使用
        # model.fc.out_features = 2                                                       # 最終層の出力数を２に変更（正常信号と異常信号に分類するため2つのクラスを生成）
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).to(device)  # 2022/03 from Kaido's code
        model.fc.out_features = 3                                                                 # 最終層の出力数を３に変更（正常信号と異常信号（強、弱）に分類するため3つのクラスを生成）
    elif networkmodel == 4:
        model = ResNet50(block, num_classes).to(device)  # ResNet50によるインスタンス生成（参考書を見てこのプログラム内で構築したネットワーク：動作OK）
    elif networkmodel == 5:                              # ResNet50によるインスタンス生成（Pytorchにある学習済みモデル）
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                          # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)   # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
        # #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True) # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
        # model.cuda()                                                                    # GPU(cuda)で使用
        # model.fc.out_features = 2                                                       # 最終層の出力数を２に変更（正常信号と異常信号に分類するため2つのクラスを生成）
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)  # 2022/03 from Kaido's code
        model.fc.out_features = 3                                                                 # 最終層の出力数を３に変更（正常信号と異常信号（強、弱）に分類するため3つのクラスを生成）
    elif networkmodel == 6:                              # ResNet152によるインスタンス生成（Pytorchにある学習済みモデル）
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True                          # Workaorund for Pytorch 1.9 Bug (#56138), need to be placed before any "torch.hub" call
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)  # 事前学習済みモデルをPytorchから持ってくる: GitHubアクセス超過エラー(Bug): urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
        # model.cuda()                                                                    # GPU(cuda)で使用
        # model.fc.out_features = 2                                                       # 最終層の出力数を２に変更（正常信号と異常信号に分類するため2つのクラスを生成）
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True).to(device) # 2022/03 from Kaido's code
        model.fc.out_features = 3                                                                 # 最終層の出力数を３に変更（正常信号と異常信号（強、弱）に分類するため3つのクラスを生成）
    # end if

    if device == 'cpu':                                       # CPUで読み込む場合

        model_path = os.path.join(Miscellaneous_DirectoryPath,'TrainedNet_cpu.mdl')  # CPUのmapで読み込み
        if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("Device = CPU; Trained Network Model has been read from 'Miscellaneous_.../TrainedNet_cpu.mdl' file. 　ネットワークモデルファイルを読込み完了")
        else:
            print("Device = CPU; Trained Network Model File 'Miscellaneous_.../TrainedNet_cpu.mdl' does NOT exist. 適合するネットワークモデルファイルがありません")
        #end if

    elif device == 'cuda':                                    # GPU(cuda)で読み込む場合

        #model_path = 'TrainedNet_cpu.mdl'                    # CPUのmapで読み込むか、または
        #if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
        #    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #    print("Device = CUDA; Trained Network Model has been read from 'TrainedNet_cpu.mdl' file. 　ネットワークモデルファイルを読込み完了")
        #else:
        #    print("Device = CUDA; Trained Network Model File 'TrainedNet_cpu.mdl' does NOT exist. 適合するネットワークモデルファイルがありません")
        ##end if

        model_path = os.path.join(Miscellaneous_DirectoryPath,'TrainedNet_cuda.mdl') # GPU(cuda)のmapで読み込む
        if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
            model.load_state_dict(torch.load(model_path))
            print("Device = CUDA; Trained Network Model has been read from 'Miscellaneous_.../TrainedNet_cuda.mdl' file. ネットワークモデルファイルを読込み完了")
        else:
            print("Device = CUDA; Trained Network Model File 'Miscellaneous_.../TrainedNet_cuda.mdl' does NOT exist. 適合するネットワークモデルファイルがありません")
        #end if
    #
    # end if
    #

    ###########################################
    ###########################################
    # Train 学習（訓練）& Validation 検証 完了
    ###########################################
    ###########################################

    #input("Training Process Ended 学習プロセス終了, Check Trained Network Data for 3 Classes ３クラス用のネットワークデータ Miscellaneous_.../TrainedNet_cuda.mdl 等をチェック; Press Enter to Continue for Judging Process 判定プロセス開始するには Enter を押す ...")




    ########################
    ########################
    # Judge 判定（判別分析）
    ########################
    ########################
    # 学習済みネットワークを用いた分類予測（判別分析:'classify'）(Judgement Process, Judge)
    # trainedNet を応用し、ラジオ波観測データを評価時間区間毎に分割して判別
    #
    # 注意：
    # 分類予測データの時刻に注意（最新時刻からさかのぼる、１日４回、６時間毎：０時、６時、１２時、１８時、または毎時、１時間毎に分析）
    # 異常である確率を表示する: [Ypredict, probabilities] = classify(trainedNet,imdsValid); の probabilities を表示
    #
    #####################
    # 判別分析期間の指定
    #####################

    # 分析開始日（マニュアル指定）=> unixtimeに変換 => 日_時刻に変換 
    #JudgePeriod_StartingDate = "20200401" # 判別分析期間の開始日（= 2020年x月x日０時０分０秒）をストリング配列(Matlab)またはstr型文字列(Python)で指定
    JudgePeriod_StartingDate     = JudgePeriod_StartingDate0  # 判別分析開始年月日 = 学習終了年月日（冒頭で設定）
    # Attention: datestr関数, datenum関数では 年yyyy 月mm 日dd 時HH 分MM 秒SS (Matlab)
    #tmp_time_corrected = datestr(datenum(JudgePeriod_StartingDate,'yyyymmdd'),'yyyymmdd_HHMMSS')                # Matlab: シリアル日付を使用して規格外時間の修正（24時を翌日00時に変換）
    year = int(JudgePeriod_StartingDate[0:4])  # string型を整数型に変換、0始まり〜1多い数まで指定
    mon  = int(JudgePeriod_StartingDate[4:6])  # string型を整数型に変換、0始まり〜1多い数まで指定
    day  = int(JudgePeriod_StartingDate[6:8])  # string型を整数型に変換、0始まり〜1多い数まで指定
    dt   = datetime(year, mon, day, 0, 0, 0)   # Python (JST) （注意: datetime関数は"24"時を扱えない） timezone はデフォルトで local = Asia/Tokyo が使用される
    # Attention: datetime関数では 年yyyy 月MM 日dd 時HH 分mm 秒ss (Matlab)
    #datetime_string          = datetime(tmp_time_corrected,'InputFormat','yyyyMMdd_HHmmss','TimeZone','local')  # Matlab: JST：入力時刻のTimeZoneは現地'local'='Asia/Tokyo'
    #JudgePeriod_StartingTime = posixtime(datetime_string)                                                       # Matlab: 判別分析期間の開始時刻 unixtime (UTC)
    JudgePeriod_StartingTime  = int(dt.timestamp())                                                              # Python: 判別分析期間の開始時刻 unixtime (UTC), timestamp() では 9*3600 引く必要なし（注意: timestamp()はfloat型, Python3のint型は無制限）
    #tmp_datetime             = datetime(JudgePeriod_StartingTime,'ConvertFrom','posixtime','TimeZone','local')  # Matlab: unixtimeからdatetime値(JST)へ変換：入力時刻のTimeZoneは現地'local'='Asia/Tokyo'
    #JudgePeriod_StartingDateTime = datestr(tmp_datetime,'yyyymmdd_HHMMSS')                                      # Matlab: datestr関数でdatetime変数を文字へ変換:文字識別子yyyymmdd_HHMMSSに注意！

    # 分析終了日（自動指定、読み込み済みデータファイルの最新日時とする：unixtime指定）=> unixtimeに変換 => 日_時刻に変換　←　終了日はスペクトログラムデータの最後の時刻とするべき
    spectrogram_maxsize          = len(spectrogram[:,0])
    Spectrogram_TimeEnd          = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[spectrogram_maxsize-1]])  # Ending   Time (unixtime) of "spectrogram" スペクトログラムの最後の時刻
    print("Spectrogram_TimeEnd   = ", Spectrogram_TimeEnd)
    #JudgePeriod_EndingTime      = RadiowaveDataExtractedEndTime                                                 # 判別分析期間の終了時刻 unixtime (UTC) = ラジオ波観測時間データの最後の時刻
    JudgePeriod_EndingTime       = Spectrogram_TimeEnd                                                           # 判別分析期間の終了時刻 unixtime (UTC) = スペクトログラムデータの最後の時刻
    #tmp_datetime             = datetime(RadiowaveDataExtractedEndTime,'ConvertFrom','posixtime','TimeZone','local') # Matlab: unixtimeからdatetime値(JST)へ変換：入力時刻のTimeZoneは現地'local'='Asia/Tokyo'
    #JudgePeriod_EndingDateTime = datestr(tmp_datetime,'yyyymmdd_HHMMSS')                                        # Matlab: datestr関数でdatetime変数を文字へ変換:文字識別子yyyymmdd_HHMMSSに注意！

    ###############################
    # 判別分析するまたはしない選択
    ###############################
    if perform_judge == 1: # 判別分析をする(=1)またはしない(=0)
        print('Start Performing Judgement Analysis...: 判別分析開始')

        #######################################
        # 判別分析条件（データ区間長など）設定
        #######################################
        # 判別分析の際、データ区間を少しずつずらす時間シフト[s]
        JudgeTimeShift  = 60.0*60.0*6.0  # = 3600*6 [s] 6時間ずつずらして判別分析する###############################################################################################
        # 判別分析データ区間長を学習データ区間長と同じに設定[s]
        JudgeTimeSpan       = NormalAbnrmlTimeSpan        #[s] 判別分析データ区間長（6, 12, 18時間 = 21600, 43200, 64800秒 等）
        JudgeTimeWidthCount = NormalAbnrmlTimeWidthCount  #[回] 判別分析データの区間幅（180, 360, 540回 等）

        ###########################
        # ディレクトリ等削除と作成
        ###########################
        # 判別分析画像データ保存用ディレクトリ設定
        SpectrogramJudge_Directory = "Spectrogram_Judge_" + City_DR_No_BCLoc_Freq
        SpectrogramJudge_Path      = os.path.join(Spectrogram_DeepLearning_Path, SpectrogramJudge_Directory)

        # ディレクトリ内の既存のファイルをディレクトリごとすべて一括削除
        if delete_judge_images_before_gen == 1:
            if os.path.isdir(SpectrogramJudge_Path):
                shutil.rmtree(SpectrogramJudge_Path)
        # end if

        # ディレクトリがなければ作成
        if not os.path.isdir(SpectrogramJudge_Path): 
            os.makedirs(SpectrogramJudge_Path)
        # end if
        

        # 判別分析期間の設定
        judge_shiftmax  = JudgePeriod_EndingTime - JudgePeriod_StartingTime  #期間の長さ（秒）、シフト余剰（最大）時間（秒）
        jjudge_shiftmax = int(judge_shiftmax//JudgeTimeShift)  #データ分析期間内での学習区間の時間シフト割り当て可能回数（シフトできない場合、解析データ不足状態なので割り当てずに打ち切る: '//'は切り捨て除算）
        #jjudge_shiftmax = 128 #######################################################################################################################Debug
        #print("jjudge_shiftmax = ", jjudge_shiftmax)
        
        # 結果の配列準備
        # For 2 classes:
        # AbnormalStateProbability  = np.zeros((2, jjudge_shiftmax))  # [2, jjudge_shiftmax]  # 判別結果が異常状態である確率を保存する配列を定義（２行）：unixtime(UTC) vs. 異常状態である確率, デフォルトは横長の行列 (２行 x jjudge_shiftmax列)
        # AbnormalStateJudgeResults = np.zeros((2, jjudge_shiftmax), dtype = int)  # [2, jjudge_shiftmax]
        # For 3 classes:
        AbnormalStateProbability  = np.zeros((4, jjudge_shiftmax))  # [4, jjudge_shiftmax]  # 判別結果が異常状態である確率を保存する配列を定義（４行）：0:unixtime(UTC) vs. 1:異常大、2:異常小、3:正常、のそれぞれの状態である確率, デフォルトは横長の行列 (４行 x jjudge_shiftmax列)
        AbnormalStateJudgeResults = np.zeros((2, jjudge_shiftmax), dtype = int)  # [2, jjudge_shiftmax] 1:unixtime, 2:判別結果のid,（class_id_reverse, AbnLrg=2, AbnSml=1, Normal=0）


        # 判別分析のための画像データを新たに作成する、または作成しない
        #if generate_judge_images == 1: # 新たな画像データの作成: 作成する=1, 作成しない=0(1以外) ################################################################################################################### Debug Needed!
        if 1:  # 常に作成する（画像１枚ずつ生成する度に判別しているため：ただし、遅い）（画像をまとめて判別するように変更すればより高速）
            print('Start Generating Images for the Judgement of RadiowaveData: ラジオ波観測データの判別分析用画像作成開始')

            # 判別分析用の画像データを生成してディレクトリ ScalogramClassify_Directory に保存
            if ((judge_shiftmax > -0.001) and (jjudge_shiftmax >= 1)): #シフト余剰（最大）が正、割り当て可能回数が１回以上ならば
                
                class_id_reverse_list = [] # クラス予測ラベル用のリスト初期化: (元のclass_id: AnbLrg=0, AbnSml=1, Normal=2 ===>> 反転後のclass_id_reverse: AbnLrg=2, AbnSml=1, Normal=0)

                # 判別分析期間スイープ
                count = 0  # スペクトログラム数のカウンター初期化
                for j in range(jjudge_shiftmax):  # j=0:jjudge_shiftmax-1

                    count = count + 1  # スペクトログラム数のカウンター加算
                    if (int(count/500)*500 == count):
                        print("Number of Spectrograms for Judge 判別解析用スペクトログラム数 = {} / {}".format(count, jjudge_shiftmax))
                    # end if

                    ##########################################################################################################
                    # 判別分析区間を設定（評価時刻から判別分析データ時間さかのぼって分析し、それをその時刻の判別結果に入れる）
                    ##########################################################################################################
                    #JudgeTime              = JudgePeriod_StartingTime + (j-1)*JudgeTimeShift  #[s] 判別分析評価時刻を判別分析区間の開始日（= 2020年１月１日０時０分０秒）に設定
                    JudgeTime               = JudgePeriod_StartingTime + j*JudgeTimeShift  #[s] 判別分析評価時刻; 判別分析区間の開始日（= 2020年１月１日０時０分０秒）から更新; Python: jは0始まり
                    #print("JudgeTime [s] =", JudgeTime )
                    JudgeTimeSpan_StartTime = JudgeTime - JudgeTimeSpan                    #[s] 判別分析データ区間の開始時刻（判別分析データ時間さかのぼる）
                    JudgeTimeSpan_EndTime   = JudgeTimeSpan_StartTime + JudgeTimeSpan      #[s] 判別分析データ区間の終了時刻（判別分析評価時刻 JudgeTime と等価）
                    # スペクトログラムの初期時刻
                    spectrogram_time_start  = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])  # スペクトログラムの最初の Unixtime [s] は Extracted Timedata の最初の時刻に等しくない

                    # 判別用 Scalogram image JPGファイル名生成
                    #スタート時刻の文字列生成
                    #filename_datetime      = datetime(JudgeTimeSpan_StartTime,'ConvertFrom','posixtime','TimeZone','local','Format','yyyyMMdd_HHmmss'); #datetime変数のまま:文字識別子yyyyMMdd_HHmmssに注意！
                    #評価時刻の文字列生成
                    #filename_datetime      = datetime(JudgeTime,'ConvertFrom','posixtime','TimeZone','local','Format','yyyyMMdd_HHmmss')  #Matlab: datetime変数のまま:文字識別子yyyyMMdd_HHmmssに注意！
                    #filename_datetime      = "{0:%Y%m%d_%H%M%S}".format(datetime.fromtimestamp(start_uxtime,tz=None)) # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                    filename_datetime       = "{0:%Y%m%d_%H%M%S}".format(datetime.fromtimestamp(JudgeTime,tz=None))    # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                    #ScalogramJudge_Fig     = "scalogram_judge_" + datestr(filename_datetime,'yyyymmdd_HHMMSS') + ".jpg"  #Matlab: datestr関数でdatetime変数を文字へ変換:文字識別子yyyymmdd_HHMMSSに注意！
                    #SpectrogramJudge_Fig   = "scpectrogram_judge_" + filename_datetime + ".png"  #Python: 文字形式yyyymmdd_HHMMSS

                    # Degug 20210516
                    #SpectrogramJudge_Fig   = "scpectrogram_judge_" + filename_datetime + ".jpg"  #Python: 文字形式yyyymmdd_HHMMSS
                    SpectrogramJudge_Fig    = "spectrogram_judge_" + filename_datetime + ".jpg"  #Python: 文字形式yyyymmdd_HHMMSS

                    #ScalogramJudge_FigPath  = fullfile(ScalogramJudge_Path, ScalogramJudge_Fig)  # Matlab
                    SpectrogramJudge_FigPath = os.path.join(SpectrogramJudge_Path, SpectrogramJudge_Fig)  # Python

                    # もし同じ名前のスペクトログラム画像データが存在しなければ新たにスペクトログラム画像データを作成する、存在すれば新たに作成しない
                    #print("os.path.isfile(SpectrogramJudge_FigPath) = ", os.path.isfile(SpectrogramJudge_FigPath))
                    if not os.path.isfile(SpectrogramJudge_FigPath) == 1:
                        #print("Generate Spectrogram '{}' for Judgement".format(SpectrogramJudge_FigPath))
                    
                        # ##############################################################################
                        # # 時間データのインデックス計算（この部分はスペクトログラム解析では使用しない）
                        # ##############################################################################
                        # # 判別分析期間から判別分析時間データ区間分を取り出し（ただし、Matlabでは時間データを取り出し、Pythonではスペクトログラムデータを取り出す）
                        # #start_index = int64(round(JudgeTimeSpan_StartTime - RadiowaveDataExtractedStartTime)/RadiowaveDataExtractedTimeStep)+1  #Matlab: 開始時インデックス（時間データのインデックス）
                        # start_index  = int((JudgeTimeSpan_StartTime - RadiowaveDataExtractedStartTime)//RadiowaveDataExtractedTimeStep)+1        #Python: 開始時インデックス（時間データのインデックス）
                        # #end_index   = int64(start_index + JudgeTimeWidthCount)-1                                                                #Matlab: 終了時インデックス（時間データのインデックス）
                        # end_index    = int(start_index + JudgeTimeWidthCount - 1)                                                                #Python: 終了時インデックス（時間データのインデックス）

                        # # 配列を部分的にコピーして判別分析用データ取得
                        # #RadiowaveDataJudge      = RadiowaveDataExtracted(start_index:end_index,1:2)  #Matlab: (JudgeTimeWidthCount行 x 2列)の配列
                        # RadiowaveDataJudge_utime = RadiowaveDataExtracted_utime[start_index:end_index]  #Python: unixtime [s]          (JudgeTimeWidthCount行 x 1列)の配列
                        # RadiowaveDataJudge_tdata = RadiowaveDataExtracted_tdata[start_index:end_index]  #Python: dB値ではなくPower[mW] (JudgeTimeWidthCount行 x 1列)の配列
                        # #print("RadiowaveDataNormal_tdata = ", RadiowaveDataNormal_tdata)

                        # ##################################################################################################################################
                        # # Generate Scalogram by WT, ウエーブレット変換によりスカログラムデータ生成
                        # signallength = size(RadiowaveDataJudge,1)
                        # # The default wavelet used in the cwt filter bank is the analytic Morse (3,60) wavelet.
                        # #filterbank = cwtfilterbank('SignalLength',signallength,'VoicesPerOctave',12); # フィルター設定 Default: Morse(3,60) Wevelet
                        
                        # filterbank = cwtfilterbank('SignalLength',signallength,...
                        #                            'VoicesPerOctave',30,...
                        #                            'TimeBandWidth',100,...
                        #                            'SamplingFrequency',SamplingFreq,...
                        #                            'FrequencyLimits',[FreqLimitLower FreqLimitUpper])
                        
                        # # Perform Wavelet Transform: ウェーブレット変換
                        # [coeffs,f,coi] = wt(filterbank, RadiowaveDataJudge(:,2)); #Wavelet coefficients, Frequency, Cone of influence: ウェーブレット変換
                        # ##################################################################################################################################

                        #####################################################################################
                        # Spectrogram スペクトログラムにおける判別解析用データの開始時、終了時のインデックス
                        # スペクトログラムの timestep は "spectrogram_timestep"
                        #####################################################################################
                        JudgeSpectrogram_TimeSpan = window_indx_size * timestep  # スペクトログラムの区間幅[sec]は時間窓[sec]と同じに設定する timestep = 120.0 [s]等
                        #print("JudgeSpectrogram_TimeSpan = ", JudgeSpectrogram_TimeSpan)
                        JudgeSpectrogram_IndxSpan = int(JudgeSpectrogram_TimeSpan//spectrogram_timestep)   # スペクトログラムの区間幅数[1]は区間幅[sec]を移動フーリエ変換ステップで割ったもの
                        ##################################
                        # Spectrogram の indices 計算結果
                        ##################################
                        JudgeSpectrogram_IndxStart = int((JudgeTimeSpan_StartTime - spectrogram_time_start)//spectrogram_timestep)  # スペクトログラム中の正常時プロットの開始時点のインデックス 
                        #print("JudgeSpectrogram_IndxStart = ",JudgeSpectrogram_IndxStart)
                        JudgeSpectrogram_IndxEnd   = JudgeSpectrogram_IndxStart + JudgeSpectrogram_IndxSpan                         # スペクトログラム中の正常時プロットの終了時点のインデックス
                        #print("JudgeSpectrogram_IndxEnd   = ",JudgeSpectrogram_IndxEnd)

                        ##################################################################
                        # Extract Spectrogram Data, スペクトログラムデータ取り出し Python3
                        ################################################################## 
                        # Generate 3D plot data: 
                        spectrogram_tmp = spectrogram[JudgeSpectrogram_IndxStart:JudgeSpectrogram_IndxEnd,:]
                        #print("Extract Spectrogram Data for Judgement: spectrogram_tmp = ",spectrogram_tmp)
                        #print("Extract Spectrogram Data for Judgement: spectrogram_tmp.shape = ",spectrogram_tmp.shape)
                        #input("Extract Spectrogram Data for Judgement: Press Enter to Continue...")
                        
                        #############################################
                        # Spectrogram without axes with fixed pixels
                        # 画像の生成（軸なし）
                        #############################################
                        #print("SpectrogramJudge_FigPath = ", SpectrogramJudge_FigPath)
                        fig = plt.figure()
                        #plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), vmin=-280, vmax=-190)  # Plot (default: 'viridis')
                        plt.imshow(np.flipud(np.transpose(spectrogram_tmp)), cmap="jet", vmin=-280, vmax=-190)  # Plot in 'jet'
                        plt.axis("off")
                        #plt.show() # テスト表示#############################################################################################################################
                        #fig.savefig(SpectrogramJudge_FigPath, bbox_inches="tight")  # 保存: Tight bounding box（文字の一部が隠れる問題を回避）白い余白が生じる
                        fig.savefig(SpectrogramJudge_FigPath, transparent=False, bbox_inches="tight", pad_inches = 0)  # 保存: 透明背景なし, 余白なし
                        #print('Radiowave Spectrogram for Judgement Saved ラジオ波スペクトログラム軸なし（判定用）を保存')
                        plt.cla()   # clear axis
                        plt.clf()   # clear figure
                        plt.close()
                        # 画像サイズチェック
                        #img = plt.imread(SpectrogramJudge_FigPath)  # 点検のためイメージ読み込み
                        #print(img.shape)                            # ピクセル数表示                    
                        #########################
                        # Pillowで画像サイズ変換、再保存...効率化可能
                        #########################
                        img = Image.open(SpectrogramJudge_FigPath)  # 画像の読み込み
                        #img_resize = img.resize(size=(227,227), resample=Image.NEAREST)    # リサイズ処理: Matlab AlexNet では 227 X 227 ? For Python 224 x 224
                        img_resize = img.resize(size=(224,224), resample=Image.NEAREST)     # リサイズ処理: Matlab AlexNet では 227 X 227 ? For Python 224 x 224
                        img_resize = img_resize.save(SpectrogramJudge_FigPath, quality=95)  # ファイルに保存############################################################
                        # 画像サイズチェック
                        img = plt.imread(SpectrogramJudge_FigPath)  # イメージ読み込み
                        #print(img.shape)                            # ピクセル数表示
                        #input("Pillowで画像サイズ変換; Press Return to go next")
                        #
                        # ここまでWarningなし
                        #
                        #注意： scalogramdata ---> spectrogramdata 等に変数名を替える
                        # 判別分析用画像データ生成と保存 (Matlab)
                        #im = ind2rgb(im2uint8(rescale(abs(coeffs))),jet(300)); #Attention: use 'abs(coeffs)', 'rescale' rescales, also 'im2uint8' scales to 0-255 level integers; jet(256) better than jet(128)
                        #im = ind2rgb(im2uint8(abs(coeffs)),jet(300))  #No Re-scale                    
                        # プロット保存 (Matlab)
                        #imwrite( imresize(im,[227 227]), ScalogramJudge_FigPath )  #Matlab Alexnet では 227 x 227
                        #pause(pausing_time)
                    #
                    # end if 判別解析用スペクトログラムを新たに生成する場合
                    

                    #############################################################################################################################################
                    # Noguchi's code for Judge （ひとつずつ判別する場合）判別時のネットワーク構造の指定（この部分は学習部の最後で実行済みのためここでは省略可能）
                    #############################################################################################################################################
                    # #print("Judge Spectrogram '{}'".format(SpectrogramJudge_FigPath))
                    # #model = SketchModel()
                    
                    # model = AlexNet(num_classes).to(device)  # ここでネットワーク構造を指定すると、学習結果を再度ロードする必要がある
                    # #print("model = ", model)
                    
                    # #param = torch.load(model_dir + model_name)
                    # #param = torch.load(model_path, map_location=torch.device('cpu')) # ファイルから読み込む場合
                    # #model.load_state_dict(param, strict=False)　# ファイルから読み込む場合

                    # # とりあえずネットワークモデルをcudaのマップで読み込むことが必要
                    # model_path = 'TrainedNet_cuda.mdl' 
                    # model.load_state_dict(torch.load(model_path))

                    # # この部分はあとでループの外に出すこと


                    model.eval()  # 検証モードに設定

                    # preprocess = transforms.Compose([
                    # transforms.Resize(64,64),
                    # #transforms.Resize(256),#縦横比を保ちつつ小さいサイズでリサイズされる
                    # #transforms.CenterCrop(224),  # 画像を正方形で切り取る
                    # transforms.ToTensor(),
                    # #normalize#正規化
                    # ])

                    #judgeImages = torchvision.datasets.ImageFolder(root = SpectrogramJudge_Path, transform = transformer)  # このあたりでサブフォルダーがないというエラーが出る##############################

                    #judge_dataloader = torch.utils.data.DataLoader(judgeImages, batch_size = 1, shuffle = False, num_workers = cpu_core)  # Matlab の imdsTrain に相当

                    img = plt.imread(SpectrogramJudge_FigPath)  # イメージをファイルから読み込む; JPGを読み込むと(1,3,224,224)となるが、PNGだと透明層があるので(1,4,224,224)となってエラー; カラーは3チャンネルで読む必要あり
                    # 1. PIL Image を標準化したテンソルにする。
                    # 2. バッチ次元を追加する。 (C, H, W) -> (1, C, H, W)
                    # 3. 計算するデバイスに転送する。
                    # 画像をチェック
                    #print(img.shape)
                    #plt.imshow(img)
                    #plt.show()
                    #input("画像をチェック; Press Return to go next")

                    # 画像の形式をTensorに変換; ここでWarning出る
                    #inputs= transformer(img).unsqueeze(dim=0).to(device)  # transformer は1450行付近参照, unsqueezeで次元を (3,224,224) => (1,3,224,224) と変換し、to(device)でdevice用の配列に変換する; ここでWarning出る
                    #
                    # ここで下記のWarningが出る（transformのWarning）
                    #
                    # /home/keisoku/.local/lib/python3.8/site-packages/torchvision/transforms/functional.py:114: 
                    # UserWarning: 
                    # The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. 
                    # This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. 
                    # You may want to copy the array to protect its data or make it writeable before converting it to a tensor. 
                    # This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:143.)
                    # img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
                    #
                    # transform に img を入力するときに明示的に np.array 形式に変換してから入力するとOK
                    #inputs = transformer(np.array(img)).unsqueeze(dim=0).to(device)  # transformer は1450行付近参照, unsqueezeで次元を (3,224,224) => (1,3,224,224) と変換し、to(device)でdevice用の配列に変換する; OK; Warning 出ない
                    #または
                    inputs = transforms.ToTensor()(np.array(img)).unsqueeze(dim=0).to(device)  # unsqueezeで次元を (3,224,224) => (1,3,224,224) と変換し、to(device)でdevice用の配列に変換; OK; Warning 出ない                
                    #print("inputs = ", inputs)
                    #print("inputs.size() = ", inputs.size())  # JPGだと(1,3,224,224)となるが、PNGだと透明層が追加されて(1,4,224,224)となるので透明層の除去が必要 => すべてJPGに統一する
                    #input("画像の形式をTensorに変換 Press Return to go next")


                    #print("device = ", device)
                    # Judge each image one after another (SLOW) 判別（or 推論）
                    with torch.no_grad():  # 微分なしで以下を実行：順伝搬を行い判別（or 推論）
                        outputs = model(inputs)  # JPGだと動作するが、PNGだとエラーとなる（PNGは透明層を除去する必要あり）                    
                        #print("outputs = ", outputs)  # ただし、outputsの数値が小さすぎる --> ネットワークモデルを正しくロードすることで改善した

                        # 確率の最も高いクラスを予測ラベルとする。
                        class_id = int(outputs.argmax(dim=1)[0])  # For 2 classes, dict: allImages.class_to_idx = 'Abnrml':0, 'Normal':1,  
                                                                  # For 3 classes, dict: allImages.class_to_idx = 'AbnLrg':0, 'AbnSml':1, 'Normal':2
                        #print("class_id = ", class_id)
                        #print("dict: allImages.class_to_idx = 'Abnrml':0, 'Normal':1")
                        
                        # 異常である確率を求める 1に近いほど異常 少数第5位までを四捨五入（outputs 内のデータの順番が決まる要因は？：学習データ用ディレクトリの順, AbnLrg, AbnSml, Normal の順？）
                        # For 2 classes
                        # Abnrml_probability = round(float((F.softmax(outputs, dim=1)[0][0])),5) # 異常
                        # For 3 classes
                        AbnLrg_probability = round(float((F.softmax(outputs, dim=1)[0][0])),5) # 異常大
                        AbnSml_probability = round(float((F.softmax(outputs, dim=1)[0][1])),5) # 異常小
                        Normal_probability = round(float((F.softmax(outputs, dim=1)[0][2])),5) # 正常
                        #print("AbnLrg_probability =", AbnLrg_probability)
                        #print("AbnSml_probability =", AbnSml_probability)
                        #print("Normal_probability =", Normal_probability)

                        # クラス予測ラベルの反転版を作成
                        # For 2 classes:
                        # 注意!! 0->正常, 1->異常に反転する
                        # class_id_reverse = 1 - class_id
                        #print("class_id_reverse = ", class_id_reverse)
                        # For 3 classes:
                        # 注意!! 0->正常, 1->異常小, 2->異常大に反転する (元のclass_id: AnbLrg=0, AbnSml=1, Normal=2 ===>> 反転後のclass_id_reverse: AbnLrg=2, AbnSml=1, Normal=0)
                        class_id_reverse = 2 - class_id
                        #print("class_id_reverse = ", class_id_reverse)
                        # クラス予測ラベル保存
                        class_id_reverse_list.append(class_id_reverse)
                    #
                    #print("type(JudgeTime) = ",type(JudgeTime))
                    #input("ひとつずつ判別 Press Return to go next")
                    
                    # 判別結果を保存（２クラス、３クラスの場合の両方についてこれで良い）
                    AbnormalStateJudgeResults[0,j] = int(JudgeTime)             # [s] unixtime of the present judgement
                    AbnormalStateJudgeResults[1,j] = int(class_id_reverse)      # Reverse Class Index 'Abnrml Large':2, 'Abnormal Small':1, 'Normal':0]

                    # 判別結果の確率を保存
                    # For 2 classes
                    # AbnormalStateProbability[0,j]  = int(JudgeTime)             # [s] unixtime of the present judgement
                    # AbnormalStateProbability[1,j]  = float(Abnrml_probability)  # Probability of Abnormal State 異常状態である確率[0,1]
                    # For 3 classes
                    AbnormalStateProbability[0,j]  = int(JudgeTime)             # [s] unixtime of the present judgement
                    AbnormalStateProbability[1,j]  = float(AbnLrg_probability)  # Probability of Abnormal Large State 異常大である確率[0,1]
                    AbnormalStateProbability[2,j]  = float(AbnSml_probability)  # Probability of Abnormal Small State 異常小である確率[0,1]
                    AbnormalStateProbability[3,j]  = float(Normal_probability)  # Probability of Normal State         正常である確率[0,1]
                    # ここは多分OK


                    #########################################################################
                    # 個別の判別はほぼOK
                    #
                    # 結果の outputs を０〜１で線形マッピングして確率を求めるにはどうするか？
                    # Softmax層追加等？
                    # AbnormalStateProbability[2,jjudge_shiftmax] に入れたい
                    #
                    # 結果を配列に記録
                    #
                    #########################################################################

                    #input("Press Return to go next")
                    
                    # 推論結果を表示する。
                    #label = labels[class_id]

                    #addText = test_img_name + " " + str(class_id)

                    ###################################################ここまで

                    # イメージデータストア
                    #imdsJudge = imageDatastore(ScalogramJudge_Fig);

                    # 判別分析
                    #[YJudgePredict, JudgeProbabilities]  = classify(trainedAlexNet,imdsJudge) # trainedAN によるclassify（分類）: Ypredict=判別予測クラスラベル, probabilities=予測確率（N行 K列の行列, Nは検証データ数, Kはクラスラベルの数=2）
                    #[YJudgePredict, JudgeProbabilities]  = classify(trainedAlexNet,imJudge)   # probabilities = [Abnormalの確率, Normalの確率]

                    # 評価時刻のunixtimeを保存
                    #AbnormalStateProbability(1,j) = JudgeTime; #unixtime を保存
                    # 判別結果の確率を保存
                    #AbnormalStateProbability(2,j) = JudgeProbabilities(1,1); # [Abnormal state, Normal state]より、Abnormal state の確率を保存
                #
                # end j 判別分析期間のスイープと画像作成終了
                print("Generating Judge Images Completed.")
                print("Number of Spectrograms for Judgement 判別解析用スペクトログラム数 = ", count)
            #
            # end if judge_shiftmax > 0 である場合の終了
        #
        # end if 判別分析のための画像データを新たに作成する場合の終了
        

        # 判別分析データ数
        print("AbnormalStateJudgeResults = ", AbnormalStateJudgeResults)
        JudgeDataCount = jjudge_shiftmax
        print("jjudge_shiftmax = ", jjudge_shiftmax)
        print("JudgeDataCount  = ", JudgeDataCount)
        
        # Judge Date-Time
        #JudgeTime_unixtime = AbnormalStateJudgeResults[0,:]
        #JudgeTime_datetime = ???

        ###############################
        # 一括イメージデータストア生成
        ###############################
        #imdsJudge = imageDatastore(ScalogramJudge_Path)

        #judgeImages = torchvision.datasets.ImageFolder(root = SpectrogramJudge_Path, transform = transformer)  # このあたりでサブフォルダーがないというエラーが出る##############################

        #judge_dataloader = torch.utils.data.DataLoader(judgeImages, batch_size = batch_size, shuffle = False, num_workers = cpu_core)  # Matlab の imdsTrain に相当

        ###############
        # 一括判別分析
        ###############
        # [YJudgePredict, JudgeProbabilities] = classify(trainedAlexNet,imdsJudge)  # trainedAN によるclassify（分類）: Ypredict=判別予測クラスラベル, probabilities=予測確率（N行 K列の行列, Nは検証データ数, Kはクラスラベルの数=2）

        # # 評価時刻のunixtimeを生成、保存
        # for j in range(JudgeDataCount)  # j = 0:JudgeDataCount-1
        #     JudgeTime = JudgePeriod_StartingTime + (j-1)*JudgeTimeShift   #[s] 判別分析評価時刻を判別分析区間の開始日（= 2020年１月１日０時０分０秒）に設定
        #     AbnormalStateProbability(1,j) = JudgeTime  # unixtime を保存
        # # end

        # # 判別結果の確率を保存
        # AbnormalStateProbability(2,:) = JudgeProbabilities(:,1)  # [Abnormal state, Normal state]の１列目：Abnormal stateの確率を保存

        # # 判別結果の分類が異常状態である確率を確保 （注：正常状態である確率 = 1 - 異常状態である確率）
        # AbnormalStateProbability = AbnormalStateProbability'  # unixtime(UTC) vs. Abnormal State:異常状態である確率: Transpose: 縦長の行列へ変換 (jjudge_shiftmax行 x 2列)


        # #######################
        # # まとめて判別する場合
        # #######################
        # model.eval()
        # with torch.no_grad():
        #     for images, labels in judge_dataloader:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = net(images)
        # print("images = ", images)
        # print("labels = ", labels)
        # print("outputs = ", outputs)


        ####################################################################
        # Save Judgement Results into a File 判別結果をPickleファイルに保存
        ####################################################################
        # # Matlab:
        # # 判別分析結果の確率変数(AbnormalStateProbability)をカレントディレクトリ内のファイル（'AbnormalStateProbability.mat', ファイルサイズ約???MB）に保存
        # save('AbnormalStateProbability.mat','AbnormalStateProbability')
        # # 判別分析結果(YJudgePredict)を'YJudgePredict.mat'に保存
        # save('YJudgePredict.mat','YJudgePredict')
        # save('JudgeDataCount.mat','JudgeDataCount')

        ###########################################################################################################################################
        # Save "AbnormalStateProbability" into a binary file under current directory "Miscellaneous_.../AbnormalStateProbability.pcl" バイナリ保存
        ###########################################################################################################################################
        print('Saving Judgement Probabilities for the Latest Time Period, 判定結果の確率データをファイルに保存中 ....')
        with open(os.path.join(Miscellaneous_DirectoryPath,'AbnormalStateProbability.pcl'), mode='wb') as fdmp:   # .pcl ファイル
            pickle.dump(AbnormalStateProbability, fdmp)                 # Judgeデータをダンプ（ファイルサイズ約??）
                                                                        # with open の後は自動的に close される

        ###########################################################################################################################################
        # Save "AbnormalStateJudgeResult" into a binary file under current directory "Miscellaneous_.../AbnormalStateJudgeResult.pcl" バイナリ保存
        ###########################################################################################################################################
        print('Saving Judgement Results for the Latest Time Period, 判定結果データをファイルに保存中 ....')
        with open(os.path.join(Miscellaneous_DirectoryPath,'AbnormalStateJudgeResults.pcl'), mode='wb') as fdmp:  # .pcl ファイル
            pickle.dump(AbnormalStateJudgeResults, fdmp)                # Judgeデータをダンプ（ファイルサイズ約??）
                                                                        # with open の後は自動的に close される

        # ################################################################################################
        # # Save "AbnormalStateProbability" into a Text file 判別結果の確率データをテキストファイルへ保存
        # ################################################################################################
        # # np.transpose(AbnormalStateProbability) を np.savetxt で保存すればよい
        # print('Save Judgement Probability into a Text File "Miscellaneous_.../AbnormalStateProbability.txt"')
        # np.savetxt(os.path.join(Miscellaneous_DirectoryPath,'AbnormalStateProbability.txt'), np.transpose(AbnormalStateProbability), fmt = '%.d  %.5f')  # unixtime vs [0,1] float 浮動少数点型

        # ####################################################################################
        # # Save "AbnormalStateJudgeResult" into a Text file 判別結果をテキストファイルへ保存
        # ####################################################################################
        # # np.transpose(AbnormalStateJudgeResults) を np.savetxt で保存すればよい
        # print('Save Judgement Results into a Text File "Miscellaneous_.../AbnormalStateJudgeResults.txt"')
        # np.savetxt(os.path.join(Miscellaneous_DirectoryPath,'AbnormalStateJudgeResults.txt'), np.transpose(AbnormalStateJudgeResults), fmt = '%.d  %.d')  # unixtime vs 1/0　整数


        #################################################################
        # ディレクトリ作成してその下に判別結果を保存する（テキストで保存）
        #################################################################
        # 判別結果の時間データを系列用の特定のディレクトリの下のテキストファイルに保存
        print('Saving Judge Results into a File in a Specific Directory...')

        # ファイル名とディレクトリ名、パスの生成
        RadiowaveData_JudgeResults_File          = "RadiowaveData_JudgeResults.txt"
        RadiowaveData_JudgeResults_Directory     = "RadiowaveData_JudgeResults_" + City_DR_No_BCLoc_Freq
        RadiowaveData_JudgeResults_DirectoryPath = os.path.join(Spectrogram_DeepLearning_Path, RadiowaveData_JudgeResults_Directory)
        RadiowaveData_JudgeResults_FilePath      = os.path.join(RadiowaveData_JudgeResults_DirectoryPath, RadiowaveData_JudgeResults_File)
        RadiowaveData_JudgeResults_AbnormalSpan_File = "RadiowaveData_JudgeResults_AbnormalSpan.txt"
        RadiowaveData_JudgeResults_AbnormalSpan_FilePath = os.path.join(RadiowaveData_JudgeResults_DirectoryPath, RadiowaveData_JudgeResults_AbnormalSpan_File)

        # ディレクトリがなければ作成
        if not os.path.isdir(RadiowaveData_JudgeResults_DirectoryPath):
            os.makedirs(RadiowaveData_JudgeResults_DirectoryPath)
        # end if

        # # 判別結果に関する変数
        # AbnormalStateJudgeResults[0,j] = int(JudgeTime)             # [s] unixtime of the present judgement
        # AbnormalStateJudgeResults[1,j] = int(class_id_reverse)      # Reverse Class Index 'Abnrml':1, 'Normal':0
        # AbnormalStateProbability[0,j]  = int(JudgeTime)             # [s] unixtime of the present judgement
        # AbnormalStateProbability[1,j]  = float(Abnrml_probability)  # Probability of Abnormal State 異常状態である確率[0,1]

        # フォーマット：年月日　  Unixtime(UTC)    異常大である確率[0-1]　          異常小である確率[0-1]     　　    異常の判定（異常大=2  異常小=1　正常=0）
        #             DateTime    Unixtime(UTC)    Abnormal Large Probability[0-1]  Abnormal Small Probability[0-1]   Abnormal State(Abnormal Large=2, Abnormal Small=1, Normal=0)

        ######################################################################
        # 判別分析結果保存（判別分析データ数はJudgeDataCount=jjudge_shiftmax）
        ######################################################################
        with open(RadiowaveData_JudgeResults_FilePath,'w') as fileID:  # Write 書き込みモード
            #print('#DateTime          Unixtime(UTC)   Abnormal Probability[0-1]  Abnormal State(Abnormal=1, Normal=0)\n', file = fileID) # Python: \n not necessary if file closed
            print('#DateTime          Unixtime(UTC)   Abnormal_Large Probability[0-1]   Abnormal_Small Probability[0-1]   Abnormal State(Abnormal_Large=2, Abnormal_Small=1, Normal=0)', file = fileID)  # File Header, ファイルヘッダー 

        with open(RadiowaveData_JudgeResults_FilePath,'a') as fileID:  # Append 追記モード
            for j in range(JudgeDataCount):  # (= jjudge_shiftmax) 判別分析データ数, j=0,JudgeDataCount-1
                # 評価時刻の文字列生成
                tmp_unixtime    = AbnormalStateProbability[0,j]  # [s] unixtime of the present judgement
                string_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.fromtimestamp(tmp_unixtime,tz=None))  # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                # 保存内容生成
                fileID.write("{}      {:12.0f}         {:7.3f}                        {:7.3f}                            {:1.0f}\n".format(
                    string_datetime, AbnormalStateProbability[0,j], AbnormalStateProbability[1,j], AbnormalStateProbability[2,j], AbnormalStateJudgeResults[1,j]))
            #
            #end for
        #
        #end with (close not necessary)

        #####################################
        # 異常大、異常小の区間のファイル生成
        #####################################
        with open(RadiowaveData_JudgeResults_AbnormalSpan_FilePath,'w') as fileID:
            
            # fileID.readline()
            count = 0
            for m in range(JudgeDataCount):  # (= jjudge_shiftmax) 判別分析データ数, j=0,JudgeDataCount-1
                count = count + 1  
                if count < JudgeDataCount:
                    # 評価時刻の文字列生成
                    tmp_start_unixtime   = AbnormalStateProbability[0,m]  # [s] unixtime of the present judgement
                    tmp_end_unixtime     = AbnormalStateProbability[0,m+1]
                    # print("AbnormalStateProbability[0,m] =", AbnormalStateProbability[0,m])
                    # print("AbnormalStateProbability[0,m+1] =", AbnormalStateProbability[0,m+1])
                    # print("tmp_unixtime =", tmp_unixtime)
                    string_start_datetime = "{0:%Y%m%d_%H}".format(datetime.fromtimestamp(tmp_start_unixtime,tz=None))  # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                    # print("string_start_datetime =", string_start_datetime)
                    string_end_datetime = "{0:%Y%m%d_%H}".format(datetime.fromtimestamp(tmp_end_unixtime,tz=None))  # String, Datetime format = YYYYmmdd_HHMMSS, timezone=None: Local (Asia/Tokyo = +9:00)
                    # print("string_end_datetime =", string_end_datetime)
        
                    if AbnormalStateJudgeResults[1,m] == 1 :
                        # fileID.write("{}  {:11.0f}\n".format(string_datetime, string_datetime[m+1]))
                        fileID.write("{}  {}\n".format(string_start_datetime, string_end_datetime))
                    elif AbnormalStateJudgeResults[1,m] == 2:
                        # fileID.write("{}  {:11.0f}\n".format(string_datetime, string_datetime[m+1]))
                        fileID.write("{}  {}\n".format(string_start_datetime, string_end_datetime))
                    # end if
                #
                # end if
            #
            # end for
        #
        # end with
    #
    # end if 判別分析をする(=1)場合の終了
    #
    ##########################################
    # For the case Judgement is NOT performed
    ##########################################
    #
    elif perform_judge == 0: # 判別分析をしない(=0)
        print('Skip Judgement Analysis')

        ######################
        # Read data from file 
        ######################

        if (os.path.isfile(os.path.join(Miscellaneous_DirectoryPath,"AbnormalStateProbability.pcl")) and os.path.getsize(os.path.join(Miscellaneous_DirectoryPath,"AbnormalStateProbability.pcl")) > 0):  # データファイルがあり、サイズがゼロではなければ読み込む
            ############################################
            # Probability データの読み込み（ロード）
            ############################################
            # file load ファイル読み込み
            print('Reading Judgement Probability File "Miscellaneous_.../AbnormalStateProbability.pcl" 判定結果の確率データファイルの読み込み中 ....')
            with open(os.path.join(Miscellaneous_DirectoryPath,'AbnormalStateProbability.pcl'), mode='rb') as fdmp:   # ファイル
                AbnormalStateProbability = pickle.load(fdmp)                # からデータをロード
                                                                            # with open の後は自動的に close される
            # end with
            print('Reading Judgement Probability Data File "Miscellaneous_.../AbnormalStateProbability.pcl" Done 判定結果ファイルの読み込み完了')
        # end if

        if (os.path.isfile(os.path.join(Miscellaneous_DirectoryPath,"AbnormalStateJudgeResults.pcl")) and os.path.getsize(os.path.join(Miscellaneous_DirectoryPath,"AbnormalStateJudgeResults.pcl")) > 0):  # データファイルがあり、サイズがゼロではなければ読み込む
            ############################################
            # JudgementResults データの読み込み（ロード）
            ############################################
            # file load ファイル読み込み
            print('Reading Judgement Result File "Miscellaneous_.../AbnormalStateJudgeResults.pcl" 判定結果ファイルの読み込み中 ....')
            with open(os.path.join(Miscellaneous_DirectoryPath,'AbnormalStateJudgeResults.pcl'), mode='rb') as fdmp:  # SpectrogramDump.pcl ファイル
                AbnormalStateJudgeResults = pickle.load(fdmp)               # からスペクトログラムデータをロード
                                                                            # with open の後は自動的に close される
            # end with
            print('Reading Judgement Result Data File "Miscellaneous_.../AbnormalStateJudgeResults.pcl" Done 判定結果ファイルの読み込み完了')
        # end if
    #
    # end if judgement is performed (=1) or not (=0)

    #input("Judging Process Ended. Press Enter to Continue for Plotting Results.")



    #############################
    #############################
    # 判別結果のグラフ表示と保存
    #############################
    #############################

    if plot_judge_results == 1:  # 判別結果をプロットする(=1)またはしない(=0)

        ##########################################
        # 必要なデータをダンプファイルから読み込む（まだ、下記のうちいくつか）
        ##########################################
        # 観測データ（ここでは読み込み不要）
        # スペクトログラム（ここでは読み込み不要）
        # 地震データ（ここでは読み込み不要）

        # 判別結果（判別後にダンプ必要）


        ##############################################################################
        # 判別結果のプロット (Unixtime, Posixtime => Datetime 変換：日付で指定の場合)
        ##############################################################################
        # Set Plot Range in Unixtime　プロットの横軸の範囲：日付（０時）で指定
        # 年月日の文字列
        #plot_starting_date = "20210401"  # Date 文字列
        #plot_ending_date   = "20210601"  # Date 文字列
        plot_starting_date = plot_starting_date0 # 冒頭部分で設定
        plot_ending_date   = plot_ending_date0   # 冒頭部分で設定
        # Conversion to datetime object; 年月日の文字列をdatetimeオブジェクトに変換
        # datetimeオブジェクト
        plot_starting_dtobject = datetime.strptime(plot_starting_date, "%Y%m%d")  # datetime object of datetime module = dtobject
        plot_ending_dtobject   = datetime.strptime(plot_ending_date,   "%Y%m%d")  # datetime object of datetime module = dtobject
        ##########################################################################################################################################################################################
        ##########################################################################################################################################################################################
        # プロットの最終時刻をスペクトログラムの最終時刻（最新時刻）にする場合（冒頭で設定）
        if plot_upto_latest_time == 1:
            plot_ending_dtobject = datetime.fromtimestamp(Spectrogram_TimeEnd,tz=None)  # プロットの最終時刻をスペクトログラムの最終時刻（最新時刻）にする; Spectrogram_TimeEnd は unixtime（秒）
        #end if
        ##########################################################################################################################################################################################
        ##########################################################################################################################################################################################
        print("plot_starting_dtobject プロット開始日時オブジェクト = ", plot_starting_dtobject)
        print("plot_ending_dtobject   プロット終了日時オブジェクト = ", plot_ending_dtobject)
        #plot_starting_time = JudgePeriod_StartingTime  # [s] unixtime, start time of the plot range プロットの横軸の開始時刻##################################################### Set!
        #plot_ending_time   = JudgePeriod_EndingTime    # [s] unixtime, end   time of the plot range プロットの横軸の終了時刻##################################################### Set!
        # datetimeオブジェクトをunixtime (posixtime, timestamp) に変換
        # unixtime （秒）
        plot_starting_time = datetime.timestamp(plot_starting_dtobject)  # datetime object => unixtime, posixtime, timestamp に変換
        plot_ending_time   = datetime.timestamp(plot_ending_dtobject)    # datetime object => unixtime, posixtime, timestamp に変換

        #----------------
        # プロット軸準備
        #----------------
        #fig = plt.figure(figsize=(17, 11), tight_layout="True")  #size in inch (100 pix/inch)
        fig = plt.figure(figsize=(15, 9),constrained_layout="True")  # constrained mode: more flexible than tight, but tentative 試験中のモード
        ax1 = fig.add_subplot(6,1,3)
        ax2 = fig.add_subplot(6,1,4)
        ax3 = fig.add_subplot(6,1,5)
        ax4 = fig.add_subplot(6,1,6)
        ax5 = fig.add_subplot(6,1,2)
        ax6 = fig.add_subplot(6,1,1)


        #-------------------------------------------------------------------------------------------------------------------
        # 1. Radiowave Data 時系列観測データのプロット unixtime vs dBm (RadiowaveDataExtracted_tdataは mW で格納されている)
        #-------------------------------------------------------------------------------------------------------------------
        #ax1.plot(RadiowaveDataExtracted_utime, 10*np.log10(RadiowaveDataExtracted_tdata), color='black',  linestyle='-')
        #print("RadiowaveDataExtracted_utime = ", RadiowaveDataExtracted_utime)
        #input("Press Enter to continue.")
        #print("datetime.fromtimestamp(RadiowaveDataExtracted_utime[0]) = ", datetime.fromtimestamp(RadiowaveDataExtracted_utime[0]))
        #input("Press Enter to continue.")
        #
        # Highlighting Part Plot ハイライト領域をすべて順次表示（プロット範囲外も含めて全て表示。ハイライト部を背景にするため、観測データプロットの前に描画）
        #
        # 参考: AbnormalStateJudgeResults[0,j] = int(JudgeTime)         # [s] unixtime of the present judgement
        #       AbnormalStateJudgeResults[1,j] = int(class_id_reverse)  # Reverse Class Index 'Abnrml Large':2, 'Abnormal Small':1, 'Normal':0]
        #
        # for n in range(len(AbnormalStateJudgeResults[0,:])):
        for j in range(JudgeDataCount):  # (= jjudge_shiftmax) 判別分析データ数, j=0,JudgeDataCount-1
            if AbnormalStateJudgeResults[1,j] == 2:  # if the reverse class index = 2 (Abnormal Large Case)
                # Abnormal Large State: 異常大の判別解析の個別区間の時間を生成
                highlight_endingtime   = datetime.fromtimestamp(AbnormalStateJudgeResults[0,j])           # unixtimeからdatetimeオブジェクトに変換 ハイライト領域の終わりの時刻がその領域の評価時刻としている（see line no. 2938, 3174 for "JudgeTime")
                highlight_startingtime = datetime.fromtimestamp(AbnormalStateJudgeResults[0,j]-6.0*3600)  # unixtimeからdatetimeオブジェクトに変換 ハイライト領域の始まりの時刻はその「判別分析区間長（６時間等）」前の時刻としている
                ax1.axvspan(highlight_startingtime, highlight_endingtime, color= 'salmon',    alpha=0.8)  #ハイライト表示 うす赤 pink, lightpink, mistyrose, orangered
            #
            elif AbnormalStateJudgeResults[1,j] == 1:  # if the reverse class index = 1 (Abnormal Small Case)
                # Abnormal Small State: 異常小の判別解析の個別区間の時間を生成
                highlight_endingtime   = datetime.fromtimestamp(AbnormalStateJudgeResults[0,j])           # unixtimeからdatetimeオブジェクトに変換 ハイライト領域の終わりの時刻がその領域の評価時刻としている（see line no. 2938, 3174 for "JudgeTime")
                highlight_startingtime = datetime.fromtimestamp(AbnormalStateJudgeResults[0,j]-6.0*3600)  # unixtimeからdatetimeオブジェクトに変換 ハイライト領域の始まりの時刻はその「判別分析区間長（６時間等）」前の時刻としている
                ax1.axvspan(highlight_startingtime, highlight_endingtime, color= 'turquoise', alpha=0.8)  #ハイライト表示 うす青 lightcyan, mediumturquoise, skyblue, paleturquoise
            #
            # end if
        # end for    
        # 
        # 横軸時間表示用のdatetimeオブジェクトのリストを作成
        #
        RadiowaveDataExtracted_datetimeobject_list = []
        for n in range(len(RadiowaveDataExtracted_utime)):
            RadiowaveDataExtracted_datetimeobject_list.append(datetime.fromtimestamp(RadiowaveDataExtracted_utime[n]))  # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
        #input("Press Enter to continue.")
        #print("RadiowaveDataExtracted_datetimeobject_list = ", RadiowaveDataExtracted_datetimeobject_list)
        #input("Press Enter to continue.")
        #
        # RadiowaveData Plot 観測データプロット、最後にプロット範囲を指定
        #
        ax1.plot(RadiowaveDataExtracted_datetimeobject_list, 10*np.log10(RadiowaveDataExtracted_tdata), color='black',  linestyle='-', linewidth=0.5)  # 横軸はdatetimeオブジェクト、縦軸は dBm = 10*log10( mW )
        ax1.set_ylabel('Power [dBm]')
        #ax1.set_title('Radiowave Data ' + City_DR_No_BCLoc_Freq) 
        #ax1.set_xlim(plot_starting_time, plot_ending_time)  # Range of x
        ax1.set_xlim(plot_starting_dtobject, plot_ending_dtobject)  # Range of datetime object = dtobject（datetimeオブジェクトで横軸の範囲指定）
        ax1.set_ylim(-120, -80)
        ax1.grid()  # grid 表示


        #-------------------------------------------------------
        # 2. Spectrogram Plot　スペクトログラムのカラープロット
        #-------------------------------------------------------
        # spectrogram; ns行 spectrogram_maxsize列 の行列 スペクトログラム [dBm]
        # spectrogram[ns,:] = 20.0 * np.log10(abs(spectrum_data))  # use "np.log10", not "math.log10"
        Spectrogram_TimeStart        = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[0]])                      # Starting Time (unixtime) of "spectrogram" スペクトログラムの最初の時刻
        #print("Spectrogram_TimeStart = ", Spectrogram_TimeStart)
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ここから不要（あとで使用するかも）
        #spectrogram_maxsize          = len(spectrogram[:,0])
        #Spectrogram_TimeEnd          = float(RadiowaveDataExtracted_utime[indx_conversion_sg2td[spectrogram_maxsize-1]])  # Ending   Time (unixtime) of "spectrogram" スペクトログラムの最後の時刻
        #print("Spectrogram_TimeEnd   = ", Spectrogram_TimeEnd)
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ここまで不要（あとで使用するかも）
        # calculate start and end indices for spectrogram 横軸の範囲をインデックスで設定
        PlotSpectrogram_IndxStart        = int((plot_starting_time - Spectrogram_TimeStart)//spectrogram_timestep)  # スペクトログラム中のプロット開始時点のインデックスを計算
        #print("PlotSpectrogram_IndxStart = ", PlotSpectrogram_IndxStart)
        PlotSpectrogram_IndxEnd          = int((plot_ending_time   - Spectrogram_TimeStart)//spectrogram_timestep)  # スペクトログラム中のプロット終了時点のインデックスを計算
        #print("PlotSpectrogram_IndxEnd   = ", PlotSpectrogram_IndxEnd)
        print("Processing Spectrum Data for Plot ... Please Wait スペクトログラムをプロット中 ... しばらくお待ちください")
        ax2.pcolormesh(np.transpose(spectrogram[PlotSpectrogram_IndxStart:PlotSpectrogram_IndxEnd,:]), cmap="jet", vmin=-280, vmax=-190)              # ２次元カラープロット: 時間がかかる（横軸インデックスの範囲のデータのみプロット）
        #ax2.set_ylabel('Freq. [Hz]')
        ax2.set_ylabel('Log(Freq) [Points]')


        #------------------------------------------------------------------------------------------------
        # 3. Judgement Probability of Abnormal/Normal State 異常／正常状態の判別結果の確率データプロット
        #------------------------------------------------------------------------------------------------
        #ax3.plot(AbnormalStateJudgeResults[0,:], AbnormalStateJudgeResults[1,:], color='red',  linestyle='-',  label='train_acc') # unixtime vs Judge 0/1
        #print("AbnormalStateJudgeResults[0,:] = ", AbnormalStateJudgeResults[0,:])
        #input("Press Enter to continue.")
        #print("datetime.fromtimestamp(AbnormalStateJudgeResults[0,0]) = ", datetime.fromtimestamp(AbnormalStateJudgeResults[0,0]))
        #input("Press Enter to continue.")
        # 横軸時間表示用のdatetimeオブジェクトのリストを作成
        AbnormalStateProbability_datetimeobject_list = []
        for n in range(len(AbnormalStateProbability[0,:])):
            AbnormalStateProbability_datetimeobject_list.append(datetime.fromtimestamp(AbnormalStateProbability[0,n]))  # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
        #input("Press Enter to continue.")
        #print("AbnormalStateJudgeResults_datetimeobject_list = ", AbnormalStateJudgeResults_datetimeobject_list)
        #input("Press Enter to continue.")
        ax3.plot(AbnormalStateProbability_datetimeobject_list, AbnormalStateProbability[1,:], color='crimson',    linestyle='-', linewidth=1,  label='probability') # 横軸はdatetimeオブジェクト、縦軸は判定結果の1/0
        ax3.plot(AbnormalStateProbability_datetimeobject_list, AbnormalStateProbability[2,:], color='slateblue',  linestyle='-', linewidth=1,  label='probability')
        ax3.grid(which = "major", axis = "y", color = "black", alpha = 0.4, linestyle = "--", linewidth = 1)
        #ax3.set_xlabel('Unixtime [s]')
        ax3.set_ylabel('Abnormal Probability')
        ax3.set_yticks([0,0.5,1])
        #ax3.set_title('Judgement Results')
        #ax3.set_xlim(plot_starting_time, plot_ending_time)  # Range of x
        ax3.set_xlim(plot_starting_dtobject, plot_ending_dtobject)  # Range of datetime object = dtobject（datetimeオブジェクトで横軸の範囲指定）


        #----------------------------------------------------------------------------------
        # 4. Judgement Results of Abnormal/Normal State 異常／正常状態の判別結果のプロット
        #----------------------------------------------------------------------------------
        #ax3.plot(AbnormalStateJudgeResults[0,:], AbnormalStateJudgeResults[1,:], color='red',  linestyle='-',  label='train_acc') # unixtime vs Judge 0/1
        #print("AbnormalStateJudgeResults[0,:] = ", AbnormalStateJudgeResults[0,:])
        #input("Press Enter to continue.")
        #print("datetime.fromtimestamp(AbnormalStateJudgeResults[0,0]) = ", datetime.fromtimestamp(AbnormalStateJudgeResults[0,0]))
        #input("Press Enter to continue.")
        # 横軸時間表示用のdatetimeオブジェクトのリストを作成
        AbnormalStateJudgeResults_datetimeobject_list = []
        for n in range(len(AbnormalStateJudgeResults[0,:])):
            AbnormalStateJudgeResults_datetimeobject_list.append(datetime.fromtimestamp(AbnormalStateJudgeResults[0,n]))  # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
        #input("Press Enter to continue.")
        #print("AbnormalStateJudgeResults_datetimeobject_list = ", AbnormalStateJudgeResults_datetimeobject_list)
        #input("Press Enter to continue.")
        ax4.plot(AbnormalStateJudgeResults_datetimeobject_list, AbnormalStateJudgeResults[1,:], color='black',  linestyle='-', linewidth=1,  label='judge_results') # 横軸はdatetimeオブジェクト、縦軸は判定結果の1/0
        ax4.set_xlabel('Year-Month-Date')
        ax4.set_ylabel('Abnormal L=2, S=1')
        ax4.set_yticks([0,1,2])
        #ax4.set_title('Judgement Results')
        #ax4.set_xlim(plot_starting_time, plot_ending_time)  # Range of x
        ax4.set_xlim(plot_starting_dtobject, plot_ending_dtobject)  # Range of datetime object = dtobject（datetimeオブジェクトで横軸の範囲指定）


        #------------------------------------------------------------------------------
        # 5. EQ Data Plot 地震データのマグニチュードをバープロット（最上段より一つ下へ）
        #------------------------------------------------------------------------------
        ax5.set_xlim(plot_starting_dtobject, plot_ending_dtobject)
        ax5.set_ylim(0,8)
        ax5.set_yticks([0,1,2,3,4,5,6,7,8])
        ax5.set(yticklabels=[0,1,2,3,4,5,6,7,8])
        ax5.grid(which="major",axis="y",color="black",alpha=0.3,linestyle="--",linewidth=1)
        #ax5.tick_params(axis='x', labelrotation= 0) # or 90 etc.
        ax5.set_ylabel('EQ Magnitude')
        #ax5.set_title('Radiowave Data ' + City_DR_No_BCLoc_Freq) 
        ax5.plot([plot_starting_dtobject,plot_ending_dtobject], [5.0,5.0], color='darkgreen',  linestyle='-', linewidth=0.7) # horizontal eye-guide 目安の横線：横軸はdatetimeオブジェクト、縦軸はマグニチュード＝５を仮定 ################!!!
        EQ_Database_datetimeobject_plotlist = []
        #
        #-------------------------------------------------------------------------------------------------------------
        # 解析する観測データによってどの地震データを使用するか選択する：富山より北東１２県の地震データをバーでプロット
        #-------------------------------------------------------------------------------------------------------------
        if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
             City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"  or
             City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
            # 横軸時間表示用のdatetimeオブジェクトのリストを作成
            for n in range(len(EQ_Database_Toyama_Uxtimes)):
                EQ_Database_datetimeobject_plotlist.append(datetime.fromtimestamp(EQ_Database_Toyama_Uxtimes[n]))  # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
            #end for
            ax5.bar(EQ_Database_datetimeobject_plotlist,EQ_Database_Toyama_Mags,width=0.1,alpha=0.8,color='crimson') # マグニチュード表示（濃い赤）
        #
        #-----------------------------------------------------------------------------------------------------------
        # 解析する観測データによってどの地震データを使用するか選択する：静岡磐田近辺６県の地震データをバーでプロット
        #-----------------------------------------------------------------------------------------------------------
        elif(City_DR_No_BCLoc_Freq == "Iwata_NE_11_Yokohama_84p7MHz" or City_DR_No_BCLoc_Freq == "Iwata_NE_16_Shizuoka_88p8MHz" or
             City_DR_No_BCLoc_Freq == "Iwata_NE_4_Shizuoka_79.2MHz"  or City_DR_No_BCLoc_Freq == "Iwata_NW_2_Tsu_78p9MHz"       or
             City_DR_No_BCLoc_Freq == "Iwata_NW_8_Nagoya_82p5MHz"):
            # 横軸時間表示用のdatetimeオブジェクトのリストを作成
            for n in range(len(EQ_Database_Shizuoka_Uxtimes)):
                EQ_Database_datetimeobject_plotlist.append(datetime.fromtimestamp(EQ_Database_Shizuoka_Uxtimes[n]))  # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
            #end for
            ax5.bar(EQ_Database_datetimeobject_plotlist,EQ_Database_Shizuoka_Mags,width=0.1,alpha=0.8,color='slateblue') # マグニチュード表示（濃い青）
        #
        #end if


        #--------------------------------------------------
        # 6. EQ Data Plot 地震データのラベル表示（最上段へ）
        #--------------------------------------------------
        EQ_Database_datetimeobject_plotlist = [] # 横軸表示用リスト（datetimeオブジェクト)初期化
        EQ_Database_epicenters_j_plotlist = []   # 震央表示用リスト
        EQ_Database_mags_plotlist = []           # マグニチュード用リスト
        ax6.set_xlim(plot_starting_dtobject, plot_ending_dtobject)  # X軸範囲設定
        ax6.set_ylim(0,1)                                           # Y軸範囲設定
        ax6.set_title('Radiowave Data ' + City_DR_No_BCLoc_Freq)    # タイトル表示
        #ax6.axis("off")  # 軸、枠、ラベルをすべて消す場合
        ax6.set_xlabel('')
        #ax6.set_ylabel('Epicenters',loc="bottom") # y軸ラベルのみ表示
        ax6.set_ylabel('Epicenters',loc="center")  # y軸ラベルのみ表示
        ax6.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False) # 軸の目盛りの数値を消去
        ax6.tick_params(bottom=False, left=False, right=False, top=False)                     # 軸の目盛りを消去
        ax6.spines['right'].set_visible(False)  # 枠消去
        ax6.spines['top'].set_visible(False)    # 枠消去
        ax6.spines['bottom'].set_visible(False) # 枠消去
        ax6.spines['left'].set_visible(False)   # 枠消去
        #
        # ax6 のグラフの下の余白の調整方法が不明、できれば、余白をなくしたい
        #
        #----------------------------------------------------------------------------------------------------------------------
        # 解析する観測データによってどの地震データを使用するか選択する：富山より北東１２県の地震データのラベルをax.text()で表示
        #----------------------------------------------------------------------------------------------------------------------
        if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
             City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"  or
             City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
            # 横軸時間表示用のdatetimeオブジェクト、震央地、マグニチュードのリストを条件（マグニチュード５以上）に合うように作成
            for n in range(len(EQ_Database_Toyama_Uxtimes)):
                if all([plot_starting_dtobject < datetime.fromtimestamp(EQ_Database_Toyama_Uxtimes[n]), datetime.fromtimestamp(EQ_Database_Toyama_Uxtimes[n]) < plot_ending_dtobject]): # プロット期間内のリスト要素のみについて
                    if (EQ_Database_Toyama_Mags[n] >= 4.5):                                                                    # Mag >= 4.5 ならば、プロット用リスト作成 #################################################################!!!
                        EQ_Database_datetimeobject_plotlist.append(datetime.fromtimestamp(EQ_Database_Toyama_Uxtimes[n]))    # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
                        EQ_Database_epicenters_j_plotlist.append(EQ_Database_Toyama_Epicenters_j[n])                         # 震央地名（日本語）リスト
                        EQ_Database_mags_plotlist.append(EQ_Database_Toyama_Mags[n])                                         # マグニチュードのリスト
                    #end if
                #end if
            #end for
        #
        #--------------------------------------------------------------------------------------------------------------------
        # 解析する観測データによってどの地震データを使用するか選択する：静岡磐田近辺６県の地震データのラベルをax.text()で表示
        #--------------------------------------------------------------------------------------------------------------------
        elif(City_DR_No_BCLoc_Freq == "Iwata_NE_11_Yokohama_84p7MHz" or City_DR_No_BCLoc_Freq == "Iwata_NE_16_Shizuoka_88p8MHz" or
             City_DR_No_BCLoc_Freq == "Iwata_NE_4_Shizuoka_79.2MHz"  or City_DR_No_BCLoc_Freq == "Iwata_NW_2_Tsu_78p9MHz"       or
             City_DR_No_BCLoc_Freq == "Iwata_NW_8_Nagoya_82p5MHz"):
            # 横軸時間表示用のdatetimeオブジェクト、震央地、マグニチュードのリストを条件（マグニチュード５以上）に合うように作成
            for n in range(len(EQ_Database_Shizuoka_Uxtimes)):
                if all([plot_starting_dtobject < datetime.fromtimestamp(EQ_Database_Shizuoka_Uxtimes[n]), datetime.fromtimestamp(EQ_Database_Shizuoka_Uxtimes[n]) < plot_ending_dtobject]): # プロット期間内のリスト要素のみについて
                    if (EQ_Database_Shizuoka_Mags[n] >= 4):                                                                  # Mag >= 4 ならば、プロット用リスト作成 #################################################################!!!
                        EQ_Database_datetimeobject_plotlist.append(datetime.fromtimestamp(EQ_Database_Shizuoka_Uxtimes[n]))  # unixtimeからdatetimeオブジェクトに変換してリスト要素を追加
                        EQ_Database_epicenters_j_plotlist.append(EQ_Database_Shizuoka_Epicenters_j[n])                       # 震央地名（日本語）リスト
                        EQ_Database_mags_plotlist.append(EQ_Database_Shizuoka_Mags[n])                                       # マグニチュードのリスト
                    #end if
                #end if
            #end for
        #end if
        #
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 隣り合うラベルが重なるときにずらす(shift)処理（ラベルの時刻差が、datetimeオブジェクトで計算してプロット期間幅の1/75より小さいときには、その値（プロット期間幅の1/75）にする）
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        EQ_Database_datetimeobject_plotlist_shifted = []  # シフトした時刻を格納するリストを初期化
        for n in range(len(EQ_Database_datetimeobject_plotlist)):
            EQ_Database_datetimeobject_plotlist_shifted.append(EQ_Database_datetimeobject_plotlist[n])  # 元のdatetimeオブジェクトリストを保持しておく（あとでラベル引き出し線で使用するため）
        #end for
        for n in range(len(EQ_Database_datetimeobject_plotlist_shifted)-1):  # n は、最大のひとつ前までの処理：0〜len(EQ_Database_datetimeobject_plotlist)-1
            if(EQ_Database_datetimeobject_plotlist_shifted[n+1] - EQ_Database_datetimeobject_plotlist_shifted[n] < (plot_ending_dtobject - plot_starting_dtobject)/90): #ずらす間隔をプロット期間幅の 1/90 にする#######################################!!!
                EQ_Database_datetimeobject_plotlist_shifted[n+1] = EQ_Database_datetimeobject_plotlist_shifted[n] + (plot_ending_dtobject - plot_starting_dtobject)/90  #ずらす間隔をプロット期間幅の 1/90 にする#######################################!!!
            #end if
        #end for
        #---------------------------
        # シフトした位置へラベル表示
        #---------------------------
        for n in range(len(EQ_Database_datetimeobject_plotlist_shifted)):
            ax6.text(EQ_Database_datetimeobject_plotlist_shifted[n],0.2,EQ_Database_epicenters_j_plotlist[n],horizontalalignment="center",rotation=90)  # テキスト位置は0.2、日本語(_j)表示、横方向にセンターリング、９０度回転
        #end for
        #------------------------------------------------------------------------------------------------------------
        # ラベル引き出し線をそれぞれのラベルについて表示（注意：ラベル引き出し線は本来のグラフ枠の外には表示できない）
        #------------------------------------------------------------------------------------------------------------
        for n in range(len(EQ_Database_datetimeobject_plotlist)):
            ax6.plot([EQ_Database_datetimeobject_plotlist[n],EQ_Database_datetimeobject_plotlist_shifted[n]],[0.0,0.16], color='black',  linestyle='--', linewidth=0.5)
        #end for


        #-------------------------
        # Plot プロット描画と保存
        #-------------------------
        plt.ion()  # Plot Interactive ON (in order not to be blocked by the figure window) プロット後にプログラム継続させる
        plt.show()
        #plt.show(block=False)  # not work 効かない
        plt.pause(0.001)  # ion()後に必要

        # Save Plot ファイルに保存
        fig.savefig(os.path.join(Miscellaneous_DirectoryPath,"PlotResults.png"), bbox_inches="tight")

        # Confirmation of Ending 終了確認
        #input("Program has Ended; Press Enter to Close Figures. 「Enter」を押すとプロットを閉じて終了")  # Keep plots till Enter is pressed: Enter を押すまでグラフが保持され、押すとグラフ表示が消えて終了
    #
    #end if 判別結果をプロットする(=1)またはしない(=0)


    #####################################################################################################################################################
    ## End of the Function 関数終了 #####################################################################################################################
    #####################################################################################################################################################
