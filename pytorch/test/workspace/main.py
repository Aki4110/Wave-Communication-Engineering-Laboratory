##################################################################
##################################################################
#
# Main Program of Radiowave-Data Analysis with Deep Learning:
#
# Train, Valid, Judge and Plot
#
# ラジオ波観測データの深層学習による解析
#
# メインプログラム: 
#       main, 解析条件の設定
#
# 関数：spectrogram_deeplearning,
#       スペクトログラム生成
#       AlexNet または ResNet18 により深層学習、検証、判別、
#       グラフ表示
#
# Python3 Version
# March 2021
# Revised on Sep. 3, 2021
#
# 2021 Oct. 10: Divided into main and deep-learning function
# 
# Not yet: Analyze Other Datasets and Perform Combined Judgement
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
# プログラム中止コマンドは import sys してから sys.exit()
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
#from   torchvision  import datasets, models
import torchvision.transforms as transforms
from   torchsummary import summary  # Torch Summary (pip3 install torchsummary) (torchsummary-1.5.1)
from   sklearn.model_selection import train_test_split  # scikit-learn (pip3 install scikit-learn)

#
# Import My Modules
#
from spectrogram_deeplearning import spectrogram_deeplearning  # 'from' is necessary

#
# Measure Computing Time
#
ComputeStartTime = time.time()  # 開始時刻 （UNIX時間（エポック秒）を浮動小数点数float型で取得）
# or using datetime object
ComputeStartDateTime = datetime.now()  # 開始時刻 （datetime オブジェクトで取得）

print('######################################################################')
print('######################################################################')
print('### Start Radiowave Data Analysis by Spectrogram and Deep Learning ###')
print('######################################################################')
print('######################################################################')

################
################
# 解析条件の設定
################
################

########################
# スペクトログラムの更新
########################
# 最新の観測結果のデータを読み込み、全期間のスペクトログラムを更新する
# 観測結果のデータ読み込み：常に読み込む（読み込みは約５秒、観測結果のサンプリング時間を計算、判定結果と同時に表示するため必要）
update_whole_spectrogram       = 0  # 1 = 全期間のスペクトログラムを更新する（約１時間）既存のスペクトログラムデータファイルは別名で保存, 0 = 更新せず既存のスペクトログラムデータファイルがあれば読み込んで利用する

################################
# 学習用画像データの作成方法設定
################################
delete_train_images_before_gen = 0  # 1 = 学習用画像データを作成する前に既存の画像データを一括消去する, 0 = 一括消去しないで残す
generate_train_images          = 1  # 1 = 新たな学習用画像データを作成する（作成する場合でも、既存のデータがあれば再作成しないでスキップ、もとのデータを残す）, 0 = 作成しない（既存のファイルは変更せずに学習、検証のみ）
set_normal_state_as_non_abnrml = 1  # 1 = 正常状態を正常状態のリストから読まず、異常状態以外のすべての期間を正常状態とする, 0 = 正常状態を正常状態のリストから読み込む
generate_reference_images      = 0  # 1 = 新たな学習用画像データを作成する場合、参考画像ファイル（時間データ、軸付きスペクトログラム）を作成する（デバッグ用：作成すると多数のファイルを生成するため遅くなる）, 0 = 作成しない
analyze_actual_spectrogram     = 1  # 1 = 本番解析（実際のスペクトログラムを生成またはファイルから読み込む）, 0 = デバッグ（データファイルを読まず、フーリエ変換も省略し、ダミースペクトログラムを生成する）

##########################
# 学習と検証プロセスの実行
##########################
####################################################################################################################################################################
# 説明：
# 学習結果のネットワーク変数(trainedAlexNet 等)はカレントディレクトリ内のファイル（'trainedAlexNet.mat', TrainedNet_???.mdl 等, ファイルサイズ約200MB）に保存される
# 学習・検証しない場合は、以前に学習した結果のネットワーク変数をファイルから読み込む
# ネットワーク変数格納ファイル：
# Matlab: trainedAlexNet.mat
# Python: TrainedNet_cpu.mdl または TrainedNet_cuda.mdl (計算環境が CPU か GPU によって異なる)
#####################################################################################################################################################################
perform_train_and_varid        = 0  # 1 = 学習、検証をする(学習結果のネットワークモデルファイル'TrainedNet_???.mdl'を生成・保存する), 0 = しない（既存のネットワークモデルファイルを読込む）
                                    # 早い段階で転移学習できるように改良すること
# Network Model ネットワークモデル選択(1: AlexNet, 2: ResNet18(Local), 3: ResNet34(GitHub), 4: ResNet50(Local), 5: ResNet50(GitHub), 6: ResNet152(GitHub) (3,5,6: Pre-Trained Network for Cuda from GitHub 事前学習済みモデル:GPU使用))
networkmodel   = 2
# Training Options トレーニングオプション 学習パラメータ Hyper-Parameters ハイパーパラメータ
num_epochs0    = 50        # エポック数（１エポックは解析がすべての画像データについて一巡すること）
LearningRate0  = 0.005     # Learning Rate 学習率（一定値）0.01 ~ 0.001 でそれほど変わらず
#batch_size0   = 64        # ミニバッチサイズ; Req. GPU Mem > AlexNet: 1 GB, ResNet18: 3GB,  ResNet50: Local 10GB
#batch_size0   = 128       # ミニバッチサイズ; Req. GPU Mem > AlexNet: 2 GB, ResNet18: 6GB,  ResNet50: Local 20GB, GitHub 15GB, ResNet152: GitHub 25GB
batch_size0   = 256        # ミニバッチサイズ; Req. GPU Mem > AlexNet: 3 GB, ResNet18: 12GB, ResNet50: Local 38GB, GitHub 30GB, ResNet152: GitHub 48GB
#batch_size0   = 512       # ミニバッチサイズ; Req. GPU Mem > AlexNet: 6 GB, ResNet18: 23GB, ResNet50: Local 49GB, GitHub 46GB
#batch_size0   = 1024      # ミニバッチサイズ; Req. GPU Mem > AlexNet: 12 GB, ResNet18: 46GB
#batch_size0   = 2048      # ミニバッチサイズ; Req. GPU Mem > AlexNet: 24 GB
device         = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自動検出: 学習用デバイス: CUDAドライバーが正常にインストールされていればGPU利用
#device        = 'cpu'     # 学習にはCPUを利用（cuda GPU がインストールされていないとき）
cpu_core0      = 4         # CPUのコア数, デフォルトで0, 0でも2でもほとんど変わらず, 2以上でmulti-process data loadingとなり処理が若干高速化される

########################
# 判別分析プロセスの実行
########################
######################################################################################################################################################################
# 判別分析結果の確率変数(AbnormalStateProbability)はその他(Miscellaneous)ディレクトリ内のファイル（'AbnormalStateProbability.mat', ファイルサイズ約???MB）に保存される
# 判別分析しない場合は、以前に分析した結果の確率変数(AbnormalStateProbability)をファイルから読み込む
######################################################################################################################################################################
perform_judge                  = 1  # 1 = 判別分析をする('AbnormalStateProbability.mat'ファイルを生成する), 0 = しない   ----------------------注意：これを 0 にするには、判別結果をダンプしておいてあとで読み込む必要あり！！！
delete_judge_images_before_gen = 0  # 1 = 判別分析用画像データを作成する前に既存の画像データを一括消去する, 0 = 一括消去しないで残す
#generate_judge_images         = 1  # 常に 1 とする: 1 = 新たな判別分析用画像データを作成する（作成する場合でも、既存のデータがあれば再作成しないでスキップ、もとのデータを残す）, 0 = 作成しない（既存のファイルは変更せずに判別分析のみ）

########################################
# 判別分析結果のプロットとプロットの保存
########################################
# 結果をグラフにプロットし、ファイルに保存する
plot_judge_results             = 1  # 1 = 結果をプロットし保存する, 0 = しない
# プロット開始、終了年月日の文字列
plot_starting_date0            = "20230501"  # プロット開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）を文字（ストリング）で指定
plot_ending_date0              = "20230630"  # プロット終了年月日"YYYYMMDD"（= 同年月日0時0分0秒）を文字（ストリング）で指定（注意！！！ ただし、下記に注意：プロットの終了時刻等を判別分析の終了時刻等に合わせる場合）
# プロットの終了時刻を最新時刻にする（判別分析の終了時刻に合わせる）場合
plot_upto_latest_time          = 1  # 1 = 上記のプロット終了年月日の設定にかかわらず観測データの最新時刻までプロットする, 0 = プロット終了時刻を上記の設定とする

##############
##############
## パスの設定
##############
##############
# 上部ディレクトリ （環境依存：環境に合わせて変更する必要あり）
Home_Path     = "/home/keisoku"                           # 現在のユーザーホームディレクトリ
#Home_Path    = "/home/gakusei"                           # 現在のユーザーホームディレクトリ（gakuseiの場合）
#Databox_Path = Home_Path + "/Dropbox"                    # 観測・解析データ収納用ディレクトリ
Databox_Path  = os.path.join(Home_Path, "Dropbox")        # 観測・解析データ収納用ディレクトリ
#Databox_Path = os.path.join(Home_Path, "Databox")        # 観測・解析データ収納用ディレクトリ（LandiskのDatabox内の観測データ、正常異常データ等を利用する場合）
print("Databox_Path = ", Databox_Path)
Analysis_Path = os.path.join(Databox_Path, "RadiowaveData_Analysis")  # データ解析・深層学習用トップディレクトリ（ほぼ変更の必要なし）
print("Analysis_Path = ", Analysis_Path)

# 中間ディレクトリ　（以下のディレクトリ構造はデータの格納場所に合わせる必要あり）
# 正常・異常データリスト収納ディレクトリ
#ListNormalAbnrml_Path  = os.path.join(Analysis_Path, "RadiowaveData_List_Normal_Abnormal_AbnLimitted")       # 正常・異常データリスト収納ディレクトリ（すでに作成済みリストあり）
#ListNormalAbnrml_Path   = os.path.join(Analysis_Path, "RadiowaveData_List_Normal_Abnormal_AbnOnly")          # 正常・異常データリスト収納ディレクトリ（すでに作成済みリストあり）
ListNormalAbnrml_Path   = os.path.join(Analysis_Path, "RadiowaveData_List_Normal_Abnormal_AbnOnly_Corrected") # 正常・異常データリスト収納ディレクトリ（すでに作成済みリストあり）津、新潟のみの修正版：後で試す
#ListNormalAbnrml_Path  = os.path.join(Analysis_Path, "RadiowaveData_List_Normal_Abnormal_NormalExtended")    # 正常・異常データリスト収納ディレクトリ（すでに作成済みリストあり）
# 等間隔平均サンプリング済み観測データファイル収納ディレクトリ
#ExtractedTimedata_Path = os.path.join(Analysis_Path, "Extracted_Timedata_dt2min")                            # 等間隔平均サンプリング済み観測データファイル収納ディレクトリ（別プログラムにより生成・随時更新）
#ExtractedTimedata_Path  = os.path.join(Analysis_Path, "Extracted_Timedata_dt2min_movingaverage30days")       # 等間隔平均サンプリング済み観測データファイル収納ディレクトリ（別プログラムにより生成・随時更新）
ExtractedTimedata_Path  = os.path.join(Analysis_Path, "Extracted_Timedata_dt2min_movingaverage20days")        # 等間隔平均サンプリング済み観測データファイル収納ディレクトリ（別プログラムにより生成・随時更新）

# ディレクトリ自動取得
# 現在のディレクトリ（スクリプト保存場所）
Scripts_Path = os.getcwd()
print("Current Scripts_Path = ",Scripts_Path)
# 現在の一つ上のディレクトリ（スペクトログラムその他のデータ保存用ディレクトリ）を自動的に取得しもとのディレクトリに戻る
os.chdir("..")  # 一つ上に移動
Spectrogram_DeepLearning_Path = os.getcwd()  # スクリプト保管場所の一つ上のディレクトリ（スカログラムその他のデータ保存用ディレクトリ）取得
os.chdir(Scripts_Path) # もと（スクリプト）のディレクトリに戻る
#print(os.getcwd())




""" ###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
City_DR_No_BCLoc_Freq = "Toyama_NE_11_Aomori_86p0MHz"  # Train Starting Date = 20150328, 173 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20150328, 203 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20150328, 471 Entries
#City_DR_No_BCLoc_Freq = "Toyama_SE_1_Iida_77p4MHz"     # Train Starting Date = 20150328, 49 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20190901, 109 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20190901, 185 Entries（注意：20190901_00:00から異常区間の設定）
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
# end if
#TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
TrainEndingDate0       = "20220730"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定(~/Dropbox/RadiowaveData_Analysis/RadiowaveData_List_Normal_Abnormal_AbnOnly_Corrected 使用の場合) 
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
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
#############################################################################
#############################################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
########################## """





""" ###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
#City_DR_No_BCLoc_Freq = "Toyama_NE_11_Aomori_86p0MHz"  # Train Starting Date = 20150328, 173 Entries
City_DR_No_BCLoc_Freq = "Toyama_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20150328, 203 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20150328, 471 Entries
#City_DR_No_BCLoc_Freq = "Toyama_SE_1_Iida_77p4MHz"     # Train Starting Date = 20150328, 49 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20190901, 109 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20190901, 185 Entries（注意：20190901_00:00から異常区間の設定）
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
# end if
TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
########################## """




###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
City_DR_No_BCLoc_Freq = "Toyama_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20150328, 471 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20190901, 185 Entries（注意：20190901_00:00から異常区間の設定）
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
# end if
#TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
TrainEndingDate0       = "20220730"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定(~/Dropbox/RadiowaveData_Analysis/RadiowaveData_List_Normal_Abnormal_AbnOnly_Corrected 使用の場合) 
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
##########################




""" ###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
#City_DR_No_BCLoc_Freq = "Toyama_NE_11_Aomori_86p0MHz"  # Train Starting Date = 20150328, 173 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20150328, 203 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20150328, 471 Entries
City_DR_No_BCLoc_Freq = "Toyama_SE_1_Iida_77p4MHz"     # Train Starting Date = 20150328, 49 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20190901, 109 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20190901, 185 Entries（注意：20190901_00:00から異常区間の設定）
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
# end if
TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
########################## """




""" ###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
#City_DR_No_BCLoc_Freq = "Toyama_NE_11_Aomori_86p0MHz"  # Train Starting Date = 20150328, 173 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20150328, 203 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20150328, 471 Entries
#City_DR_No_BCLoc_Freq = "Toyama_SE_1_Iida_77p4MHz"     # Train Starting Date = 20150328, 49 Entries
City_DR_No_BCLoc_Freq = "Yatsuo_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20190901, 109 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20190901, 185 Entries（注意：20190901_00:00から異常区間の設定）
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
# end if
TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
########################## """




""" ###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
#City_DR_No_BCLoc_Freq = "Toyama_NE_11_Aomori_86p0MHz"  # Train Starting Date = 20150328, 173 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20150328, 203 Entries
#City_DR_No_BCLoc_Freq = "Toyama_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20150328, 471 Entries
#City_DR_No_BCLoc_Freq = "Toyama_SE_1_Iida_77p4MHz"     # Train Starting Date = 20150328, 49 Entries
#City_DR_No_BCLoc_Freq = "Yatsuo_NE_4_SuzuV_81p9MHz"    # Train Starting Date = 20190901, 109 Entries
City_DR_No_BCLoc_Freq = "Yatsuo_NE_5_Niigata_82p3MHz"  # Train Starting Date = 20190901, 185 Entries（注意：20190901_00:00から異常区間の設定）
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
# end if
TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
########################## """




###########################################################################################################################################
# ラジオ波観測データ系列 （観測地_方位_順_放送地点_周波数） の指定：等間隔平均サンプリング済み観測データファイル名と一致させる
###########################################################################################################################################
City_DR_No_BCLoc_Freq = "Iwata_NW_2_Tsu_78p9MHz"       # Train Starting Date = 20170916, 308 Entries
######################
# 学習開始ー終了年月日
######################
if  (City_DR_No_BCLoc_Freq == "Toyama_NE_11_Aomori_86p0MHz" or City_DR_No_BCLoc_Freq == "Toyama_NE_4_SuzuV_81p9MHz" or 
     City_DR_No_BCLoc_Freq == "Toyama_NE_5_Niigata_82p3MHz" or City_DR_No_BCLoc_Freq == "Toyama_SE_1_Iida_77p4MHz"):
    TrainStartingDate0 = "20150328"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  富山観測データ
elif(City_DR_No_BCLoc_Freq == "Yatsuo_NE_4_SuzuV_81p9MHz"   or City_DR_No_BCLoc_Freq == "Yatsuo_NE_5_Niigata_82p3MHz"):
    TrainStartingDate0 = "20190831"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定　八尾観測データ：注意！学習開始時は正常区間と仮定してコード作成
elif(City_DR_No_BCLoc_Freq == "Iwata_NW_2_Tsu_78p9MHz"):
    TrainStartingDate0 = "20170916"  # 学習開始年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定  磐田観測データ
# end if
#TrainEndingDate0       = "20210901"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定
TrainEndingDate0       = "20220730"  # 学習終了年月日YYYYMMDD（= 同年月日0時0分0秒）をストリングで指定(~/Dropbox/RadiowaveData_Analysis/RadiowaveData_List_Normal_Abnormal_AbnOnly_Corrected 使用の場合) 
#####################################
# 判別分析開始年月日 = 学習終了年月日
#####################################
JudgePeriod_StartingDate0      = TrainEndingDate0  # 判別分析開始年月日"YYYYMMDD"（= 同年月日0時0分0秒）をストリングで指定（ただし、デフォルトで学習終了年月日とする）
# 判別分析終了年月日 = 時系列データの最新時刻
#############################################################################
## 関数 function 呼び出し ###################################################
#############################################################################
spectrogram_deeplearning (update_whole_spectrogram, delete_train_images_before_gen, generate_train_images, set_normal_state_as_non_abnrml,
                    generate_reference_images, analyze_actual_spectrogram, TrainStartingDate0, TrainEndingDate0, 
                    perform_train_and_varid, networkmodel, num_epochs0, LearningRate0, batch_size0, device, cpu_core0,
                    perform_judge, delete_judge_images_before_gen, JudgePeriod_StartingDate0, 
                    plot_judge_results, plot_starting_date0, plot_ending_date0, plot_upto_latest_time,
                    Home_Path, Databox_Path, Analysis_Path, ListNormalAbnrml_Path, ExtractedTimedata_Path,
                    Scripts_Path, Spectrogram_DeepLearning_Path, City_DR_No_BCLoc_Freq)
##########################
##########################
# 解析終了 End of Analysis
##########################
##########################




#############################
#############################
# 終了プロセス Ending Process
#############################
#############################
#
# 計算時間計測 Measure Computing Time with 'time' Object in Unix Time (Cumbersome)
#
ComputeEndTime = time.time()                                    # 終了時刻 （UNIX時間（エポック秒）を浮動小数点数float型で取得）
ComputeElapsedTime   = int(ComputeEndTime - ComputeStartTime)   # 秒単位で引き算
ComputeElapsedHour   = ComputeElapsedTime // 3600               # 秒を3600で割った商 ＝ 時
ComputeElapsedMinute = (ComputeElapsedTime % 3600) // 60        # その余りを60で割った商 ＝ 分
ComputeElapsedSecond = (ComputeElapsedTime % 3600 % 60)         # その余り ＝ 秒
print ()
print ("Time Elapsed for All Analyses 全解析に要した時間 (time): " + str(ComputeElapsedHour) + "h " + str(ComputeElapsedMinute).zfill(2) + "m " + str(ComputeElapsedSecond).zfill(2) + "s")
#
# Same with 'datetime' Object (Simple)
#
ComputeEndDateTime     = datetime.now()                             # 終了時刻 （datetime オブジェクトで取得）
ComputeElapsedDateTime = ComputeEndDateTime - ComputeStartDateTime  # DateTimeで引き算
#print (AnalysisElapsedDateTime)
print ("Time Elapsed for All Analyses 全解析に要した時間 (datetime): " + str(ComputeElapsedDateTime))

# Confirm Ending 終了確認
input("Program has Ended; Press Enter to Close Figures. 「Enter」を押すとプロットを閉じて終了")  # Keep plots till Enter is pressed: Enter を押すまでグラフが保持され、押すとグラフ表示が消えて終了

##########################
##########################
##########################
# END OF THE PROGRAM 終了
##########################
##########################
##########################
