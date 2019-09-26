'''
Date : 12/18/2018

Author : Melih Can Yesilli
 

Feature matrix for turning experiment data are computed and features are ranked with 
recursive feature elimination method. Combinations of features are used to classify
chatter / no chatter cases for the experiment

'''
import time
start2 = time.time()
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import skew,kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR,SVC,LinearSVC
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os,sys
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#%% Add paths------------------------------------------------------------------
folderToLoad1 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2inch',
                                    )
sys.path.insert(0,folderToLoad1)
os.path.join(folderToLoad1)

folderToLoad2 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2.5inch',
                                    )
sys.path.insert(0,folderToLoad2)
os.path.join(folderToLoad2)

folderToLoad3 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '3.5inch',
                                    )
sys.path.insert(0,folderToLoad3)
os.path.join(folderToLoad3)

folderToLoad4 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '4.5 inc',
                                    )
sys.path.insert(0,folderToLoad4)
os.path.join(folderToLoad4)


#%% Choose the case that you want to work on ----------------------------------
#for 2 inch case:
#namets = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"]
#label_c=np.full((20,1),1)
#label_u=np.full((19,1),0)
#label=np.concatenate((label_c,label_u))

#for 2.5 inch case:
#namets = ["c_570_014","c_570_015s","c_770_005","i_570_012","i_570_014","i_570_015s","i_770_005","s_570_003","s_570_005_2","s_570_005","s_570_008","s_570_010","s_570_015_2","s_570_015","s_770_002","s_770_005"]
#label_c=np.full((7,1),1)
#label_u=np.full((9,1),0)
#label=np.concatenate((label_c,label_u))

#for 3.5 inch case:
#namets=["c_1030_002","c_770_015","i_770_010","i_770_010_2","i_770_015","s_570_015","s_570_025","s_570_025_2","s_570_030","s_770_005","s_770_008","s_770_010","s_770_010_2","s_770_015"];
#label_c=np.full((5,1),1)
#label_u=np.full((9,1),0)
#label=np.concatenate((label_c,label_u))

#for 4.5 inch case:
namets=["c_570_035","c_570_040","c_1030_010","c_1030_015","c_1030_016","i_1030_010","i_1030_012","i_1030_013","i_1030_014","s_570_005","s_570_010","s_570_015","s_570_025","s_570_035","s_570_040","s_770_010","s_770_015","s_770_020","s_1030_005","s_1030_007","s_1030_013","s_1030_014"];
label_c=np.full((9,1),1)
label_u=np.full((13,1),0)
label=np.concatenate((label_c,label_u))

#%% Upload the Decompositions and compute the feature from them----------------
#name of datasets
numberofcase = len(namets) 

#load datasets
for i in range (0,numberofcase):
    name =  'chatter_%d' %(i+1)
    nameofdata = 'WPT_Recon_Level3_%s.mat' %(namets[i])
    pathofdata = os.path.join(folderToLoad4, nameofdata)
    exec("%s = sio.loadmat(pathofdata)" % (name))
    exec('%s = %s["recon"]' %(name,name))
    
#compute features

featuremat= np.zeros((numberofcase,10))
for i in range (0,numberofcase):
    name =  'chatter_%d' %(i+1)
    exec('ts=%s' %name)
    featuremat[i,0] = np.average(ts)
    featuremat[i,1] = np.std(ts)
    featuremat[i,2] = np.sqrt(np.mean(ts**2))   
    featuremat[i,3] = max(abs(ts))
    featuremat[i,4] = skew(ts)
    L=len(ts)
    featuremat[i,5] = sum(np.power(ts-featuremat[i,0],4)) / ((L-1)*np.power(featuremat[i,1],4))
    featuremat[i,6] = featuremat[i,3]/featuremat[i,2]
    featuremat[i,7] = featuremat[i,3]/np.power((np.average(np.sqrt(abs(ts)))),2)
    featuremat[i,8] = featuremat[i,2]/(np.average((abs(ts))))
    featuremat[i,9] = featuremat[i,3]/(np.average((abs(ts))))

#%% load frequency domain features (At different levels of WPT) and combine them 
#   with the time domain features

# Level 4
    
#2 inch case:
#freq_feature_data_name = 'Freq_Features_2inch'
#2.5 inch case:
#freq_feature_data_name = 'Freq_Features_2.5inch'
#3.5 inch case:
#freq_feature_data_name = 'Freq_Features_3.5inch'
#4.5 inch case:
#freq_feature_data_name = 'Freq_Features_4.5inch'

#Level 3
#2 inch case:
#freq_feature_data_name = 'Freq_Features_2inch_WPT_Level3'
#2.5 inch case:
#freq_feature_data_name = 'Freq_Features_2.5inch_WPT_Level3'
#3.5 inch case:
#freq_feature_data_name = 'Freq_Features_3.5inch_Level3'
#4.5 inch case:
freq_feature_data_name = 'Freq_Features_4.5inch_Level3'

#Level 2
#2 inch case:
#freq_feature_data_name = 'Freq_Features_2inch_WPT_Level2'
#2.5 inch case:
#freq_feature_data_name = 'Freq_Features_2.5inch_WPT_Level2'
#3.5 inch case:
#freq_feature_data_name = 'Freq_Features_3.5inch_Level2'
#4.5 inch case:
#freq_feature_data_name = 'Freq_Features_4.5inch_Level2'

#Level 1
#2 inch case:
#freq_feature_data_name = 'Freq_Features_2inch_WPT_Level1'
#2.5 inch case:
#freq_feature_data_name = 'Freq_Features_2.5inch_WPT_Level1'
#3.5 inch case:
#freq_feature_data_name = 'Freq_Features_3.5inch_Level1'
#4.5 inch case:
#freq_feature_data_name = 'Freq_Features_4.5inch_Level1'


pathofdata = os.path.join(folderToLoad4, freq_feature_data_name)        
freq_features = sio.loadmat(pathofdata)
freq_features = freq_features['Freq_Features']

#concatanate the frequency and time domain features 
featuremat = np.concatenate((featuremat, freq_features),axis = 1)

#%%
#creating train, test, accuracy, meanscore and deviation matrices
F_traincomb = np.zeros((int(numberofcase*0.66),14))
F_testcomb = np.zeros((int(numberofcase*0.33)+1,14))
accuracy1 = np.zeros((14,10))
accuracy2 = np.zeros((14,10))
deviation1 = np.zeros((14,1))
deviation2 = np.zeros((14,1))
meanscore1 = np.zeros((14,1))
meanscore2 = np.zeros((14,1))
duration1 = np.zeros((14,10))
meanduration = np.zeros((14,1))

#repeat the procedure ten times 
Rank=[]
RankedList=[]
for o in range(0,10):
    
    #split into test and train set
    F_train,F_test,Label_train,Label_test= train_test_split(featuremat,label, test_size=0.33)
    
    #classification
#    clf = SVC(kernel='linear')
#    clf = LogisticRegression()
#    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf = GradientBoostingClassifier()
    
    #recursive feature elimination
    selector = RFE(clf, 1, step=1)
    Label_train=np.ravel(Label_train)
    selector = selector.fit(F_train, Label_train)
    rank = selector.ranking_
    Rank.append(rank)
    rank = np.asarray(rank)
    
    #create a list that contains index number of ranked features
    rankedlist = np.zeros((14,1))
    

    
    #finding index of the ranked features and creating new training and test sets with respect to this ranking
    for m in range (1,15):
        k=np.where(rank==m)
        rankedlist[m-1]=k[0][0]
        F_traincomb[:,m-1] = F_train[:,int(rankedlist[m-1][0])]
        F_testcomb[:,m-1] = F_test[:,int(rankedlist[m-1][0])] 
    RankedList.append(rankedlist)
    
    #trying various combinations of ranked features such as ([1],[1,2],[1,2,3]...)
    for p in range(0,14): 
        start1 = time.time()
        clf.fit(F_traincomb[:,0:p+1],Label_train)
        score1=clf.score(F_testcomb[:,0:p+1],Label_test)
        score2=clf.score(F_traincomb[:,0:p+1],Label_train)
        accuracy1[p,o]=score1
        accuracy2[p,o]=score2
        end1=time.time()
        duration1[p,o] = end1 - start1


#computing mean score and deviation for each combination tried above        
for n in range(0,14):
    deviation1[n,0]=np.std(accuracy1[n,:])
    deviation2[n,0]=np.std(accuracy2[n,:])
    meanscore1[n,0]=np.mean(accuracy1[n,:])
    meanscore2[n,0]=np.mean(accuracy2[n,:])
    meanduration[n,0]=np.mean(duration1[n,:])
    

results = np.concatenate((meanscore1,deviation1,meanscore2,deviation2),axis=1)
results = 100*results    
#total duration for algorithm  
end2 = time.time()
duration2 = end2-start2
print('Total elapsed time: {}'.format(duration2))
        
    
#plotting mean accuracies and deviations

#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#line1 = ax1.plot(meanscore1,'b-',label=r'Test set score')
#line2 = ax1.plot(meanscore2,'g-',label=r'Training set score')
#line3 = ax2.plot(deviation1,'r--',label=r'Test set deviation')
#line4 = ax2.plot(deviation2,'c--',label=r'Training set deviation')
#lines = line1+line2+line3+line4
#labs = [l.get_label() for l in lines]
#ax1.legend(lines, labs, loc=1, fontsize = 38 )
#ax1.set_xlabel(r'Number of Features',fontsize = 38)
#ax1.set_ylabel(r'Score of Classification',fontsize =38)
#ax2.set_ylabel(r'Deviation',fontsize = 38)
#for tick in ax1.xaxis.get_major_ticks():
#    tick.label.set_fontsize(38) 
#for tick in ax1.yaxis.get_major_ticks():
#    tick.label.set_fontsize(38) 
#ax2.tick_params(labelsize = 38)
#plt.show()
#plt.savefig('Number_of_features_vs_deviation_accuracyWPT_4_5inch.pdf',bbox_inches = 'tight', dpi=300)

#how_many_times_rank = np.zeros((14,14))
#for i in range (0,14):
#    for j in range(0,10):
#        a = RankedList[j][i][0]
#        a = int(a)
#        how_many_times_rank[a,i]=how_many_times_rank[a,i]+1
#
#sio.savemat('number_of_times_feature_ranks_4.5inch_WPT_Level4.mat',mdict={'times_feature_rank':how_many_times_rank})