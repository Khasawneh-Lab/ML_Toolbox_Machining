'''
Author : Melih Can Yesilli

Date : 12/21/2018 

Description : This code takes time series belongs to turning experiment and applies 
EEMD to this data to generate IMFs. IMFs are used for feature extraction for machine
learning. Informative IMFs are selected by using Fisher Discriminant Ratio(FDR) feature
ranking method. Top ranked features are selected to be used in feature matrix for whole cases. 

Turning experiment data includes 2, 2.5, 3.5 and 4.5 inch stickout cases with different number of stable,
intermediate chatter and chatter state time series. This code is able to apply above mentioned method
for whole cases.

'''

import numpy as np
import pandas as pd
import scipy.io as sio

#------------------------------------------------------------------------------
#for 2 inch case:
namets = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"]
label_c=np.full((20,1),1)
label_u=np.full((19,1),0)
label=np.concatenate((label_c,label_u))

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
#namets=["c_570_035","c_570_040","c_1030_010","c_1030_015","c_1030_016","i_1030_010","i_1030_012","i_1030_013","i_1030_014","s_570_005","s_570_010","s_570_015","s_570_025","s_570_035","s_570_040","s_770_010","s_770_015","s_770_020","s_1030_005","s_1030_007","s_1030_013","s_1030_014"];
#label_c=np.full((9,1),1)
#label_u=np.full((13,1),0)
#label=np.concatenate((label_c,label_u))

#------------------------------------------------------------------------------

#name of datasets
numberofcase = len(namets) 

#load datasets
for i in range (0,numberofcase):
    name =  'chatter_%d' %(i+1)
    nameofdata = '%s_downsampled.mat' %(namets[i])
    exec("%s = sio.loadmat(nameofdata)" % (name))
    exec('%s = %s["tsDS"]' %(name,name))

#Ensemble Emprical Mode Decomposition (EEMD)
from PyEMD import EEMD

eemd = EEMD()
emd = eemd.EMD
emd.trials = 200      #default = 100
emd.noise_width = 0.2 #default = 0.05

#signal
S = chatter_1[:,1]
t = chatter_1[:,0]

eIMFs = eemd.eemd(S, t)
nIMFs = eIMFs.shape[0]

