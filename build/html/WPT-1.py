from WPT_Feature_Extraction import WPT_Feature_Extraction
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#parameters

stickout_length='2'
WPT_Level = 4
Classifier = 'SVC'
plotting = True

results = WPT_Feature_Extraction(stickout_length, WPT_Level, Classifier, plotting)     