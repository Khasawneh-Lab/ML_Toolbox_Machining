
# import libraries 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import os
from matplotlib import rc

matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)  

def plot_results(res_path,param_tuning,feature_ranking,n_feature,methods,clsf_names,cv,layout,ylabel_index):
    
    '''
    This function performs classification with respect to feature matrix generated with respect to informative wavelet packet
 
    :param str (res_path): The file path where classification results are saved
    
    :param boolean (param_tuning): True if parameter tuning is used, otherwise it is set to False
    
    :param boolean (feature_ranking): True if feature ranking is applied, otherwise it is set to False
 
    :param int (n_feature): Number of features used in classification
    
    :param str (res_path): The file path where classification results are saved
    
    :param list (methods): List of methods whose results will be plotted
    
    :param list (clsf_names): Name of classifiers
    
    :param list (label_name): Name of the .npy file which includes labels of time series
    
    :param int (cv): k-fold stratified cross validation

    :param list (layout): Layout for subplot (e.g. [2,2])
    
    :param int array (ylabel_index): Index of subplots which will have a y-axis label
  
    :Returns:
        :fig:
            Plot shows the classification results. 
            
    :Example:            
        .. doctest::    
            
      
        from WPT_EEMD_ML.Plot_Results import plot_results
        
        # inputs
        
        methods = ['WPT']  
        clsf_names = ['SVM','LR','RF','GB']
        cv = 5
        layout = [1,1]
        ylabel_index = np.array([1])
        res_path = 'D:\\Repositories\\WPT_EEMD_ML_Machining\\test\\WPT_Output\\'
        param_tuning = False
        feature_ranking = True
        n_feature = 14
        
        fig = plot_results(methods,clsf_names,cv,layout,ylabel_index,res_path,param_tuning,feature_ranking,n_feature)
        
    '''
    
    # fontsize for the figure
    fs=20
    
    # arrays for run number and number of classifiers
    run_number = np.linspace(1,cv,cv,dtype=int)
    Classifier = np.linspace(1,len(clsf_names),len(clsf_names),dtype=int)
    
    # classification report paths
    path=[]
    path.append(res_path)
    
    # add paths
    foldersToLoad = {}
    for i in range(len(path)):
        foldersToLoad[i] = os.path.join(path[i]) 
    
    
    # load the results -------------------------------------------------------
    if feature_ranking:
        
        # generate the matrices which store the metrics for classification 
        metrics_train=np.zeros((len(run_number),len(Classifier),7,n_feature,len(methods)))
        metrics_test=np.zeros((len(run_number),len(Classifier),7,n_feature,len(methods)))
        
        # import all reports
        cr_test = np.empty((len(Classifier),len(run_number),n_feature),dtype=object)
        cr_train = np.empty((len(Classifier),len(run_number),n_feature),dtype=object)
        
        for m,mtd in enumerate(methods):
            for i in run_number:
                for j in Classifier:
                    for k in range(n_feature):
                        if param_tuning:
                            name_test = mtd+'_CR_Test_PT_Combn_'+str(k+1)+'_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
                            name_train = mtd+'_CR_Train_PT_Combn_'+str(k+1)+'_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
                        else:
                            name_test = mtd+'_CR_Test_Combn_'+str(k+1)+'_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
                            name_train = mtd+'_CR_Train_Combn_'+str(k+1)+'_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
       
                        with open(foldersToLoad[m]+name_test, 'rb') as f:
                            cr_test[j-1,i-1,k] = pickle.load(f)
                        with open(foldersToLoad[m]+name_train, 'rb') as f:
                            cr_train[j-1,i-1,k] = pickle.load(f)        
                        
                        # overall matrics - training set
                        metrics_test[i-1,j-1,0,k,m]= cr_test[j-1,i-1,k]['accuracy']            
                        metrics_test[i-1,j-1,1,k,m]= cr_test[j-1,i-1,k]['macro avg']['f1-score']  
                        metrics_test[i-1,j-1,2,k,m]= cr_test[j-1,i-1,k]['macro avg']['precision']   
                        metrics_test[i-1,j-1,3,k,m]= cr_test[j-1,i-1,k]['macro avg']['recall'] 
                        metrics_test[i-1,j-1,4,k,m]= cr_test[j-1,i-1,k]['weighted avg']['f1-score'] 
                        metrics_test[i-1,j-1,5,k,m]= cr_test[j-1,i-1,k]['weighted avg']['precision'] 
                        metrics_test[i-1,j-1,6,k,m]= cr_test[j-1,i-1,k]['weighted avg']['recall'] 
                        
                        # overall matrics - test_set
                        metrics_train[i-1,j-1,0,k,m]= cr_train[j-1,i-1,k]['accuracy']            
                        metrics_train[i-1,j-1,1,k,m]= cr_train[j-1,i-1,k]['macro avg']['f1-score']  
                        metrics_train[i-1,j-1,2,k,m]= cr_train[j-1,i-1,k]['macro avg']['precision']   
                        metrics_train[i-1,j-1,3,k,m]= cr_train[j-1,i-1,k]['macro avg']['recall'] 
                        metrics_train[i-1,j-1,4,k,m]= cr_train[j-1,i-1,k]['weighted avg']['f1-score'] 
                        metrics_train[i-1,j-1,5,k,m]= cr_train[j-1,i-1,k]['weighted avg']['precision'] 
                        metrics_train[i-1,j-1,6,k,m]= cr_train[j-1,i-1,k]['weighted avg']['recall']  
    else:
        
        metrics_train=np.zeros((len(run_number),len(Classifier),7,len(methods)))
        metrics_test=np.zeros((len(run_number),len(Classifier),7,len(methods)))
        
        # import all reports
        cr_test = np.empty((len(Classifier),len(run_number)),dtype=object)
        cr_train = np.empty((len(Classifier),len(run_number)),dtype=object)
        
        for m,mtd in enumerate(methods):
            for i in run_number:
                for j in Classifier:
                    if param_tuning:
                            name_test = mtd+'_CR_Test_PT_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
                            name_train = mtd+'_CR_Train_PT_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
                    else:
                        for k in range(n_feature):
                            name_test = mtd+'_CR_Test_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
                            name_train = mtd+'_CR_Train_Classifier_'+str(j)+'_RunNumber_'+str(i)+'.pkl'
       
                    with open(foldersToLoad[m]+name_test, 'rb') as f:
                        cr_test[j-1,i-1] = pickle.load(f)
                    with open(foldersToLoad[m]+name_train, 'rb') as f:
                        cr_train[j-1,i-1] = pickle.load(f)        
                    
                    # overall matrics - training set
                    metrics_test[i-1,j-1,0,m]= cr_test[j-1,i-1]['accuracy']            
                    metrics_test[i-1,j-1,1,m]= cr_test[j-1,i-1]['macro avg']['f1-score']  
                    metrics_test[i-1,j-1,2,m]= cr_test[j-1,i-1]['macro avg']['precision']   
                    metrics_test[i-1,j-1,3,m]= cr_test[j-1,i-1]['macro avg']['recall'] 
                    metrics_test[i-1,j-1,4,m]= cr_test[j-1,i-1]['weighted avg']['f1-score'] 
                    metrics_test[i-1,j-1,5,m]= cr_test[j-1,i-1]['weighted avg']['precision'] 
                    metrics_test[i-1,j-1,6,m]= cr_test[j-1,i-1]['weighted avg']['recall'] 
                    
                    # overall matrics - test_set
                    metrics_train[i-1,j-1,0,m]= cr_train[j-1,i-1]['accuracy']            
                    metrics_train[i-1,j-1,1,m]= cr_train[j-1,i-1]['macro avg']['f1-score']  
                    metrics_train[i-1,j-1,2,m]= cr_train[j-1,i-1]['macro avg']['precision']   
                    metrics_train[i-1,j-1,3,m]= cr_train[j-1,i-1]['macro avg']['recall'] 
                    metrics_train[i-1,j-1,4,m]= cr_train[j-1,i-1]['weighted avg']['f1-score'] 
                    metrics_train[i-1,j-1,5,m]= cr_train[j-1,i-1]['weighted avg']['precision'] 
                    metrics_train[i-1,j-1,6,m]= cr_train[j-1,i-1]['weighted avg']['recall']  
                
    
    # plotting----------------------------------------------------------------
    
    # green_diamond = dict(markerfacecolor='g', marker='D')
    if feature_ranking:
        for m,mtd in enumerate(methods):
            plt.figure(figsize=[16,10])
            for j in Classifier:
                plt.subplot(2,2,j)
                   
                ax=plt.gca()
                ax.set_title(clsf_names[j-1],fontsize=fs)
               
                # train_position = [1.2, 2.9, 4.6, 6.3]
                meanpointprops = dict(marker='P', markeredgecolor='firebrick',markerfacecolor='firebrick')
                
                bp1 = ax.boxplot(metrics_test[:,j-1,0,:,m],showfliers=True,patch_artist=True,showmeans =True,meanprops=meanpointprops )
                bps = [bp1]
                    
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(15)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(15)
                    
                if np.any(j-1==ylabel_index-1):
                    ax.set_ylabel('Score',fontsize=fs)
                if j==3 or j==4:
                    ax.set_xlabel('Number of features',fontsize=fs)    
                ax.ticklabel_format(axis='y',style='sci')
                ax.tick_params(axis='both', labelsize=fs-2)
                
    
                face_colors= ['#1b9e77','sandybrown']
                outline_color = ['#7570b3','k']
                
                
                
                
                boxes = []
                for i,bp in enumerate(bps):
                    for box in bp['boxes']:
                        # change outline color
                        box.set( color=face_colors[i], linewidth=2)
                        # change fill color
                        box.set( facecolor = face_colors[i] )
                    
                    ## change color and linewidth of the whiskers
                    for whisker in bp['whiskers']:
                        whisker.set(color=face_colors[i], linewidth=1.5)
                    
                    ## change color and linewidth of the caps
                    for cap in bp['caps']:
                        cap.set(color=face_colors[i], linewidth=1.5)
                    
                    ## change color and linewidth of the medians
                    for median in bp['medians']:
                        median.set(color='k', linewidth=2)
                    
                    ## change the style of fliers and their fill
                    for flier in bp['fliers']:
                        flier.set(marker='D',markerfacecolor=face_colors[i], alpha=0.8)
                    boxes.append(bp['boxes'][0])
                boxes.append(bp['means'][0])
                    
                    # for mean in bp['means']:
                    #     mean.set(marker="X",markerfacecolor='r',alpha=0.5)
                if m==0:    
                    fig=plt.gcf()
                    if layout!=[1,1]:
                        ax.legend(boxes, [r'Test Set', r'Training Set',r'Means'], loc='best',ncol=1, prop={'size': fs}) 
                        ax.legend(boxes, [r'Test Set',r'Means'], loc='best',ncol=1, prop={'size': fs}) 
                    else:
                        ax.legend(boxes, [r'Test Set', r'Training Set',r'Means'], loc='best',ncol=1, prop={'size': fs}) 
                        ax.legend(boxes, [r'Test Set',r'Means'], loc='best',ncol=1, prop={'size': fs})    
        # save_name = 'file_path'+'figure_name.png'
        # plt.savefig(save_name,bbox_inches = 'tight', dpi=600)
    else:   
        plt.figure(figsize=[10,7])
        for m,mtd in enumerate(methods):

            plt.subplot(layout[0],layout[1],m+1)

         
            ax=plt.gca()
            ax.set_title(mtd,fontsize=fs)
        
            test_positions = [0, 1.2, 2.4, 3.6]    
            train_position = [0.6, 1.8, 3, 4.2]
            
            meanpointprops = dict(marker='P', markeredgecolor='firebrick',markerfacecolor='firebrick')
            
            bp1 = ax.boxplot(metrics_test[:,:,0,m],showfliers=True,patch_artist=True,positions = test_positions,showmeans =True,meanprops=meanpointprops )
            bp2 = ax.boxplot(metrics_train[:,:,0,m], showfliers=True,patch_artist=True,positions = train_position, showmeans =True ,meanprops=meanpointprops )
            bps = [bp1, bp2]
                
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
                
            if np.any(m==ylabel_index-1):
                ax.set_ylabel('Score',fontsize=fs)
            ax.ticklabel_format(axis='y',style='sci')
            ax.tick_params(axis='both', labelsize=fs-2)
            
            plt.xticks(test_positions,clsf_names)
            plt.xticks([x / 2 for x in [x + y for x, y in zip(test_positions, train_position)]],['SVM','LR','RF','GB'])
            face_colors= ['#1b9e77','sandybrown']
            outline_color = ['#7570b3','k']
            
            
            
            
            boxes = []
            for i,bp in enumerate(bps):
                for box in bp['boxes']:
                    # change outline color
                    box.set( color=face_colors[i], linewidth=2)
                    # change fill color
                    box.set( facecolor = face_colors[i] )
                
                ## change color and linewidth of the whiskers
                for whisker in bp['whiskers']:
                    whisker.set(color=face_colors[i], linewidth=1.5)
                
                ## change color and linewidth of the caps
                for cap in bp['caps']:
                    cap.set(color=face_colors[i], linewidth=1.5)
                
                ## change color and linewidth of the medians
                for median in bp['medians']:
                    median.set(color='k', linewidth=2)
                
                ## change the style of fliers and their fill
                for flier in bp['fliers']:
                    flier.set(marker='D',markerfacecolor=face_colors[i], alpha=0.8)
                boxes.append(bp['boxes'][0])
            boxes.append(bp['means'][0])
                
                # for mean in bp['means']:
                #     mean.set(marker="X",markerfacecolor='r',alpha=0.5)
            if m==0:    
                if layout!=[1,1]:
                    ax.legend(boxes, [r'Test Set', r'Training Set',r'Means'], loc='best',ncol=1, prop={'size': fs}) 
                else:
                    ax.legend(boxes, [r'Test Set', r'Training Set',r'Means'], loc='best',ncol=1, prop={'size': fs}) 
                
        # save_name = 'file_path'+'figure_name.png'
        # plt.savefig(save_name,bbox_inches = 'tight', dpi=600)
    fig = plt.gcf()
    return fig

if __name__ == "__main__":
    
    methods = ['WPT']  
    clsf_names = ['SVM','LR','RF','GB']
    cv = 5
    layout = [1,1]
    ylabel_index = np.array([1])
    res_path = 'D:\\Repositories\\WPT_EEMD_ML_Machining\\test\\WPT_Output\\'
    param_tuning = True
    feature_ranking = False
    n_feature = 14
    
    fig = plot_results(res_path,param_tuning,feature_ranking,n_feature,methods,clsf_names,cv,layout,ylabel_index)
