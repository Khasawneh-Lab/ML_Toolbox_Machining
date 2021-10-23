import numpy as np
import time
import pickle
from WPT_EEMD_ML.WPT_Feature_Extraction import WPT_Feature_Extraction
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def WPT_Classification(data_path,list_name,label_name,WF,L,IWP,fs,cv,param_tuning,feature_ranking,saving,saving_path,*args):
    
    '''
    This function performs classification with respect to feature matrix generated with respect to informative wavelet packet
    
    :param str (data_path): Path to folder where list of time series names and time series data is kept
    
    :param str (list_name): Name of the .txt file which includes names of time series data
    
    :param list (label_name): Name of the .npy file which includes labels of time series
    
    :param str (WF): Wavelet function (see types_ of wavelet functions available in PyWavelet package)

    :param int (L): Transformation level 
    
    :param int (IWP): Informative Wavelet Packet Number (IWP)
  
    :param int (fs): Sampling frequency
    
    :param int (cv): k-fold cross validation. "cv" represents the k.
    
    :param boolean (param_tuning): It is set to true if user wants to apply parameter tuning.
    
    :param boolean (feature_ranking): It is set to true if user wants to apply recursive feature elimination.
  
    :param boolean (saving): It is set to 'True' to save the reconstructed signals
    
    :param str (saving_path): The path where user wants to save classification reports.
    
    :param list (*args): Additional parameters such as saving file name for the reconstructed signals if saving is set to true.
            
    :Returns:
        :clas_rep_test:
            Object array that includes classification reports for each classification. If feature selection flag is set to true, it will have a shape of (14,cv,4). 
            Otherwise, the shape is (cv,4).
            
        :time:
            Time elapsed during the reconstruction of the signals
    :Example:            
        .. doctest::    
            
      
        from WPT_EEMD_ML.WPT_Classification import WPT_Classification
        
        # inputs
        
        data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
        list_name = 'time_series_name_2inch.txt'
        WF = 'db10'
        L=4
        IWP = 3
        label_name = '2_inch_Labels_2Class.npy'
        saving = True
        fs = 10000 
        param_tuning = False
        feature_ranking =True
        cv = 5
        saving_path = 'D:\\Repositories\\WPT_EEMD_ML_Machining\\test\\WPT_Output'
        save_name = "Reconstructions"
        
        classification_reports = WPT_Classification(data_path,list_name,label_name,WF,L,IWP,fs,cv,param_tuning,feature_ranking,saving,saving_path,save_name)   
    
    '''
    # start timer
    start = time.time()
    
    # feature matrix
    if saving:
        feature_mat,labels = WPT_Feature_Extraction(data_path, list_name,label_name,WF,L,IWP,fs,saving,args[0])
    else:
        feature_mat,labels = WPT_Feature_Extraction(data_path, list_name,label_name,WF,L,IWP,fs,saving)
        
    # Classification ---------------------------------------------------------
  
    target_names = ['Chatter', 'Stable']
    
    # make a list of the parameters to tune
    if param_tuning:
        # set ranges for the hyperparameters of the classifiers
        gamma_range = np.logspace(-3, 3, num=10)
        lambda_range = np.logspace(-3, 3, num=10)
        max_depth_range  = np.linspace(2,8,4)
        n_estimators_range = np.linspace(100,200,5)
        n_estimators_range = n_estimators_range.astype(int)
        
        params = [] # the list that contains the paramters to tune for each classifier
        if feature_ranking:
            params.append({'C': lambda_range, 'kernel':['linear'],'gamma':gamma_range}) # SVM paramters
        else:
            params.append({'C': lambda_range, 'kernel': ('rbf','sigmoid'),'gamma':gamma_range}) # SVM paramters
        params.append({'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')})
        params.append({'criterion':('gini', 'entropy'),'n_estimators':n_estimators_range,'max_depth':max_depth_range})
        params.append({'n_estimators':n_estimators_range,'criterion':('friedman_mse', 'mse')})
    
    # list of classifiers
    clf = []
    if feature_ranking:
        Rank = []      #list of rankings
        clas_rep_test = np.zeros((14,cv,4),dtype=object)
        clas_rep_train = np.zeros((14,cv,4),dtype=object)
        clf.append(SVC(kernel='linear'))
    else:
        clas_rep_test = np.zeros((cv,4),dtype=object)
        clas_rep_train = np.zeros((cv,4),dtype=object)        
        clf.append(SVC())
    clf.append(LogisticRegression())
    clf.append(RandomForestClassifier())
    clf.append(GradientBoostingClassifier())            
    

    for i, model in enumerate(clf):
        print("classifier: {}".format(i))
        # cross validation split
        skf= StratifiedKFold(n_splits=cv, shuffle =True, random_state=35) # set a constant random_state number to be able to get same train-test for all classifiers
        k=0
        for train_ind, test_ind in skf.split(feature_mat, labels):
            # obtain train and test set
            X_train, X_test = feature_mat[train_ind], feature_mat[test_ind]
            y_train, y_test = labels[train_ind], labels[test_ind] 
            
            # standardize the training and test set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  
             
            if feature_ranking:
                # recursive feature elimination
                selector = RFE(model, n_features_to_select=1, step=1)
                selector = selector.fit(X_train, y_train)
                rank = selector.ranking_
                Rank.append(rank)
                rank = np.asarray(rank)
                
                X_train_new = np.zeros((len(X_train),14))
                X_test_new = np.zeros((len(X_test),14))
                
                # generate feature matrices based on ranking and their combinations
                
                for m in range (14):
                    print("combination:{}".format(m+1))
                    r=np.where(rank==m+1)[0][0]
                    X_train_new[:,m] = X_train[:,r]
                    X_test_new[:,m] = X_test[:,r]                
                
                    if param_tuning:
                        # paramter tuning
                        param_tune = GridSearchCV(model, params[i])
                        # use one half of the training set to tune the parameters
                        param_tune.fit(X_train_new[:,0:m+1],y_train)
                        best_params = param_tune.best_params_
                        for key in sorted(best_params.keys()):
                            setattr(model,key,best_params[key])

                    #retrain/train the model
                    model.fit(X_train_new[:,0:m+1],y_train) 
                    
                    # predicted labels
                    predicted_labels_test = model.predict(X_test_new[:,0:m+1])   
                    predicted_labels_train = model.predict(X_train_new[:,0:m+1])                    

                    #classification report
                    cr_test = classification_report(y_test, predicted_labels_test, target_names=target_names,output_dict=True)
                    cr_train = classification_report(y_train, predicted_labels_train, target_names=target_names,output_dict=True)
                    
                    #store reports in the object array 
                    clas_rep_test[m,k,i] = cr_test
                    clas_rep_train[m,k,i] = cr_train
                    
                    if saving:
                        if param_tuning:
                            save_name = 'WPT_CR_Test_PT_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                            
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_test,f)
                            f.close()
                        
                            save_name = 'WPT_CR_Train_PT_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                    
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_train,f)
                            f.close()
                        else:
                            save_name = 'WPT_CR_Test_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                            
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_test,f)
                            f.close()
                        
                            save_name = 'WPT_CR_Train_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                    
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_train,f)
                            f.close()
                k=k+1
            else:
                if param_tuning:
                    # paramter tuning
                    param_tune = GridSearchCV(model, params[i])
                    # use one half of the training set to tune the parameters
                    param_tune.fit(X_train,y_train)
                    best_params = param_tune.best_params_
                    for key in sorted(best_params.keys()):
                        setattr(model,key,best_params[key])
                        
                #retrain/train the model
                model.fit(X_train,y_train)
                
                # predicted labels
                predicted_labels_test = model.predict(X_test)   
                predicted_labels_train = model.predict(X_train)
            
                #classification report
                cr_test = classification_report(y_test, predicted_labels_test, target_names=target_names,output_dict=True)
                cr_train = classification_report(y_train, predicted_labels_train, target_names=target_names,output_dict=True)
                    
                #store reports in the object array 
                clas_rep_test[k,i] = cr_test
                clas_rep_train[k,i] = cr_train                  
                
                
                if saving:
                    if param_tuning:
                        save_name = 'WPT_CR_Test_PT_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                        
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_test,f)
                        f.close()
                    
                        save_name = 'WPT_CR_Train_PT_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_train,f)
                        f.close()
                    else:
                        save_name = 'WPT_CR_Test_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                        
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_test,f)
                        f.close()
                    
                        save_name = 'WPT_CR_Train_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_train,f)
                        f.close()            
                
                # increment
                k=k+1
    end = time.time()
    print("Elapsed time for classification: {} seconds".format(end-start)) 
            
    return clas_rep_test,clas_rep_train



