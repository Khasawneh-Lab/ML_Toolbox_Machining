
import time
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
from WPT_EEMD_ML.EEMD_Feature_Extraction import EEMD_IMF_Compute,EEMD_Feature_Compute


# Transfer learning application which trains on one dataset and test on another one
    
def EEMD_Transfer_Learning(data_paths,list_names,Decomps, info_IMFs, cv,param_tuning,feature_ranking,saving,*args):
    """
    

    Parameters
    ----------
    data_paths : list
        The list that contains the file directory where the experimental signals are stored. Algorithm will assume the first half of the paths provided in the list as the training set while the rest will be considered as the test set. This is also the same other inputs whose type is list.
    list_names : list
        The name of the txt files that contain the name of the data files for each case of cutting configurations.
    Decomps : list
        The list that includes the strings which represent whether the IMFs are computed for training and test sets. 'NA' means that IMFs for the corresponding data set is not available, while 'A' means available.
    info_IMFs : list
        The list of informative IMF numbers
    cv : int
        k-fold cross-validation.
    param_tuning : Boolean
        Set it to True if you would like tune hyperparameters of the classifiers
    feature_ranking : Boolean
        Set it to True if you would like to rank the features using RFE
    saving : boolean
        Set it to True if you would like to save the results
    *args : str
        Provide the path where the results will be saved using if the saving is set to True


    Returns
    -------
    clas_rep_test : dict
        Report of the classification results for test set.
    clas_rep_train : dict
        Report of the classification results for train set.

    """
    # get the path to data files from user
    start = time.time()
    
    feature_training =np.zeros((1,7))
    feature_test=np.zeros((1,7))
    label_train = np.zeros((1,))
    label_test = np.zeros((1,))
    n_features = 7
    
    for i,path in enumerate(data_paths):
        # compute feature matrices for training and testing cases
        list_name = list_names[i]
        EEMDecs = Decomps[i]
        p = info_IMFs[i]
        
        infoEMF,split_labels = EEMD_IMF_Compute(path,list_name, EEMDecs, False)
        labels = split_labels[:,2]
        feature_mat = EEMD_Feature_Compute(infoEMF,p)
       
        if i<len(data_paths)//2:
            feature_training = np.append(feature_training,feature_mat, axis=0)
            label_train = np.append(label_train,labels,axis=0)
        else:
            feature_test = np.append(feature_test,feature_mat, axis=0)
            label_test = np.append(label_test,labels,axis=0)           
    
    feature_training = feature_training[1:]
    feature_test = feature_test[1:]
    label_train = label_train[1:]
    label_test = label_test[1:]

    
    
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
        clas_rep_test = np.zeros((n_features,cv,4),dtype=object)
        clas_rep_train = np.zeros((n_features,cv,4),dtype=object)
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
        
        # generate lists that stores the indices of training and test sets
        train_index = []
        test_index = []
        
        # indices of training set
        for train_ind, test_ind in skf.split(feature_training, label_train):
            # obtain train and test set
            train_index.append((train_ind,test_ind))
            
        # indices of test set
        for train_ind, test_ind in skf.split(feature_test, label_test):
            # obtain train and test set
            test_index.append((train_ind,test_ind))   
            
        for k in range(cv):
            
            # generate the training set and test for training data set, but 
            # we only used the training part of the split
            X_train_train, X_train_test = feature_training[train_index[k][0]], feature_training[train_index[k][1]]
            y_train_train, y_train_test = label_train[train_index[k][0]], label_train[train_index[k][1]]             

            # generate the training set and test for test data set, but 
            # we only used the training part of the split
            X_test_train, X_test_test = feature_test[test_index[k][0]], feature_test[test_index[k][1]]
            y_test_train, y_test_test = label_test[test_index[k][0]], label_test[test_index[k][1]]  
    

            # standardize the training and test set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_train)
            X_test = scaler.transform(X_test_train)     
            
            # the labels
            y_train = y_train_train
            y_test = y_test_train
            
            if feature_ranking:
                # recursive feature elimination
                selector = RFE(model, n_features_to_select=1, step=1)
                selector = selector.fit(X_train, y_train)
                rank = selector.ranking_
                Rank.append(rank)
                rank = np.asarray(rank)
                
                X_train_new = np.zeros((len(X_train),n_features))
                X_test_new = np.zeros((len(X_test),n_features))
                
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
                        saving_path = args[0]
                        if param_tuning:
                            save_name = 'EEMD_CR_Test_PT_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                            
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_test,f)
                            f.close()
                        
                            save_name = 'EEMD_CR_Train_PT_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                    
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_train,f)
                            f.close()
                        else:
                            save_name = 'EEMD_CR_Test_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                            
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_test,f)
                            f.close()
                        
                            save_name = 'EEMD_CR_Train_Combn_'+str(m+1)+'_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                    
                            f = open(saving_path+'\\'+save_name+".pkl","wb")
                            pickle.dump(cr_train,f)
                            f.close()
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
                    saving_path = args[0]
                    if param_tuning:
                        save_name = 'EEMD_CR_Test_PT_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                        
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_test,f)
                        f.close()
                    
                        save_name = 'EEMD_CR_Train_PT_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_train,f)
                        f.close()
                    else:
                        save_name = 'EEMD_CR_Test_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                        
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_test,f)
                        f.close()
                    
                        save_name = 'EEMD_CR_Train_Classifier_'+str(i+1)+'_RunNumber_'+str(k+1)
                
                        f = open(saving_path+'\\'+save_name+".pkl","wb")
                        pickle.dump(cr_train,f)
                        f.close()            
    end = time.time()
    print("Elapsed time for classification: {} seconds".format(end-start)) 
            
    return clas_rep_test,clas_rep_train                
                        
            
