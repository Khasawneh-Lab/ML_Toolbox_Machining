import numpy as np
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def EEEMD_Classification(feature_mat,cv, labels,param_tuning, feature_ranking, saving, *args):
    """
    Thsi function performs time series classification based on provided feature matrix.
    User is able to choose the option for tuning parameters or feature ranking by setting the 
    corresponding parameters. Classifications are performed using cross validation.

    Parameters
    ----------
    feature_mat : np.array
        Feature matrix (2D array) 
    cv : int
        k-fold cross validation
    labels : np.array
        Labels needed for classfication
    param_tuning : boolean
        It is set to True if user wants to tune parameter of the classifiers
    feature_ranking : boolean
        True if user wants to rank features and try different combinations
    saving : boolean
        It is set to True if user wants to save the results
    *args : 
        saving_path : str
            User can also give saving path of the results as input if saving is set to True.

    Returns
    -------
    clas_rep_test : dict
        Report of the classification results for test set.
    clas_rep_train : dict
        Report of the classification results for train set.

    """
    n_feature=7
    target_names = ['Chatter', 'Stable']
    
    # start timer
    start = time.time()
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
        clas_rep_test = np.zeros((n_feature,cv,4),dtype=object)
        clas_rep_train = np.zeros((n_feature,cv,4),dtype=object)
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
                
                X_train_new = np.zeros((len(X_train),n_feature))
                X_test_new = np.zeros((len(X_test),n_feature))
                
                # generate feature matrices based on ranking and their combinations
                
                for m in range (n_feature):
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
                
                # increment
                k=k+1
    end = time.time()
    print("Elapsed time for classification: {} seconds".format(end-start)) 
            
    return clas_rep_test,clas_rep_train