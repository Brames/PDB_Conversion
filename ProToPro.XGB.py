"""
=============================================================================
This takes 'Dataset' file which contains both persistent homology and other 
protein features and uses that data to predict cross protein B-factor for an
individual protein using boosted gradient regression

Enter 'python ProToPro.XG.py pdbname feature_set N_est LR N'

pdbname: 4 character pdbid e.g. 1aba
feature_set: 'Yes' for all features including MWCG
             'Sp' for only selected features using feature selection from RF
             No for all features except MWCG
                          
N_est: Number of estimators
LR:    Learning Rate
N: version or iteration
             
============================================================================
"""
print(__doc__)

import sys,os
import numpy as np
from random import *
import random
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Perforing grid search

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print("Modules loaded, loading dataset...")

Full_Data = pd.read_pickle('/mnt/home/bramerda/Documents/Persistent_Homology/Dataset_CA')

print("Data loaded, starting ensemble...")
#print(list(Full_Data))
#print(Full_Data.head())
#print(Full_Data.tail())
#print(Full_Data.shape)
#exit()

if(sys.argv[2]=='Yes'):
  print("Including MWCG features...")
  fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/[CA]XGB_ProToPro_MWCG'+sys.argv[3]+'_'+sys.argv[4]+'_'+sys.argv[5]+'.txt'
  feats = ['Cexp_W0', 'Cexp_W1', 'Cexp_B0', 'Cexp_B1', 'Nexp_W0', 'Nexp_W1', 'Nexp_B0', 'Nexp_B1', 'Oexp_W0', 'Oexp_W1', 'Oexp_B0', 'Oexp_B1', 'CLor_W0', 'CLor_W1', 'CLor_B0', 'CLor_B1', 'NLor_W0', 'NLor_W1', 'NLor_B0', 'NLor_B1', 'OLor_W0', 'OLor_W1', 'OLor_B0', 'OLor_B1', 'Ce_W0', 'Ce_W1', 'Ce_B0', 'Ce_B1', 'Ne_W0', 'Ne_W1', 'Ne_B0', 'Ne_B1', 'Oe_W0', 'Oe_W1', 'Oe_B0', 'Oe_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1', 'Occupancy', 'Phi', 'Psi', 'Area', 'H', 'G', 'I', 'E', 'B', 'b', 'T', 'C', 'CC_1', 'CC_2', 'CC_3', 'CN_1', 'CN_2', 'CN_3', 'CO_1', 'CO_2', 'CO_3', 'Res', '500', '750', '1000', '1500', '2000', '2500', '3000', '4000', '5000', '15000', 'ASN', 'ILE', 'GLN', 'ALA', 'ARG', 'GLY', 'MET', 'ASP', 'TYR', 'LEU', 'PRO', 'GLU', 'THR', 'TRP', 'LYS', 'VAL', 'SER', 'PHE', 'CYS', 'HIS', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value'] 
  print(feats)
elif(sys.argv[2]=='Sp'):
  print("Including selected features...")
  fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/[CA]XGB_ProToPro_Spec'+sys.argv[3]+'_'+sys.argv[4]+'_'+sys.argv[5]+'.txt'
  feats = ['C_W0', 'C_W1', 'C_B0', 'C_B1', 'N_W0', 'N_W1', 'N_B0', 'N_B1', 'O_W0', 'O_W1', 'O_B0', 'O_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1','Cexp_W0', 'Cexp_W1', 'Cexp_B0', 'Cexp_B1', 'Nexp_W0', 'Nexp_W1', 'Nexp_B0', 'Nexp_B1', 'Oexp_W0', 'Oexp_W1', 'Oexp_B0', 'Oexp_B1',
'CLor_W0', 'CLor_W1', 'CLor_B0', 'CLor_B1', 'NLor_W0', 'NLor_W1', 'NLor_B0', 'NLor_B1', 'OLor_W0', 'OLor_W1', 'OLor_B0', 'OLor_B1', 'CC_1', 'CC_2', 'CC_3', 'CN_1', 'CN_2', 'CN_3', 'CO_1', 'CO_2', 'CO_3', 'Phi', 'Psi', 'Area', 'Res', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value']
  print(feats)
else:
  print("Not including MWCG features...")
  fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/[CA]No_XGB_ProToPro'+sys.argv[3]+'_'+sys.argv[4]+'_'+sys.argv[5]+'.txt'
  feats = ['Cexp_W0', 'Cexp_W1', 'Cexp_B0', 'Cexp_B1', 'Nexp_W0', 'Nexp_W1', 'Nexp_B0', 'Nexp_B1', 'Oexp_W0', 'Oexp_W1', 'Oexp_B0', 'Oexp_B1', 'CLor_W0', 'CLor_W1', 'CLor_B0', 'CLor_B1', 'NLor_W0', 'NLor_W1', 'NLor_B0', 'NLor_B1', 'OLor_W0', 'OLor_W1', 'OLor_B0', 'OLor_B1', 'Ce_W0', 'Ce_W1', 'Ce_B0', 'Ce_B1', 'Ne_W0', 'Ne_W1', 'Ne_B0', 'Ne_B1', 'Oe_W0', 'Oe_W1', 'Oe_B0', 'Oe_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1', 'Occupancy', 'Phi', 'Psi', 'Area', 'H', 'G', 'I', 'E', 'B', 'b', 'T',  'N', 'O', 'S', 'Res', '500', '750', '1000', '1500', '2000', '2500', '3000', '4000', '5000', '15000', 'ASN', 'ILE', 'GLN', 'ALA', 'ARG', 'GLY', 'MET', 'ASP', 'TYR', 'LEU', 'PRO', 'GLU', 'THR', 'TRP', 'LYS', 'VAL', 'SER', 'PHE', 'CYS', 'HIS', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value']
  print(feats)

test_set = Full_Data[Full_Data.Protein==sys.argv[1]]

#--------------------- include for CA only ----------------------------
#test_set = test_set[test_set.Type_x=='CA']
#----------------------------------------------------------------------

 
X_test = test_set[feats] #.astype('float')
y_test = test_set[['BF']]

#print(test_set.head())
#print(test_set.shape)
#print(y_test.head())
#print(y_test.shape)


#print(list(X_test))

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(X_test)
#exit()

train_set = Full_Data[Full_Data.Protein!=sys.argv[1]]

X_train = train_set[feats] #.to_numeric()
y_train = train_set[['BF']] #.to_numeric()

print(("data loaded for {0}").format(sys.argv[1]))
#---------------------------- Grid Search --------------------------------------------

# k-fold cross validation evaluation of xgboost model
# CV model

def cross_val(n_est,lr,loss_f):
  model = GradientBoostingRegressor()
  kfold = KFold(n_splits=10, random_state=7)
  results = cross_val_score(model, X_train, y_train, cv=kfold)
  print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

  parameter_candidates = [n_est,lr,loss_f]

# Create a classifier object with the classifier and parameter candidates
  clf = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on data1's feature and target data
  clf.fit(X_train, y_train)

  print('Best N Estimator:',clf.best_estimator_.n_estimators)
  print('Best learning rate:',clf.best_estimator_.learning_rate) 
  exit()

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

n_est = {'n_estimators': [100,500,1000]}
lr = {'learning_rate': [0.5,0.1,0.05]}
lss = ['ls','quantile','lad','huber']
loss_f = {'loss': lss}

#cross_val(n_est,lr,loss_f)
#exit()

alpha = 0.975 
nest  = int(sys.argv[3]) #1000
learn = float(sys.argv[4]) 

print("Running XG Boost using \n pdb id {0} \n which contains {1} residues with {2} estimators.".format(sys.argv[1],X_test.shape[0],nest) )

LS = lss[1]

clf = GradientBoostingRegressor(loss=LS,alpha=alpha, 
                                n_estimators=nest, max_depth=4,
                                learning_rate=learn, min_samples_leaf=9,
                                min_samples_split=9, verbose=1)

print("Fitting...")
y_train = y_train.values.ravel()
clf.fit(X_train, y_train)
y_clf =clf.predict(X_test)
y_test = y_test.values


finalCC=np.hstack((y_test.reshape((y_test.shape[0],1)),y_clf.reshape((y_test.shape[0],1))))

#pred_CA = np.float64(model.predict([X_image_test_CA,X_test_CA]).ravel())
#y_test_CA = np.float64(y_test_CA)
#CC_CA = np.corrcoef(pred_CA,y_test_CA)
np.savetxt('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/GBT_CA_pred_'+sys.argv[3]+'_'+sys.argv[5]+'.txt',finalCC)


coef_1 = np.corrcoef(y_clf,y_test[:,0].ravel())[0][1]
#coef_1 = np.corrcoef(y_clf,y_test)[0][1]
print(("Alpha: {0}, Learning Rate: {1}, N: {2}, CC: {3}").format(alpha,learn,nest,coef_1))
#print(coef_1)

f = open(fname, 'w+')
f.write(("Protein: {0}\n").format(sys.argv[1]))
f.write(("Residues: {0}\n").format(X_test.shape[0]))
f.write(("Training_Set_Size: {0}\n").format(X_train.shape[0]))
f.write(("Estimators: {0}\n").format(nest))
f.write(("XG_Boost_Result: \n"))
#f.write(("MWCG_+_Secondary_Featrues + AA + Type(CNOS) + Resolution + Protein Size + Local Density + R Value \n"))
f.write("Corr_Coef: {0}\n".format(coef_1))
f.write(("{0} \n").format(list(X_test)))
#f.write(("Result using only 10,000 random samples w random.seed(3) \n"))
f.write(("Learning Rate: {0} \n").format(learn))
f.write(("Loss: {0} \n").format(LS))
f.close()

exit()
