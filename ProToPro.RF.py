"""
==========================================================
Protein-To-Protein Random Forest
Features Include: MWCG + Secondary Structure Information + Global 
==========================================================
"""
print(__doc__)

import sys,os
import numpy as np
import pandas as pd
from random import *
import random
from sklearn.ensemble import RandomForestRegressor

Full_Data = pd.read_pickle('/mnt/home/bramerda/Documents/Persistent_Homology/Dataset')

if(sys.argv[3]=='Yes'):
  print("Including MWCG features...")
  fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/RF_ProToPro_MWCG'+sys.argv[2]+'.txt'
  feats = ['C_W0', 'C_W1', 'C_B0', 'C_B1', 'N_W0', 'N_W1', 'N_B0', 'N_B1', 'O_W0', 'O_W1', 'O_B0', 'O_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1', 'CC_1', 'CC_2', 'CC_3', 'CN_1', 'CN_2', 'CN_3', 'CO_1', 'CO_2', 'CO_3', 'Occupancy', 'Phi', 'Psi', 'Area', 'H', 'G', 'I', 'E', 'B', 'b', 'T',  'N', 'O', 'S', 'Res', '500', '750', '1000', '1500', '2000', '2500', '3000', '4000', '5000', '15000', 'ASN', 'ILE', 'GLN', 'ALA', 'ARG', 'GLY', 'MET', 'ASP', 'TYR', 'LEU', 'PRO', 'GLU', 'THR', 'TRP', 'LYS', 'VAL', 'SER', 'PHE', 'CYS', 'HIS', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value']
else:
  print("Not including MWCG features...")
  fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/RF_ProToPro'+sys.argv[2]+'.txt'
  feats = ['C_W0', 'C_W1', 'C_B0', 'C_B1', 'N_W0', 'N_W1', 'N_B0', 'N_B1', 'O_W0', 'O_W1', 'O_B0', 'O_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1', 'Occupancy', 'Phi', 'Psi', 'Area', 'H', 'G', 'I', 'E', 'B', 'b', 'T',  'N', 'O', 'S', 'Res', '500', '750', '1000', '1500', '2000', '2500', '3000', '4000', '5000', '15000', 'ASN', 'ILE', 'GLN', 'ALA', 'ARG', 'GLY', 'MET', 'ASP', 'TYR', 'LEU', 'PRO', 'GLU', 'THR', 'TRP', 'LYS', 'VAL', 'SER', 'PHE', 'CYS', 'HIS', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value']

test_set = Full_Data[Full_Data.Protein==sys.argv[1]] 

X_test = test_set[feats] #.astype('float')
y_test = test_set[['BF']]

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(X_test)
#exit()

train_set = Full_Data[Full_Data.Protein!=sys.argv[1]]

X_train = train_set[feats] #.to_numeric()
y_train = train_set[['BF']] #.to_numeric()

print(("data loaded for {0}").format(sys.argv[1]))

#import matplotlib
#from matplotlib import pyplot as plt

n_trees=int(sys.argv[2])
#njob = int(sys.argv[3])
#n_trees=i

print("Running Random Forest with {0} trees, target protein PDB ID: {1}, which has {2} residues. \n".format(n_trees,sys.argv[1],X_test.shape[0]) )

regr_rf = RandomForestRegressor(n_estimators=n_trees,random_state=2,verbose=1)#,n_jobs=njob) #,n_jobs=-1,oob_score = True,verbose=0)
y_train = y_train.values.ravel()
regr_rf.fit(X_train,y_train)

y_rf   = regr_rf.predict(X_test)
y_test = y_test.values
coef_1 = np.corrcoef(y_rf,y_test[:,0].ravel())[0][1] 
print("Pearson Correlation Coefficient: {0}".format(coef_1))

f = open(fname, 'w+')
f.write(("Protein: {0}\n").format(sys.argv[1]))
f.write(("Residues: {0}\n").format(X_test.shape[0]))
f.write(("Training_Set_Size: {0}\n").format(X_train.shape[0]))
f.write(("Trees: {0}\n").format(n_trees))
f.write(("Random Forest Result: \n"))
#f.write(("MWCG_+_Secondary_Features + AA + Type(CNOS) + Resolution + Protein Size + Local Density + R Value \n"))
#f.write(("Angle + MWCG_+ Area + + Local Density + R Value \n"))
f.write("Corr_Coef: {0}\n".format(coef_1))
f.write(("{0} \n").format(list(X_test)))
#f.write(("Result using only 10,000 random samples w random.seed(3) \n"))
f.close()


importances = regr_rf.feature_importances_
nkey=list(X_test)
#print(nkey[col])

def plot():

  font = {'family' : 'normal', 'weight' : 'normal','size'   : 10}
  matplotlib.rc('font', **font)
  figure(num=None, figsize=(9, 6.5), dpi=160, facecolor='w', edgecolor='k')
  
  fig = plt.figure(figsize=(9, 6.5), dpi=160)
  
  print(importances)
  # plot
  
  plt.bar(range(len(importances)), importances,align='center')
  
  plt.xticks(range(len(importances)),list(X_test),rotation=70)
  plt.xlabel('Features')
  plt.ylabel('Average Feature Importance')
  plt.axis('tight')
  plt.show()

#for f in range(X_test.shape[1]):
    #print("{0} feature {1} {2}".format(f , key[12+f], importances[f]))
np.save('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/importances_rf_special',importances)

def sum_importance():
  print("This needs to be updated for PH")
  exit()
  feat = [np.sum(importances[0:1]),importances[2],np.sum(importances[4:11]),np.sum(importances[11:20]),np.sum(importances[20:24]),importances[24],np.sum(importances[25:35]),np.sum(importances[35:55]),np  .sum(importances[55:58]),importances[58] ]
  feat_dict=['Angle','Area','Secondary','MWCG','Atom Type', 'Resolution','Protein Size','Amino Acid Type','Packing Density','R-Value']

  plt.bar(range(len(feat)), feat,align='center')
  #plt.xticks(range(len(feat)),feat_dict,rotation=70)
  plt.ylabel('Average Feature Importance')
  #plt.axis('tight')
  plt.show()

def feat_sel():
  #print sorted(zip(map(lambda x: round(x, 4), regr_rf.feature_importances_), key[12:]), reverse=True)
  #feat = sorted(zip(map(lambda x: round(x, 4), regr_rf.feature_importances_), key[12:]), reverse=True)
  feat = zip(map(lambda x: round(x, 4), regr_rf.feature_importances_), list(X_test))
  f = open('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/RF_Feat_Importance.txt', 'w+')
  f.write("{0} \n".format(sys.argv[1]))
  for i in range(0,len(feat)):
    f.write("{0} {1}\n".format(feat[i][0],feat[i][1]))
    #f.write("{0} {1}\n".format(i[1],i[0]) )
  f.close()
  exit()

#plot()
#sum_importance()
feat_sel()
exit()