"""
==========================================================
Adds Secondary Structure Information and 9 Element Specific FRI Kernels(MWCG)
==========================================================
"""
print(__doc__)

import sys,os
import numpy as np
import pandas as pd
import time

df_fpdb = open('/mnt/home/bramerda/Documents/Persistent_Homology/Test/1yzm.pdb', "r")
df_pdb=[]

for line in df_fpdb:
  if (line[:4]=='ATOM'):
    line = line[5:16] + " " + line[17:22] + " " + line[22:60]+ " "+line[60:81]
    arr=np.array(line.split())
    df_pdb.append(arr)
  
key=['Index','Atom_Type','Residue_Name','Residue_Chain','Residue_Number','X','Y','Z','Occupancy','B_Factor','Atom']

df_pdb = pd.DataFrame(df_pdb,columns=key)

df_pdb = df_pdb[df_pdb['Atom_Type']=='CA']
#df_pdb = df_pdb[(df_pdb['Atom_Type']=='CA')|(df_pdb['Atom']=='N')]
#df_pdb = df_pdb[(df_pdb['Atom_Type']=='CA')|(df_pdb['Atom']=='O')]
#print(df_pdb.head())
#exit()
df_pdb = df_pdb.reset_index(drop=True)
#print(df_pdb.head())

def dist_mat(df_pdb,eta):
  #df_data = df_pdb[df_pdb.Atom != 'H'].reset_index(drop=True)
  df_data=df_pdb
  #df_data = pd.DataFrame(data,columns=key)
  df_data[['X','Y','Z','B_Factor']]=df_data[['X','Y','Z','B_Factor']].apply(pd.to_numeric)
  dist=np.empty((df_data.shape[0],df_data.shape[0]))
  
  print("   Calculating Distance Matrix for eta = {}...".format(eta))
  
  a_m = np.array((df_data.X[:],df_data.Y[:],df_data.Z[:]))
  
  length = df_data.shape[0]
  
  for i in range(0,length):
    for j in range(i+1,length):
      dst       = np.linalg.norm(a_m[:,i]-a_m[:,j])
      dist[i,j] = 1-np.exp(-1.00*((dst/float(eta))**2));
    dist[i,i]=0  
  W = np.maximum( dist, dist.transpose() )

  np.save('/mnt/home/bramerda/Documents/Persistent_Homology/Test/CC_Distance_Matrix_k2_e'+str(eta)+'.npy', W)
  np.savetxt('/mnt/home/bramerda/Documents/Persistent_Homology/Test/CC_Distance_Matrix_k2_e'+str(eta)+'.txt', W)

#if(not(os.path.isfile('/mnt/home/bramerda/Documents/Perstent_Homology/Test/Distance_Matrix.npy'))):
#  dist_mat(df_pdb)

dist_mat(df_pdb,2)
dist_mat(df_pdb,6)
dist_mat(df_pdb,16)


dist_mat_2  = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Test/CC_Distance_Matrix_k2_e2.npy')
dist_mat_6  = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Test/CC_Distance_Matrix_k2_e6.npy')
dist_mat_16 = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Test/CC_Distance_Matrix_k2_e16.npy')

print(dist_mat_2[0:3,0:3])
exit()

pdb[['X','Y','Z','B_Factor']]=pdb[['X','Y','Z','B_Factor']].apply(pd.to_numeric)

n = pdb.shape[0]


CC_df = pdb[pdb.Atom=='C' & pdb.Atom=='N']
print(CC_df.head())
exit()

CC = np.zeros((CC_df.shape[0],CC_df.shape[0]))
CN = np.zeros((CN_df.shape[0],CN_df.shape[0]))
CO = np.zeros((CO_df.shape[0],CO_df.shape[0]))


print(CC.shape)
print(CN.shape)
print(CO.shape)
exit()

print("starting loop...") 

sigma_1 = 3.0;
sigma_2 = 1.0;
sigma_3 = 1.0;
e_1     = 16.0;
e_2     = 2.0;
e_3     = 31.0;  

#def FRI():
for i in range(0,pdb.shape[0]):
  print(i,pdb.shape[0])
  for j in range(0,pdb.shape[0]):
    if(j!=i):
      dist = dist_mat[i,j]
      Kernel_l = float(1.00)/(float(1.00)+(dist/e_1)**(sigma_1));   
      Kernel_e = float(1.00)/(float(1.00)+(dist/e_2)**(sigma_2));   
      Kernel_e2 = np.exp(-1.00*((dist/e_3)**(sigma_3)));
      if (pdb.Atom[j]=='C'):
        CC[i,0] = CC[i,0]-Kernel_l
        CC[i,1] = CC[i,1]-Kernel_e
        CC[i,2] = CC[i,2]-Kernel_e2
      elif (pdb.Atom[j]=='N'):
        CN[i,0] = CN[i,0]-Kernel_l
        CN[i,1] = CN[i,1]-Kernel_e
        CN[i,2] = CN[i,2]-Kernel_e2
      elif (pdb.Atom[j]=='O'):
        CO[i,0] = CO[i,0]-Kernel_l
        CO[i,1] = CO[i,1]-Kernel_e
        CO[i,2] = CO[i,2]-Kernel_e2



pdb['CC_1']=CC[:,0]
pdb['CC_2']=CC[:,1]
pdb['CC_3']=CC[:,2]
pdb['CN_1']=CN[:,0]
pdb['CN_2']=CN[:,1]
pdb['CN_3']=CN[:,2]
pdb['CO_1']=CO[:,0]
pdb['CO_2']=CO[:,1]
pdb['CO_3']=CO[:,2]

del pdb['index']



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#t1 = time.time()
#total = t1-t0
#print(total)
print('fini')
pdb.to_pickle('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+sys.argv[1]+'/Secondary_FRI')
np.save('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+sys.argv[1]+'/Secondary_FRI', pdb)


