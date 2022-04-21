from __future__ import print_function   # if you are using Python 2
import dionysus as d
import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import distance_matrix

# ---------------------------- Load pdb ------------------------------------------------------------

df_fpdb = open('/mnt/home/bramerda/Documents/Centrality/365/364_full/'+sys.argv[1]+'.pdb', "r")
df_pdb=[]

for line in df_fpdb:
  #if (line[:4]=='ATOM'):
  line = line[5:16] + " " + line[17:22] + " " + line[22:60]+ " "+line[60:81]
  arr=np.array(line.split())
  df_pdb.append(arr)
  
key=['Index','Atom_Type','Residue_Name','Residue_Chain','Residue_Number','X','Y','Z','Occupancy','B_Factor','Atom']
#key=['dum','Index','Atom_Type','Residue_Name','Residue_Chain','Residue_Number','X','Y','Z','Occupancy','B_Factor','Atom']

df_pdb = pd.DataFrame(df_pdb,columns=key)


print("   Calculating Distance Matrix...")
df_pdb[['X','Y','Z','B_Factor']]=df_pdb[['X','Y','Z','B_Factor']].apply(pd.to_numeric)

# ----------------------------- Generate Point Clouds ----------------------------------------------
n_kap  = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0,11.0]
eta    = [1.0,2.0,3.0,4.0,5.0,10.0,15.0,20.0]
n = len(eta)
m = 2*len(n_kap)
X_image = np.zeros((df_pdb[df_pdb.Atom_Type=='CA'].shape[0],n,m,3))


CA_pdb=df_pdb[df_pdb.Atom_Type=="CA"]
CA_pdb[['X','Y','Z','B_Factor']]=CA_pdb[['X','Y','Z','B_Factor']].apply(pd.to_numeric)
CA_pdb=CA_pdb.reset_index()

for CA in range(0,CA_pdb.shape[0]):
  element=["C","N","O"]
  for ele in range(0,3):
    
    CX_pdb=df_pdb[df_pdb.Atom==element[ele]]
    CX_pdb[['X','Y','Z','B_Factor']]=CX_pdb[['X','Y','Z','B_Factor']].apply(pd.to_numeric)
    CX_pdb=CX_pdb.reset_index()
    Loc = CX_pdb
    if(element[ele]!="C"):
      Glo=pd.concat([CA_pdb.iloc[[CA]],CX_pdb])
      remove_index=0
    else:
      Glo=CX_pdb
      remove_index=(Glo.index[Glo.Index==CA_pdb.Index[CA]])[0]
    
    dist=squareform(pdist(Glo[['X','Y','Z']],'euclidean'))
        
    for k in range(0,n):
      for L in range(0,m):
        kk=np.mod(L,m/2)
        Rips=np.empty((Glo.shape[0],Glo.shape[0]))    
    
        for i in range(0,Glo.shape[0]):
          for j in range(i,Glo.shape[0]):
            if(dist[i,j]<=11):
              if(i!=j):
                ker=np.exp(-1.00*((dist[i,j]/eta[kk])**(n_kap[L])));
                Rips[i,j] = Rips[i,j]-ker   
                Rips[j,i]=Rips[i,j]
        
        Rips_Glo  = d.fill_rips(squareform(Rips),1,1)
        Per_Glo = d.homology_persistence(Rips_Glo)
        dgms1 = d.init_diagrams(Per_Glo,Rips_Glo)
        Rips= np.delete(Rips, remove_index, 0)
        Rips= np.delete(Rips, remove_index, 1)
        
        Rips_Loc  = d.fill_rips(squareform(Rips),1,1)
        Per_Loc = d.homology_persistence(Rips_Loc)
        dgms2 = d.init_diagrams(Per_Loc,Rips_Loc)
        bdist = d.bottleneck_distance(dgms1[1], dgms2[1])
        print("Bottleneck distance between 1-dimensional persistence diagrams:", bdist)
        exit()
        #print(L,eta[k],n_kap[L])
        
X_image[CA,k,L,ele]    
    

  
exit()



dists = pdist(Loc,'euclidean')
Loc = squareform(dists)

Loc_Rips  = d.fill_rips(dists,1,8)          # Create Rips

print("sucess")
exit() 

Loc_P = d.homology_persistence(Loc_Rips) # Create Create Persistence Diagram


Glo_Rips = d.fill_rips(Glo)
Glo_P = d.homology_persistence(Glo_Rips)
Dgm = d.init_diagrams(Loc_P, Glo_P)
bdist = d.bottleneck_distance(dgms1[1], dgms2[1])

exit()

f1 = d.fill_rips(np.random.random((20, 2)), 2, 1)
m1 = d.homology_persistence(f1)
dgms1 = d.init_diagrams(m1, f1)
f2 = d.fill_rips(np.random.random((20, 2)), 2, 1)
m2 = d.homology_persistence(f2)
dgms2 = d.init_diagrams(m2, f2)
wdist = d.wasserstein_distance(dgms1[1], dgms2[1], q=2)
print("2-Wasserstein distance between 1-dimensional persistence diagrams:", wdist)

bdist = d.bottleneck_distance(dgms1[1], dgms2[1])
print("Bottleneck distance between 1-dimensional persistence diagrams:", bdist)


