import numpy as np
import pandas as pd
import sys,os
import glob
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


fname='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/Image/'
sname='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/Image'
kname='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/Image_BF'

tmp_RF = pd.read_pickle('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+sys.argv[1]+'/Secondary_FRI_b')
#['Index','Atom_Type','Residue_Name','Residue_Chain','Residue_Number']
CA_df=tmp_RF[tmp_RF.Atom_Type=="CA"]
CA_df =CA_df.reset_index(drop=True)

#CA_df.loc[CA_df.Index == '10'].index[0]

names=glob.glob(fname+'C_*')
Nnames=glob.glob(fname+'N_*')
Onames=glob.glob(fname+'O_*')
#print(len(names),len(Nnames),len(Onames))

if not(os.path.isfile(sname+".npy")):

  if(len(names)==CA_df.shape[0] and len(Nnames)==CA_df.shape[0] and len(Onames)==CA_df.shape[0]):
  #if(len(names)==len(Nnames) and len(Nnames)==len(Onames)):
  
    print("All files accounted for, generating image data.")
  
    cols=range(1,11)
    #size=len(names)
    size=CA_df.shape[0]
    image=np.empty([size,8,10,3])
    image_bf = np.empty([size,2])
    
    #for i in names:
    for i in range(0,CA_df.shape[0]):
    
      i= CA_df.Index[i]
      #i=(i.split("_")[2]).split(".")[0]
      Cname=fname+'C_'+i+'.txt'
      Nname=fname+'N_'+i+'.txt'
      Oname=fname+'O_'+i+'.txt'
      if os.path.isfile(Cname) and os.path.isfile(Nname) and os.path.isfile(Oname): #If file exits load file, scale data, combine images
        C=np.loadtxt(Cname,skiprows=1,usecols=cols)
        scaler.fit(C)
        C=scaler.transform(C)
        N=np.loadtxt(Nname,skiprows=1,usecols=cols)
        scaler.fit(N)
        N=scaler.transform(N)
        #print(Oname)
        O=np.loadtxt(Oname,skiprows=1,usecols=cols)
        scaler.fit(O)
        O=scaler.transform(O)
    
        j=CA_df.loc[CA_df.Index == i].index[0]
        image[j,:,:,0]=C
        image[j,:,:,1]=N
        image[j,:,:,2]=O
        image_bf[j,0] = i
        #bf_series=CA_df.loc[CA_df.Index == i].B_Factor.reset_index(drop=True) # special for 2q52 and 2q4n
        #image_bf[j,1] =bf_series[0] # special for 2q52 and 2q4n
        
        image_bf[j,1] = CA_df.loc[CA_df.Index == i].B_Factor
        
      else:
        print('{0} missing data'.format(i))
        
    np.save(sname,image,allow_pickle=True)
    np.save(kname,image_bf,allow_pickle=True)
  
  else:
    print("{0} missing data files, image data not compiled.".format(sys.argv[1]))
    exit()
  
  
  print(CA_df.shape,image.shape,image_bf.shape)

else:
  image=np.load(sname+".npy")
  if(CA_df.shape[0]!=image.shape[0]):
    print(CA_df.shape,image.shape)
  a=2

 