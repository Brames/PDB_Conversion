#Call R to generate persistent homology data

import sys, os, glob
import pandas as pd
import numpy as np
import pprint
from pandas import *
import subprocess

im_location='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/'
location='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/BD_Image/'

#subprocess.call("rm "+location+"*.csv",shell=True)

def check():
  if(os.path.isfile(im_location+'Dif_Im.npy') and os.path.isfile(im_location+'Glo_Im.npy')):
    print("Images already exist.")
    exit()

#check()  
for element in ['C','N','O']:
  print("Generating ",element," BD persistence")
  subprocess.call("Rscript /mnt/home/bramerda/Documents/Persistent_Homology/Local_Persistence.R "+element+" "+sys.argv[1],shell=True)
    
Lnames=glob.glob('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/BD_Image/Loc_CC*.csv')
Gnames=glob.glob('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/BD_Image/Glo_CC*.csv')

num_CA=len(Lnames)

Glo_BD_Image=np.empty((num_CA,3,48,3))
Loc_BD_Image=np.empty((num_CA,3,48,3))

# For each CA we take the row sum of the persistence diagram which contains 48 bins
# We consider Birth, Death, and Persistence images as well as C, N, and O channels
#    B - Birth
#    D - Death
#    P - Persistence

counter=0

for ele in ['C','N','O']:
  CA =0
  print(ele)
  #create images for element  
  temp_names=glob.glob('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/BD_Image/Glo_C'+ele+'*.csv')
  temp_names.sort()

  Loc_temp_names=glob.glob('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/BD_Image/Loc_C'+ele+'*.csv')
  Loc_temp_names.sort()  

  for i in temp_names:
    k='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/BD_Image/Loc_C'+i[-16:]
    temp_df=pd.read_csv(i)
    tmp_One=temp_df[temp_df.dimension==1]
    tmp_One=tmp_One.reset_index()
    
    Loc_temp_df=pd.read_csv(k)
    Loc_tmp_One=Loc_temp_df[Loc_temp_df.dimension==1]
    Loc_tmp_One=Loc_tmp_One.reset_index()

    tmp_img_B=np.zeros((tmp_One.shape[0],48))
    tmp_img_D=np.zeros((tmp_One.shape[0],48))
    tmp_img_P=np.zeros((tmp_One.shape[0],48))
     
    Loc_tmp_img_B=np.zeros((Loc_tmp_One.shape[0],48))
    Loc_tmp_img_D=np.zeros((Loc_tmp_One.shape[0],48))
    Loc_tmp_img_P=np.zeros((Loc_tmp_One.shape[0],48))
    
    for j in range(0,tmp_One.shape[0]):
      tmp_B=int(tmp_One.Birth[j]*48) # Birth
      tmp_D=int(tmp_One.Death[j]*48) # Death    
      
      tmp_img_B[j,tmp_B]=1
      tmp_img_D[j,tmp_D]=1
      tmp_img_P[j,tmp_B:tmp_D]=1

    for L in range(0,Loc_tmp_One.shape[0]):
      Loc_tmp_B=int(Loc_tmp_One.Birth[L]*48) # Birth
      Loc_tmp_D=int(Loc_tmp_One.Death[L]*48) # Death
      
      Loc_tmp_img_B[L,Loc_tmp_B]=1
      Loc_tmp_img_D[L,Loc_tmp_D]=1
      Loc_tmp_img_P[L,Loc_tmp_B:Loc_tmp_D]=1
    
    Glo_BD_Image[CA,0,:,counter]=np.sum(tmp_img_B,axis=0)
    Glo_BD_Image[CA,1,:,counter]=np.sum(tmp_img_D,axis=0)
    Glo_BD_Image[CA,2,:,counter]=np.sum(tmp_img_P,axis=0)

    Loc_BD_Image[CA,0,:,counter]=np.sum(Loc_tmp_img_B,axis=0)
    Loc_BD_Image[CA,1,:,counter]=np.sum(Loc_tmp_img_D,axis=0)
    Loc_BD_Image[CA,2,:,counter]=np.sum(Loc_tmp_img_P,axis=0)

    #Difference of Images
    Loc_BD_Image[CA,0,:,counter]=Glo_BD_Image[CA,0,:,counter]-Loc_BD_Image[CA,0,:,counter]
    Loc_BD_Image[CA,1,:,counter]=Glo_BD_Image[CA,1,:,counter]-Loc_BD_Image[CA,1,:,counter]
    Loc_BD_Image[CA,2,:,counter]=Glo_BD_Image[CA,2,:,counter]-Loc_BD_Image[CA,2,:,counter]
    
    CA=CA+1 
  
  counter=counter+1

print("Saving images...")
np.save(im_location+'Dif_Im_11.npy',Loc_BD_Image)
np.save(im_location+'Glo_Im_11.npy',Glo_BD_Image)

print("Cleaning up temp files...")
subprocess.call("rm "+location+"*.csv",shell=True)

exit()