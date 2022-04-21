import subprocess
import sys,os
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

if os.path.isdir('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/Image'):
  #print('directory exists')
  a=2
else:
  subprocess.call('mkdir /mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/Image',shell=True)

tmp_RF = pd.read_pickle('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+sys.argv[1]+'/Secondary_FRI_b')
#['Index','Atom_Type','Residue_Name','Residue_Chain','Residue_Number']
tmp_RF['Atom_Type'] = tmp_RF['Atom_Type'].str.decode('utf-8')
tmp_RF['Index'] = tmp_RF['Index'].str.decode('utf-8') 

CA_df=tmp_RF[tmp_RF.Atom_Type=="CA"]

#print(CA_df.head())
#print(CA_df.Index.iloc[1])
#print(CA_df.shape)
#exit()

#loop through CA in parallel
  #if image doesn't exist run Rscript

def Rscript(i_CA,element):
  if os.path.isfile('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/Image/'+element+'_'+str(CA_df.Index.iloc[i_CA])+'.txt'):
    #print("{1} {0} exists".format(CA_df.Index.iloc[i_CA],element))
    a = 2
  else:    
    print("{2} {1} {0} does not exist".format(CA_df.Index.iloc[i_CA],element,sys.argv[1]))
    #if(sys.argv[2]=='Run'):
    subprocess.call("Rscript /mnt/home/bramerda/Documents/Persistent_Homology/Image_Local.R "+element+" "+sys.argv[1]+" 8 "+str(CA_df.Index.iloc[i_CA]),shell=True)
    
  #print(i_CA)

inputs=range(0,CA_df.shape[0])

num_cores = multiprocessing.cpu_count()-1 
#print("{0} cores".format(num_cores))  

results = Parallel(n_jobs=num_cores)(delayed(Rscript)(i,sys.argv[2]) for i in inputs)

#print("Success!")



#result = subprocess.run(["Rscript /mnt/home/bramerda/Documents/Persistent_Homology/Local.R C 1akg 4"], stdout=subprocess.PIPE)
#result = subprocess.run(["Rscript", "/mnt/home/bramerda/Documents/Persistent_Homology/Image_Local.R","C","1akg","4"], capture_output=True)
#print(result.stdout)




#print(check)


print('fini')