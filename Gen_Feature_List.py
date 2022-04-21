"""
This collects the persistent homology data and combines it with previously generated protein features

File is saved as pickle "Dataset"
"""
print(__doc__)

import sys, os.path
import pandas as pd
import glob
import numpy as np
import time

#h=sys.argv[1]

#tmp_RF = pd.read_pickle('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+h+'/Secondary_FRI_b')
#print(list(tmp_RF))
#tmp_RF=tmp_RF.rename(columns={15000: '15000'})
#tmp_RF.rename(columns={'kkk': '15000'})
#print(list(tmp_RF))
#tmp_RF.to_pickle('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+h+'/Secondary_FRI_b')
#exit()
#------------------- For CA Only --------------------------------------
#tmp_RF = tmp_RF[tmp_RF.Atom_Type=='CA']
#print(list(tmp_RF))
#----------------------------------------------------------------------
#tmp_RF = tmp_RF.rename(columns={'Atom_Type': 'Type','B_Factor':'BF'})

#C_e = pd.read_csv('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+h+'/Local_CC12_exp_bd_p6.csv')

#for i in range(0,C_e.shape[0]):
#  print(tmp_RF.Index.iloc[i],C_e.Index.iloc[i])

#temp = pd.merge(tmp_RF,C_e,on=['X','Y','Z','BF'],how='outer')
#print(tmp_RF.shape[0],C_e.shape[0])
#exit()

names = glob.glob('/mnt/home/bramerda/Documents/Persistent_Homology/Results/*')
names.sort()
#names.reverse()

fname='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'
name_list=['1ob4','1ob7','2olx','3md5','1nko','2oct','3fva']
for i in name_list:
  names.remove(fname+i)
 
#names.remove('/mnt/home/bramerda/Documents/Persistent_Homology/Results/4axy') 
count = 0
counter=0
rf_counter=0

key=['Index','Type','X','Y','Z','BF','Atom'] 

def add_protein(i,count):
  global PH,counter,rf_counter
  C_e = pd.read_csv(i+'/Local_CC12.csv')
  N_e = pd.read_csv(i+'/Local_CN12.csv')
  O_e = pd.read_csv(i+'/Local_CO12.csv')
  
  C_e = C_e.rename(columns={'Was_0': 'Cexp_W0', 'Was_1': 'Cexp_W1','Btl_0': 'Cexp_B0', 'Btl_1': 'Cexp_B1'})
  N_e = N_e.rename(columns={'Was_0': 'Nexp_W0', 'Was_1': 'Nexp_W1','Btl_0': 'Nexp_B0', 'Btl_1': 'Nexp_B1'})
  O_e = O_e.rename(columns={'Was_0': 'Oexp_W0', 'Was_1': 'Oexp_W1','Btl_0': 'Oexp_B0', 'Btl_1': 'Oexp_B1'})
  
  if(C_e.shape[0]!=N_e.shape[0] or C_e.shape[0]!=O_e.shape[0]):
    print("Local CC12 issue") 
  C_l = pd.read_csv(i+'/Local_CC12_Lor.csv')
  N_l = pd.read_csv(i+'/Local_CN12_Lor.csv')
  O_l = pd.read_csv(i+'/Local_CO12_Lor.csv')
  
  C_l = C_l.rename(columns={'Was_0': 'CLor_W0', 'Was_1': 'CLor_W1','Btl_0': 'CLor_B0', 'Btl_1': 'CLor_B1'})
  N_l = N_l.rename(columns={'Was_0': 'NLor_W0', 'Was_1': 'NLor_W1','Btl_0': 'NLor_B0', 'Btl_1': 'NLor_B1'})
  O_l = O_l.rename(columns={'Was_0': 'OLor_W0', 'Was_1': 'OLor_W1','Btl_0': 'OLor_B0', 'Btl_1': 'OLor_B1'})    
  
  if(C_l.shape[0]!=N_l.shape[0] or C_l.shape[0]!=O_l.shape[0]):
    print("Local CC12_Lor issue")
  
  C_exp = pd.read_csv(i+'/Local_CC12_exp_bd.csv')
  N_exp = pd.read_csv(i+'/Local_CN12_exp_bd.csv')
  O_exp = pd.read_csv(i+'/Local_CO12_exp_bd.csv')
  
  C_exp = C_exp.rename(columns={'Was_0': 'Ce_W0', 'Was_1': 'Ce_W1','Btl_0': 'Ce_B0', 'Btl_1': 'Ce_B1'})
  N_exp = N_exp.rename(columns={'Was_0': 'Ne_W0', 'Was_1': 'Ne_W1','Btl_0': 'Ne_B0', 'Btl_1': 'Ne_B1'})
  O_exp = O_exp.rename(columns={'Was_0': 'Oe_W0', 'Was_1': 'Oe_W1','Btl_0': 'Oe_B0', 'Btl_1': 'Oe_B1'})

  if(C_exp.shape[0]!=N_exp.shape[0] or C_exp.shape[0]!=O_exp.shape[0]):
    print("Local CC12_exp_bd issue")

  C_bd_3 = pd.read_csv(i+'/Local_CC12_exp_bd_p3.csv')
  N_bd_3 = pd.read_csv(i+'/Local_CN12_exp_bd_p3.csv')
  O_bd_3 = pd.read_csv(i+'/Local_CO12_exp_bd_p3.csv')

  C_bd_3 = C_bd_3.rename(columns={'Was_0': 'C3_W0', 'Was_1': 'C3_W1','Btl_0': 'C3_B0', 'Btl_1': 'C3_B1'})
  N_bd_3 = N_bd_3.rename(columns={'Was_0': 'N3_W0', 'Was_1': 'N3_W1','Btl_0': 'N3_B0', 'Btl_1': 'N3_B1'})
  O_bd_3 = O_bd_3.rename(columns={'Was_0': 'O3_W0', 'Was_1': 'O3_W1','Btl_0': 'O3_B0', 'Btl_1': 'O3_B1'})

  if(C_bd_3.shape[0]!=N_bd_3.shape[0] or C_bd_3.shape[0]!=O_bd_3.shape[0]):
    print("Local CC12_exp_bd_p3 issue")
  
  C_bd_6 = pd.read_csv(i+'/Local_CC12_exp_bd_p6.csv')
  N_bd_6 = pd.read_csv(i+'/Local_CN12_exp_bd_p6.csv')
  O_bd_6 = pd.read_csv(i+'/Local_CO12_exp_bd_p6.csv')
  
  C_bd_6 = C_bd_6.rename(columns={'Was_0': 'C6_W0', 'Was_1': 'C6_W1','Btl_0': 'C6_B0', 'Btl_1': 'C6_B1'})
  N_bd_6 = N_bd_6.rename(columns={'Was_0': 'N6_W0', 'Was_1': 'N6_W1','Btl_0': 'N6_B0', 'Btl_1': 'N6_B1'})
  O_bd_6 = O_bd_6.rename(columns={'Was_0': 'O6_W0', 'Was_1': 'O6_W1','Btl_0': 'O6_B0', 'Btl_1': 'O6_B1'})

  if(C_bd_6.shape[0]!=N_bd_6.shape[0] or C_bd_6.shape[0]!=O_bd_6.shape[0]):
    print("Local CC12_exp_bd_p6 issue")

  #print(C_e.shape[0],C_bd_6.shape[0],C_e.shape[0],C_bd_3.shape[0],C_e.shape[0],C_l.shape[0])
    
  
  temp = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(C_e,N_e,on=key),O_e,on=key),C_l,on=key),N_l,on=key),O_l,on=key),C_exp,on=key),N_exp,on=key),O_exp,on=key),C_bd_3,on=key),N_bd_3,on=key),O_bd_3,on=key),C_bd_6,on=key),N_bd_6,on=key),O_bd_6,on=key)
  temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
  #print(temp.head())
  #print(list(temp))
  #exit() 

  tmp_RF = pd.read_pickle('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+i[-4:]+'/Secondary_FRI_b')
  #------------------- For CA Only --------------------------------------
  tmp_RF = tmp_RF[tmp_RF.Atom_Type=='CA']
  #----------------------------------------------------------------------
  tmp_RF = tmp_RF.rename(columns={'Atom_Type': 'Type','B_Factor':'BF'})
  
  #print(tmp_RF.dtypes)
  #print(temp.dtypes)
  #if(tmp_RF.shape[0]!=temp.shape[0]):
  
  #print(i[-4:],tmp_RF.shape[0],temp.shape[0])
  #print(list(tmp_RF))
  #print(tmp_RF.BF.dtype)
  #print(list(temp))
  #print(temp.BF.dtype)
  #exit()
  
  tmp_RF.Index=tmp_RF.Index.astype('int64')
  
  temp = pd.merge(tmp_RF,temp,on=['X','Y','Z','BF','Type','Index'],how='left')
  #temp = pd.merge(tmp_RF,temp,on=['Index','X','Y','Z','BF','Atom'],how='left')
  
  #print(temp.shape)
  #print(list(temp))
  #exit() 
  
  #print(temp.head())
  #print(temp.tail())
  
  #if(tmp_RF.shape[0]!=temp.shape[0]):
  #print(i[-4:],tmp_RF.shape[0],temp.shape[0])
    
  temp = temp.fillna(0)
  temp['Protein'] = i[-4:] 
  counter=counter+temp.shape[0]
  rf_counter=rf_counter+tmp_RF.shape[0]
  
  #print(rf_counter,counter)
  #time.sleep(0.5)
  
  #print(count) 
  if(count>0):
    #print(PH.shape[1],temp.shape[1])
    #print(list(PH)) 
    #print(list(temp))
    PH = PH.append(temp)
    #print(i[-4:],PH.shape[0])
  else: 
    PH = temp
  #print(i[-4:],temp.shape,PH.shape) 


#for i in ['2i49','2q52','1aba']:
 # i='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+i
for i in names:
  add_protein(i,count)
  count = count+1
  if (os.path.isfile(i+'/Local_CC12_exp_bd.csv') and os.path.isfile(i+'/Local_CN12_exp_bd.csv') and os.path.isfile(i+'/Local_CO12_exp_bd.csv')):
    if(os.path.isfile(i+'/Local_CC12_exp_bd_p3.csv') and os.path.isfile(i+'/Local_CN12_exp_bd_p3.csv') and os.path.isfile(i+'/Local_CO12_exp_bd_p3.csv')):
      if(os.path.isfile(i+'/Local_CC12_exp_bd_p6.csv') and os.path.isfile(i+'/Local_CN12_exp_bd_p6.csv') and os.path.isfile(i+'/Local_CO12_exp_bd_p6.csv')):
        if(os.path.isfile('/mnt/home/bramerda/Documents/Secondary_Prediction/Results/'+i[-4:]+'/Secondary_FRI_b')):
          print(i[-4:])
          #add_protein(i,count)
          #count = count+1
          #print(i[-4:],PH.shape)
          a=2      
        else:
          print("{0} missing secondary FRI file".format(sys.argv[1]))
      else:
        print("{0} missing secondary Local_CX_exp_bd_p6 file".format(i[-4:]))
    else:
      print("{0} missing secondary Local_CX_exp_bd_p3 file".format(i[-4:]))
  else:
    print("{0} missing secondary Local_CX_exp_bd file".format(i[-4:]))     
          
print('done')
print(count,counter,rf_counter)       
#print(PH)
#print(PH.shape)

#print(PH.Protein.head())
#print(PH.Protein.tail())

#print(list(PH))
print("Dataset successfully created")

PH.to_pickle("Dataset_CA")
#print(PH.shape)
exit()
