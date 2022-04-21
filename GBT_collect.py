import numpy as np
import pandas as pd
import sys,os
import glob

#Takes CNN_CA_+"# of epochs"_"... file CC results and compiles the max CC for each of the n versions 

names = glob.glob('/mnt/home/bramerda/Documents/Persistent_Homology/Results/*')
names.sort()
fname='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'
name_list=['1ob4','1ob7','2olx','3md5','1nko','2oct','3fva']
for i in name_list:
  names.remove(fname+i)

count=0

#results = np.zeros(len(names), dtype={'names':('pdb', '50', '100','250','500','1000','max'),
 #                         'formats':('S5', 'f8', 'f8','f8','f8','f8','f8')})

results = np.zeros(len(names), dtype={'names':('pdb', '1', '2','3','4','5','6','7','8','9','10','max'),
                          'formats':('S5', 'f8', 'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8')})

print(results.shape)

def read_line(filen,num):
  tmp='/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+filen+'/[CA]No_XGB_ProToPro500_0.25_'+str(num)+'.txt'
  if(os.path.isfile(tmp)):
    fp = open(tmp)
    for i, line in enumerate(fp):
        if i == 5:
            temp=line.split()
        elif i > 5:
            break
    fp.close() 
    return temp[1]

for i in names:
  j=i[-4:]
  results[count]['pdb']=j

  #for k in (50,100,250,500,1000):
  for k in range(1,11):
    results[count][str(k)]=read_line(j,k)
  count=count+1  

#data=results[['50','100','250','500','1000']]
data=results[['1','2','3','4','5','6','7','8','9','10']]
data_array=data.view(np.float).reshape(data.shape + (-1,))

for i in range(0,results.shape[0]):
  results['max'][i]=np.nanmax(data_array[i,:])

#print(results['max'])
print(results)
print(np.nanmean(results['max']))

#print(" 50 CC: {0}\n100 CC: {1}\n250 CC: {2}\n500 CC: {3}\n1000 CC: {4}\nmax CC: {5}".format(np.nanmean(results['50']),np.nanmean(results['100']),np.nanmean(results['250']),np.nanmean(results['500']),np.nanmean(results['1000']),np.nanmean(results['max'])))

  