#Versions------------------------------|
#Keras 2.0.0 with tensorflow 1.3.0     |
#--------------------------------------|
import sys,os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, core
from keras.optimizers import SGD
from keras.models import Model
from keras import optimizers
#from keras.utils import plot_model
from keras.layers import Input
from keras.layers.merge import concatenate
#import pydot
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import time

t0 = time.time()
 
learn  = float(sys.argv[2]) #0.0005  #0.001 
ep     = int(sys.argv[3]) #40    #100
b_size = int(sys.argv[4]) # 500 #100

#b_size = 10000
#ep     = 10
fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/[CA]No_CNN_ProToPro.txt'
feats = ['Cexp_W0', 'Cexp_W1', 'Cexp_B0', 'Cexp_B1', 'Nexp_W0', 'Nexp_W1', 'Nexp_B0', 'Nexp_B1', 'Oexp_W0', 'Oexp_W1', 'Oexp_B0', 'Oexp_B1', 'CLor_W0', 'CLor_W1', 'CLor_B0', 'CLor_B1', 'NLor_W0', 'NLor_W1', 'NLor_B0', 'NLor_B1', 'OLor_W0', 'OLor_W1', 'OLor_B0', 'OLor_B1', 'Ce_W0', 'Ce_W1', 'Ce_B0', 'Ce_B1', 'Ne_W0', 'Ne_W1', 'Ne_B0', 'Ne_B1', 'Oe_W0', 'Oe_W1', 'Oe_B0', 'Oe_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1', 'Occupancy', 'Phi', 'Psi', 'Area', 'H', 'G', 'I', 'E', 'B', 'b', 'T',  'N', 'O', 'S', 'Res', '500', '750', '1000', '1500', '2000', '2500', '3000', '4000', '5000', '15000', 'ASN', 'ILE', 'GLN', 'ALA', 'ARG', 'GLY', 'MET', 'ASP', 'TYR', 'LEU', 'PRO', 'GLU', 'THR', 'TRP', 'LYS', 'VAL', 'SER', 'PHE', 'CYS', 'HIS', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value']

 
Image = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Image_Data.npy')
Full_Data = pd.read_csv('/mnt/home/bramerda/Documents/Persistent_Homology/Dataset_CA_p3.csv')
CA_Data = Full_Data[Full_Data.Type=='CA'] 

# - Image Train and Test Set
Image_Key = pd.read_pickle('/mnt/home/bramerda/Documents/Persistent_Homology/Image_Key_p3')
#Image_Key = pd.DataFrame(Image_Key,columns=['pdb','CA','BF'])
Image_Key.rename(columns= {0:'pdb',1:'CA',2:'BF'},inplace=True)

#Image_Key['CA']=Image_Key['CA'].astype(str).astype(int)

Image_Key_ind = Image_Key[Image_Key['pdb']==(sys.argv[1])].index
X_image_train = Image[Image_Key[Image_Key.pdb!=sys.argv[1]].index,:,:,:]
X_image_test = Image[Image_Key[Image_Key.pdb==sys.argv[1]].index,:,:,:]

# - Data Train and Test Set
test_set = CA_Data[Full_Data.Protein==sys.argv[1]]
X_test = test_set[feats] #.astype('float')
y_test = test_set[['BF']]

train_set = CA_Data[Full_Data.Protein!=sys.argv[1]]
X_train = train_set[feats] #.to_numeric()  #X train
y_train = train_set[['BF']] #.to_numeric() #y train


# ------------------------------------------------------
if(X_image_train.shape[0]!=X_train.shape[0]):
  print("Data dimension mismatch")
  print(X_image_train.shape, X_train.shape, y_train.shape)
  exit()
  
if(X_image_test.shape[0]!=X_test.shape[0]):
  print("Data dimension mismatch")
  print(X_image_test.shape, X_test.shape, y_test.shape)
  exit()  
# ------------------------------------------------------
print('Input train/test dimensions okay...')

X_image_train, X_train, y_train = shuffle(X_image_train, X_train, y_train, random_state=0)


print("Batch Size:{0}\nEpochs {1}\nLearning Rate: {2}\n".format(b_size,ep,learn))
print("training size {0}".format(X_train.shape[0]))

visible = Input(shape=(8,10,3))
conv1 = Conv2D(14, kernel_size=2, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=2, activation='relu')(conv1)
drop1 = Dropout(0.5)(conv2)
#pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#hidden1 = Dense(10, activation='relu')(drop1)
hiddenF = Dense(1, activation='relu')(drop1)
flat1 = Flatten()(hiddenF)

global_1 = Input(shape=(X_train.shape[1],))

merge = concatenate([flat1, global_1])
hidden2  = Dense(2*X_test.shape[1], activation = 'relu')(merge)
drop2 = Dropout(0.5)(hidden2)
#drop2 = Dropout(0.25)(hidden2)
hidden3  = Dense(100, activation = 'relu')(drop2)
drop3 = Dropout(0.25)(hidden3)
hidden4  = Dense(10, activation = 'relu')(drop3)
output  = Dense(1, activation = 'relu')(hidden4) 
 
model = Model(inputs=[visible,global_1], outputs=output) 
#model = Model(inputs=global_1, outputs=output)
# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file=fname+'convolutional_neural_network.png')
#exit()



errors = ['mean_absolute_error','mean_squared_error',]
opt = optimizers.Adam(lr=learn)
#opt = SGD(lr=learn)
ls = errors[1]

print(2*X_test.shape[1])
print(ls)
#exit()

model.compile(loss=ls, optimizer=opt,
              metrics=['mse','mae','cosine'])

print('okay till now')

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_train = X_train.values
y_train = y_train.values
print(X_train.shape,y_train.shape)
#print(y_train.head())
#exit()

model.fit([X_image_train,X_train], y_train, batch_size=b_size, epochs=ep,verbose=0) #verbose=1 for display bar, 0 for nothing

print('bingo')
#filename = '/mnt/home/bramerda/Documents/Secondary_Prediction/ProToPro/DNN/Results/'+sys.argv[1]+'/Final_CNN_Model.sav'
#pickle.dump(model, open(filename, 'wb'))

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_test = X_test.values
y_test = y_test.values


score = model.evaluate([X_image_test,X_test], y_test, batch_size=b_size)

#model.fit(X_train, y_train, batch_size=b_size, epochs=ep,validation_data=(X_test,y_test))
#score = model.evaluate(X_test, y_test, batch_size=b_size)

print("\nTest Loss: {0}\n".format(score))

pred = np.float64(model.predict([X_image_test,X_test]).ravel())
#pred = np.float64(model.predict(X_test).ravel())
y_test = np.float64(y_test)
y_test=y_test.reshape((y_test.shape[0],))
CC = np.corrcoef(pred,y_test)




finalCC=np.hstack((y_test.reshape((y_test.shape[0],1)),pred.reshape((y_test.shape[0],1))))

#pred_CA = np.float64(model.predict([X_image_test_CA,X_test_CA]).ravel())
#y_test_CA = np.float64(y_test_CA)
#CC_CA = np.corrcoef(pred_CA,y_test_CA)
print(finalCC)
print("data not saved...")
exit()

np.savetxt('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/CNN_CA_pred_'+sys.argv[3]+'_'+sys.argv[5]+'.txt',finalCC)

print("\nCC: {0}\n".format(CC[0,1]))
#print("\nCA Only CC: {0}\n".format(CC_CA[0,1]))

print("Batch Size:{0}\nEpochs {1}\nLearning Rate: {2}\nTraining Size: {3}".format(b_size,ep,learn,X_train.shape[0]))

#np.save('Actul.npy',y_test)
#np.save('Pred.npy',pred)
t1 = time.time()
total_time = t1-t0

f = open('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/CNN_CA_'+sys.argv[3]+'_'+sys.argv[5]+'.txt', 'w+')
f.write(("Protein: {0}\n").format(sys.argv[1]))
f.write(("Residues: {0}\n").format(X_test.shape[0]))
f.write(("Training_Set_Size: {0}\n").format(X_train.shape[0]))
#f.write(("Estimators: {0}\n").format(nest))
f.write(("CNN_Result: \n"))
#f.write(("MWCG_+_Secondary_Featrues + AA + Type(CNOS) + Resolution + Protein Size + Local Density + R Value \n"))
f.write("CA_Corr_Coef: {0}\n".format(CC[0,1]))
#f.write("CA_Corr_Coef: {0}\n".format(CC_CA[0,1]))
#f.write(("{0} \n").format(key[12:]))
#f.write(("Result using only 10,000 random samples w random.seed(3) \n"))
f.write(("Learning_Rate: {0} \n").format(learn))
f.write(("Epoch: {0} \n").format(ep))
f.write(("Batch_Size: {0} \n").format(b_size))
f.write(("Train_Size: {0} \n").format(y_train.shape[0]))
f.write(("Test_Size: {0} \n").format(y_test.shape[0]))
f.write(("Test_Loss: {0} \n").format(score[0]))
f.write(("Total_Time: {0} \n").format(total_time))
f.close

print("Sucessfully finished")
exit()

# https://stackoverflow.com/questions/45528285/cnn-image-recognition-with-regression-output-on-tensorflow



