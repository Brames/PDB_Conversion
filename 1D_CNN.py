#Versions------------------------------|
#Keras 2.0.0 with tensorflow 1.3.0     |
#--------------------------------------|
import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, core,MaxPooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras import optimizers
from keras.layers import Input
from keras.layers.merge import concatenate
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import time
from keras import backend as K


def CC_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

t0 = time.time()
 
learn  = 0.001
ep     = int(sys.argv[2])
b_size = 32

#b_size = 10000
#ep     = 10
fname = '/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/[CA]No_CNN_ProToPro.txt'
feats = ['Cexp_W0', 'Cexp_W1', 'Cexp_B0', 'Cexp_B1', 'Nexp_W0', 'Nexp_W1', 'Nexp_B0', 'Nexp_B1', 'Oexp_W0', 'Oexp_W1', 'Oexp_B0', 'Oexp_B1', 'CLor_W0', 'CLor_W1', 'CLor_B0', 'CLor_B1', 'NLor_W0', 'NLor_W1', 'NLor_B0', 'NLor_B1', 'OLor_W0', 'OLor_W1', 'OLor_B0', 'OLor_B1', 'Ce_W0', 'Ce_W1', 'Ce_B0', 'Ce_B1', 'Ne_W0', 'Ne_W1', 'Ne_B0', 'Ne_B1', 'Oe_W0', 'Oe_W1', 'Oe_B0', 'Oe_B1', 'C3_W0', 'C3_W1', 'C3_B0', 'C3_B1', 'N3_W0', 'N3_W1', 'N3_B0', 'N3_B1', 'O3_W0', 'O3_W1', 'O3_B0', 'O3_B1', 'C6_W0', 'C6_W1', 'C6_B0', 'C6_B1', 'N6_W0', 'N6_W1', 'N6_B0', 'N6_B1', 'O6_W0', 'O6_W1', 'O6_B0', 'O6_B1', 'Occupancy', 'Phi', 'Psi', 'Area', 'H', 'G', 'I', 'E', 'B', 'b', 'T',  'N', 'O', 'S', 'Res', '500', '750', '1000', '1500', '2000', '2500', '3000', '4000', '5000', '15000', 'ASN', 'ILE', 'GLN', 'ALA', 'ARG', 'GLY', 'MET', 'ASP', 'TYR', 'LEU', 'PRO', 'GLU', 'THR', 'TRP', 'LYS', 'VAL', 'SER', 'PHE', 'CYS', 'HIS', 'Short_Density', 'Med_Density', 'Long_Density', 'R_Value']

 
Image = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Image_Data.npy')
Image_1 = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Revised_GloIm_Data.npy')
Image_2 = np.load('/mnt/home/bramerda/Documents/Persistent_Homology/Revised_DifIm_Data.npy')
Full_Data = pd.read_csv('/mnt/home/bramerda/Documents/Persistent_Homology/Dataset_CA_p3.csv')
CA_Data = Full_Data[Full_Data.Type=='CA'] 

# - Image Train and Test Set
Image_Key = pd.read_pickle('/mnt/home/bramerda/Documents/Persistent_Homology/Image_Key_p3')
Image_Key.rename(columns= {0:'pdb',1:'CA',2:'BF'},inplace=True)

Image_Key_ind = Image_Key[Image_Key['pdb']==(sys.argv[1])].index
X_image_train = Image[Image_Key[Image_Key.pdb!=sys.argv[1]].index,:,:,:]
X_image_test = Image[Image_Key[Image_Key.pdb==sys.argv[1]].index,:,:,:]

X_image_train_1 = Image_1[Image_Key[Image_Key.pdb!=sys.argv[1]].index,:,:,:]
X_image_test_1 = Image_1[Image_Key[Image_Key.pdb==sys.argv[1]].index,:,:,:]

X_image_train_2 = Image_2[Image_Key[Image_Key.pdb!=sys.argv[1]].index,:,:,:]
X_image_test_2 = Image_2[Image_Key[Image_Key.pdb==sys.argv[1]].index,:,:,:]

# - Data Train and Test Set
test_set = CA_Data[Full_Data.Protein==sys.argv[1]]
X_test = test_set[feats] #.astype('float')
y_test = test_set[['BF']]

train_set = CA_Data[Full_Data.Protein!=sys.argv[1]]
X_train = train_set[feats] #.to_numeric()  #X train
y_train = train_set[['BF']] #.to_numeric() #y train

#Shuffle Data
X_train,X_image_train,X_image_train_1,X_image_train_2,y_train = shuffle(X_train,X_image_train,X_image_train_1,X_image_train_2, y_train, random_state=0)

drp=0.25
# CNN 1
input1 = keras.layers.Input(shape=(8,10,3)) 
x1 = Conv2D(16, kernel_size=(3,3), activation='relu')(input1)
#x1 = MaxPooling2D(pool_size=(3,3))(x1)
x1 = Dropout(drp)(x1)
x1 = Conv2D(8, kernel_size=(2,2), activation='relu')(x1)
x1 = Dropout(drp)(x1)
x1 = Flatten()(x1)
#x1 = Dense(1, activation='relu')(x1)

# CNN 2
input2 = keras.layers.Input(shape=(3,48,3))
x2 = Conv2D(8,kernel_size=(3,3), activation='relu')(input2)
x2 = Dropout(drp)(x2)
#x2 = MaxPooling1D(pool_size=1)(x2)
x2 = Conv2D(4, kernel_size=(1,3), activation='relu')(x2) 
x2 = Dropout(drp)(x2)
x2 = Flatten()(x2)
#x2 = Dense(1, activation = 'relu')(x2) 

# CNN 3
input3 = keras.layers.Input(shape=(3,48,3))
x3 = Conv2D(8, kernel_size=(3,3), activation='relu')(input3)
x3 = Dropout(drp)(x3)
#x3 = MaxPooling1D(pool_size=1)(x3)
x3 = Conv2D(16, kernel_size=(1,3), activation='relu')(x3)
x3 = Dropout(drp)(x3)
x3 = Flatten()(x3)
#x3 = Dense(1, activation='relu')(x3) 

#xv=[x1,x2,x3]

#concat = keras.layers.Concatenate(axis=-1)(xv)
#out = keras.layers.Dense(1)(concat)

# DNN 
input4 = keras.layers.Input(shape=(X_train.shape[1],))
xv = [x1,x2,x3,input4]
x4 = keras.layers.Concatenate(axis=-1)(xv)
#x4 = concatenate([out,input4])
#x4 = Dropout(drp)(x4)
#x4  = Dense(100, activation='relu')(x4)
#x4 = Dropout(drp)(x4)
x4  = Dense(50, activation='relu')(x4)
x4 = Dropout(drp)(x4)
x4  = Dense(10, activation='relu')(x4)
x4 = Dropout(drp)(x4)
x4  = Dense(1, activation='relu')(x4)


Input=[input1,input2,input3,input4]
model = keras.models.Model(inputs=Input, outputs=x4)
#model = keras.models.Model(inputs=[input1,input2,input3,input4], outputs=x4)
 
# summarize layers
print(model.summary())
#print("\nBatch Size:{0}\nEpochs {1}\nLearning Rate: {2}\n".format(b_size,ep,learn))
#print("training size {0}\n".format(X_image_train.shape[0]))
# plot graph
#plot_model(model, to_file=fname+'convolutional_neural_network.png')
#exit()

errors = ['mean_absolute_error','mean_squared_error']
opt=keras.optimizers.Adam(lr=learn, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
ls = errors[1]

model.compile(loss=ls, optimizer=opt,metrics=['mse'])
#model.compile(loss=CC_loss, optimizer=opt,metrics=['mae','mse'])

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_train = X_train.values
y_train = y_train.values

train=[X_image_train,X_image_train_1,X_image_train_2,X_train]
#train=[X_train]

model.fit(train, y_train, batch_size=b_size, epochs=ep,verbose=1) #verbose=1 for display bar, 0 for nothing

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_test = X_test.values
y_test = y_test.values

test=[X_image_test,X_image_test_1,X_image_test_2,X_test]
#test=[X_test]

score = model.evaluate(test,y_test,batch_size=b_size)

print("\nTest Loss: {0}\n".format(score))

pred = np.float64(model.predict(test).ravel())

y_test = np.float64(y_test)
y_test=y_test.reshape((y_test.shape[0],))
CC = np.corrcoef(pred,y_test)

finalCC=np.hstack((y_test.reshape((y_test.shape[0],1)),pred.reshape((y_test.shape[0],1))))

print("Batch Size:{0}\nEpochs {1}\nLearning Rate: {2}\nCC: {3}\n".format(b_size,ep,learn,CC[0,1]) ) 
t1 = time.time()
total_time = t1-t0
print("Total time: {0}".format(total_time))

print("data not saved...\n")
exit()

np.savetxt('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/CNN_CA_pred_'+sys.argv[3]+'_'+sys.argv[5]+'.txt',finalCC)

print("\nCC: {0}\n".format(CC[0,1]))
#print("\nCA Only CC: {0}\n".format(CC_CA[0,1]))


#np.save('Actul.npy',y_test)
#np.save('Pred.npy',pred)
t1 = time.time()
total_time = t1-t0

f = open('/mnt/home/bramerda/Documents/Persistent_Homology/Results/'+sys.argv[1]+'/CNN_CA_'+sys.argv[3]+'_'+sys.argv[5]+'.txt', 'w+')
f.write(("Protein: {0}\n").format(sys.argv[1]))

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



