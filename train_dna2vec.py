
# In[ ]:
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="5"
type = "HAN_dna2vec_SelfSoftAttention/GM12878"

from model import get_model
import numpy as np
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
model_dir = "./model/specificModel/" + type

def mkdir(path):
    """
    make dictionary
    :param path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en,self.pr])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)

        mkdir(model_dir)

        self.model.save_weights(model_dir + "/GM12878_Model" + str(epoch) + ".h5")


        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
name=names[0]
#The data used here is the sequence processed by data_processing.py.

model=None
model=get_model()
model.summary()

Data_dir='./data/%s/'%name

print('Loading balanced train ' + name + ' data from ' + Data_dir)
train=np.load(Data_dir+'%s_train.npz'%name)
X_enhancers_train,X_promoters_train,labels_train=train['X_en_tra'],train['X_pr_tra'],train['y_tra']

X_en_tra, X_en_val,X_pr_tra,X_pr_val, y_tra, y_val=train_test_split(
    X_enhancers_train,X_promoters_train,labels_train,test_size=0.05,stratify=labels_train,random_state=250)

del X_enhancers_train
del X_promoters_train
del labels_train
gc.collect()

print ('Training %s cell line specific model ...'%name)

back = roc_callback(val_data=[X_en_val, X_pr_val, y_val], name=name)
history=model.fit([X_en_tra, X_pr_tra], y_tra, validation_data=([X_en_val, X_pr_val], y_val), epochs=15, batch_size=64,
                  callbacks=[back])

t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("start Time:"+t1+"end time："+t2)

del X_en_tra
del X_en_val
del X_pr_tra
del X_pr_val
del y_tra
del y_val
gc.collect()

print("\nrun test\n")
data_path_test='./data/%s/'%name

print('Loading test ' + name + ' data from ' + data_path_test)
test=np.load(Data_dir+'%s_test.npz'%name)
X_enhancers_test,X_promoters_test,labels_test=test['X_en_tes'],test['X_pr_tes'],test['y_tes']

print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
y_pred = model.predict([X_enhancers_test, X_promoters_test], batch_size=50, verbose=1)
auc = roc_auc_score(labels_test, y_pred)
print("AUC : ", auc)
aupr = average_precision_score(labels_test, y_pred)
print("AUPR : ", aupr)