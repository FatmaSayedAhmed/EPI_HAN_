

# In[ ]:
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="0"
type = "HAN_dna2vec_OneHot_SelfSoftAttention/GM12878"

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
        self.en_dna2vec = val_data[0]
        self.pr_dna2vec = val_data[1]
        self.en_oneHot = val_data[2]
        self.pr_oneHot = val_data[3]
        self.y = val_data[4]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en_dna2vec,self.pr_dna2vec, self.en_oneHot,self.pr_oneHot])
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

Data_dir_dna2vec='./data/%s/'%name

print('Loading balanced train ' + name + ' data from ' + Data_dir_dna2vec)
train_dna2vec=np.load(Data_dir_dna2vec+'%s_train.npz'%name)
X_enhancers_train_dna2vec,X_promoters_train_dna2vec,labels_train_dna2vec=train_dna2vec['X_en_tra'],train_dna2vec['X_pr_tra'],train_dna2vec['y_tra']

X_en_tra_dna2vec, X_en_val_dna2vec,X_pr_tra_dna2vec,X_pr_val_dna2vec, y_tra_dna2vec, y_val_dna2vec=train_test_split(
    X_enhancers_train_dna2vec,X_promoters_train_dna2vec,labels_train_dna2vec,test_size=0.05,stratify=labels_train_dna2vec,random_state=250)

del X_enhancers_train_dna2vec
del X_promoters_train_dna2vec
del labels_train_dna2vec
gc.collect()

data_path_train_oneHot='./data/%s/train/'%name

print('Loading balanced train ' + name + ' data from ' + data_path_train_oneHot)
X_enhancers_train_oneHot = np.load(data_path_train_oneHot + name + '_enhancers_train.npy')
X_promoters_train_oneHot = np.load(data_path_train_oneHot + name + '_promoters_train.npy')
labels_train_oneHot = np.load(data_path_train_oneHot + name + '_labels_train.npy')

X_en_tra_oneHot, X_en_val_oneHot,X_pr_tra_oneHot,X_pr_val_oneHot, y_tra_oneHot, y_val_oneHot=train_test_split(
    X_enhancers_train_oneHot,X_promoters_train_oneHot,labels_train_oneHot,test_size=0.05,stratify=labels_train_oneHot,random_state=250)

del X_enhancers_train_oneHot
del X_promoters_train_oneHot
del labels_train_oneHot
gc.collect()


print ('Training %s cell line specific model ...'%name)

back = roc_callback(val_data=[X_en_val_dna2vec, X_pr_val_dna2vec, X_en_val_oneHot, X_pr_val_oneHot, y_val_dna2vec], name=name)
history=model.fit([X_en_tra_dna2vec, X_pr_tra_dna2vec, X_en_tra_oneHot, X_pr_tra_oneHot], y_tra_dna2vec, validation_data=([X_en_val_dna2vec, X_pr_val_dna2vec, X_en_val_oneHot, X_pr_val_oneHot], y_val_dna2vec), epochs=15, batch_size=64,
                  callbacks=[back])

t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("start Time:"+t1+"end time："+t2)

del X_en_tra_dna2vec
del X_en_val_dna2vec
del X_pr_tra_dna2vec
del X_pr_val_dna2vec
del y_tra_dna2vec
del y_val_dna2vec

del X_en_tra_oneHot
del X_en_val_oneHot
del X_pr_tra_oneHot
del X_pr_val_oneHot
del y_tra_oneHot
del y_val_oneHot

gc.collect()

print("\nrun test\n")
data_path_test_dna2vec='./data/%s/'%name

print('Loading test ' + name + ' data from ' + data_path_test_dna2vec)
test_dna2vec=np.load(Data_dir_dna2vec+'%s_test.npz'%name)
X_enhancers_test_dna2vec,X_promoters_test_dna2vec,labels_test_dna2vec=test_dna2vec['X_en_tes'],test_dna2vec['X_pr_tes'],test_dna2vec['y_tes']

data_path_test_oneHot='./data/%s/test/'%name

print('Loading test ' + name + ' data from ' + data_path_test_oneHot)
X_enhancers_test_oneHot = np.load(data_path_test_oneHot + name + '_enhancers_test.npy')
X_promoters_test_oneHot = np.load(data_path_test_oneHot + name + '_promoters_test.npy')
labels_test_oneHot = np.load(data_path_test_oneHot + name + '_labels_test.npy')

print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
y_pred = model.predict([X_enhancers_test_dna2vec, X_promoters_test_dna2vec, X_enhancers_test_oneHot, X_promoters_test_oneHot], batch_size=50, verbose=1)
auc = roc_auc_score(labels_test_dna2vec, y_pred)
print("AUC : ", auc)
aupr = average_precision_score(labels_test_dna2vec, y_pred)
print("AUPR : ", aupr)

