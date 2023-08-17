# In[ ]:
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="1"
type = "HAN_OneHot_SelfSoftAttention/HUVEC"

from model import get_model
import numpy as np
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

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
        y_pred = self.model.predict([self.en,self.pr], batch_size=64, verbose=1)
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        # f1_val = f1_score(self.y, np.round(y_pred(-1)))

        mkdir(model_dir)

        self.model.save_weights(model_dir + "/HUVEC_Model" + str(epoch) + ".h5")

        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        # print('\r f1_score_val: %s ' % str(round(f1_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
cell_line=names[1]

data_path_train='./data/%s/train/'%cell_line
# data_path_imbltrain='./data/%s/imbltrain/'%cell_line

model=None
model=get_model()
model.summary()

print('Loading balanced train ' + cell_line + ' data from ' + data_path_train)
X_enhancers_train = np.load(data_path_train + cell_line + '_enhancers_train.npy')
X_promoters_train = np.load(data_path_train + cell_line + '_promoters_train.npy')
labels_train = np.load(data_path_train + cell_line + '_labels_train.npy')

X_en_tra, X_en_val,X_pr_tra,X_pr_val, y_tra, y_val=train_test_split(
    X_enhancers_train,X_promoters_train,labels_train,test_size=0.05,stratify=labels_train,random_state=250)

del X_enhancers_train
del X_promoters_train
del labels_train
gc.collect()

print ('Traing %s cell line specific model ...'%cell_line)


back = roc_callback(val_data=[X_en_val, X_pr_val, y_val], name=cell_line)
history=model.fit([X_en_tra, X_pr_tra], y_tra, validation_data=([X_en_val, X_pr_val], y_val), epochs=15, batch_size=64,
                  callbacks=[back])

t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("start time:"+t1+"end time："+t2)

del X_en_tra
del X_en_val
del X_pr_tra
del X_pr_val
del y_tra
del y_val
gc.collect()

print("\nrun test\n")
data_path_test='./data/%s/test/'%cell_line

print('Loading test ' + cell_line + ' data from ' + data_path_test)
X_enhancers_test = np.load(data_path_test + cell_line + '_enhancers_test.npy')
X_promoters_test = np.load(data_path_test + cell_line + '_promoters_test.npy')
labels_test = np.load(data_path_test + cell_line + '_labels_test.npy')

print("****************Testing %s cell line specific model on %s cell line****************" % (cell_line, cell_line))
y_pred = model.predict([X_enhancers_test, X_promoters_test], batch_size=50, verbose=1)
auc = roc_auc_score(labels_test, y_pred)
print("AUC : ", auc)
aupr = average_precision_score(labels_test, y_pred)
print("AUPR : ", aupr)
