

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

type = "HAN_dna2vec_OneHot_SelfSoftAttention/GM12878"
from model import get_model
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


model_dir = "./model/specificModel/" + type

models=['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
m=models[0]
model=None
model=get_model()
model.load_weights(model_dir + "/GM12878Model14.h5")

cell_line = 'GM12878'

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
