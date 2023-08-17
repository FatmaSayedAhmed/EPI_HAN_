import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

name='HeLa-S3'

train_dir='./data/%s/train/'%name
test_dir='./data/%s/test/'%name

# oneHot Map:
#   A -> [1,0,0,0]
#   T -> [0,1,0,0]
#   C -> [0,0,1,0]
#   G -> [0,0,0,1]

def sentence2oneHot(str_set):
    oneHotMap = {'A': [1., 0., 0., 0.],
                 'T': [0., 1., 0., 0.],
                 'C': [0., 0., 1., 0.],
                 'G': [0., 0., 0., 1.],
                 'N': [0., 0., 0., 0.]
                 }
    oneHot=[]
    for sr in str_set:
        oneHot_sr = []
        for i in range(len(sr)):
            oneHot_sr.append(oneHotMap[sr[i]])
        oneHot.append(oneHot_sr)
    return np.asarray(oneHot)

print ('Loading seq data...')

enhancers_tra=open(train_dir+'%s_enhancer.fasta'%name,'r').read().splitlines()[1::2]
promoters_tra=open(train_dir+'%s_promoter.fasta'%name,'r').read().splitlines()[1::2]
y_tra=np.loadtxt(train_dir+'%s_label.txt'%name)

enhancers_tes=open(test_dir+'%s_enhancer_test.fasta'%name,'r').read().splitlines()[1::2]
promoters_tes=open(test_dir+'%s_promoter_test.fasta'%name,'r').read().splitlines()[1::2]
y_tes=np.loadtxt(test_dir+'%s_label_test.txt'%name)

print('size of train data')
print('pos_samples:'+str(int(sum(y_tra))))
print('neg_samples:'+str(len(y_tra)-int(sum(y_tra))))

print('size of test data')
print('pos_samples:'+str(int(sum(y_tes))))
print('neg_samples:'+str(len(y_tes)-int(sum(y_tes))))

X_en_tra = sentence2oneHot(enhancers_tra)
X_pr_tra = sentence2oneHot(promoters_tra)

X_en_tes = sentence2oneHot(enhancers_tes)
X_pr_tes = sentence2oneHot(promoters_tes)

separate_path = (train_dir + name + '_enhancers_train.npy')
np.save(separate_path, X_en_tra)
separate_path1 = (train_dir + name + '_promoters_train.npy')
np.save(separate_path1, X_pr_tra)
separate_path2 = (train_dir + name + '_labels_train.npy')
np.save(separate_path2, y_tra)

separate_path = (test_dir + name + '_enhancers_test.npy')
np.save(separate_path, X_en_tes)
separate_path1 = (test_dir + name + '_promoters_test.npy')
np.save(separate_path1, X_pr_tes)
separate_path2 = (test_dir + name + '_labels_test.npy')
np.save(separate_path2, y_tes)
