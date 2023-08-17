# EPI-HAN ( Identification of Enhancer Promoter Interaction using Hierarchical Attention Network)
Enhancer-Promoter Interaction (EPI) recognition is crucial for understanding human development. In transcriptional regulation, EPI in the genome plays a significant role. In genome-wide association studies (GWAS), EPIs can help enhance statistical strength and improve the mechanistic understanding of disease- or trait-associated genetic variants. However, in terms of time, effort, and resources, experimental methods to classify EPIs cost too much. Therefore, increasing research efforts are focusing on the development of computational methods that use deep learning and other machine learning methods to solve these problems. 

Unfortunately, one of the main challenges when dealing with EPIs prediction is the long sequences of the enhancers and promoters, and most existing computational approaches suffer from this problem. Although there have been significant advances in deep learning models for predicting EPIs, they currently overlook the hierarchical organization of lengthy DNA sequences.  To address the above challenge, in this paper, we propose that incorporating the structure of the sequence into the model design will result in more accurate EPIs predictions, so our contributions are the development of a new deep neural network that includes a Hierarchical Attention Network (HAN) and proposing a hybrid embedding strategy. The proposed EPI-HAN model has two distinct features: (i) a hybrid embedding strategy is utilized to enrich the input features of the hierarchical attention network, (ii) the hierarchical structure of the HAN comprises two attention layers that operate at the level of individual tokens and smaller sequences, allowing it to selectively focus on important features when creating the overall representation of the sequence. In some cell lines, benchmark comparisons reveal that EPI-HAN outperforms state-of-the-art methods in terms of AUROC and AUPR performance metrics.

# File Description 
- Data_Augmentation.R

  A tool for data augmentation was provided by Mao et al. (2017). The details of the tool can be seen at https://github.com/wgmao/EPIANN.

  To amplify the positive samples in the training set to 20 times to achieve class balance, we used this tool.
  
- fasta2oneHot.py

  Perform pre-processing of DNA sequences:
  1. convert the enhancer and promoter sequences into one-hot vectors by using this oneHot Map:
     A -> [1,0,0,0]
     T -> [0,1,0,0]
     C -> [0,0,1,0]
     G -> [0,0,0,1]
  2. save the one-hot embeddings into .npy files

- dna2vec_sequence_processing.py

  Perform pre-processing of DNA sequences:

  1. Transform the enhancer and promoter gene sequences into word sequences (6-mers), marking a word as "NULL" if it contains "N".
  2. Create a dictionary with 4^6+1 words.
  3. Transform each gene sequence into a list of dictionary-compliant word indexes (each word has its own unique index).
  4. save the dna2vec embeddings into .npz files
  
- embedding_matrix.npy

    The weight of the embedding layer was calculated using Ng's pre-trained DNA vector (2017).
    
- model.py

    It contains the implementation of our proposed EPI-HAN model

- layers.py

    It contains custom layer implementation for self-attention mechanisms in TensorFlow

- train_dna2vec.py

  Perform model training using the dna2vec embedding

- train_OneHot.py

    Perform model training using the one-hot embedding

- train_dna2vec_oneHot.py

  Perform model training using the combination of dna2vec and one-hot embeddings.

- test.py

  Evaluate the performance of the model.
  
You can find the weight of the model mentioned in our paper under the directory ./model

Directory|Content 
  ---|---
  model/OneHot/SelfAttention/| the weight of EPI-HAN using the one-hot embedding and self-attention mechanism on each cell line.
  model/OneHot/SoftAttention/| the weight of EPI-HAN using the one-hot embedding and soft attention mechanism on each cell line.
  model/OneHot/SelfSoftAttention/| the weight of EPI-HAN using the one-hot embedding and self + soft attention mechanisms on each cell line.
  model/Dna2vec/SelfAttention/| the weight of EPI-HAN using the dna2vec embedding and self-attention mechanism on each cell line.
  model/Dna2vec/SoftAttention/| the weight of EPI-HAN using the dna2vec embedding and soft attention mechanism on each cell line.
  model/Dna2vec/SelfSoftAttention/| the weight of EPI-HAN using dna2vec embedding and self + soft attention mechanisms on each cell line.
  model/Dna2vec_OneHot/SelfAttention/| the weight of EPI-HAN using the dna2vec + one-hot embeddings and self-attention mechanism on each cell line.
  model/Dna2vec_OneHot/SoftAttention/| the weight of EPI-HAN using the dna2vec + one-hot embeddings and soft attention mechanism on each cell line.
  model/Dna2vec_OneHot/SelfSoftAttention/| the weight of EPI-HAN using dna2vec + one-hot embeddings and self + soft attention mechanisms on each cell line.

References:

  Mao, W. et al. (2017) Modeling Enhancer-Promoter Interactions with Attention-Based Neural Networks. bioRxiv, 219667.

  Ng, P. (2017) dna2vec: Consistent vector representations of variable-length k-mers. arXiv:1701.06279.
