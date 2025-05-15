#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# Import libraries
import seaborn as sn
import pandas as pd
import json, os
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from collections import defaultdict
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from transformers import set_seed
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForSequenceClassification #, BertModel, BertTokenizer

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

import logging


# In[2]:


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# Define logger
logger = logging.getLogger(__name__)


# In[3]:


# Read dataset
root_path = os.getcwd()

project = "bigvul" # bigvul  devign  reveal

if project == "bigvul":
    test_data = pd.read_csv('bigvul_updated_with_cpp_target.csv')
    checkpoint_dir = './checkpoints'
    seed_index = 9


# Specify a constant seeder for processes
seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]
seed = seeders[seed_index]
logger.info(f"SEED: {seed}")
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
set_seed(seed)

#test_data = test_data.iloc[0:200,:]


# In[4]:


test_data.head


# In[5]:


# Model checkpoint and fine-tuning logic
FINE_TUNE = False  # Set this to False if you don't want to fine-tune the model and load from checkpoint

save_path = os.path.join(checkpoint_dir, 'best_weights.pt')


# In[6]:


# Pre-trained tokenizer
model_variation = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer
#bert-base-uncased #bert-base # roberta-base # distilbert-base-uncased #distilbert-base # microsoft/codebert-base-mlm
# 'albert-base-v2'

# tokenizer = RobertaTokenizer(vocab_file="../../tokenizer_training/cpp_tokenizer/cpp_tokenizer-vocab.json",
#                              merges_file="../../tokenizer_training/cpp_tokenizer/cpp_tokenizer-merges.txt")


# In[7]:


test_data = test_data[["processed_func", "target", "target_cppcheck", "index"]]


# In[8]:


word_counts = test_data["processed_func"].apply(lambda x: len(x.split()))
max_length = word_counts.max()
print("Maximum number of words:", max_length)


# In[9]:


vc = test_data["target"].value_counts()

print(vc)

print("Percentage: ", (vc[1] / vc[0])*100, '%')

n_categories = len(vc)
print(n_categories)


# In[10]:


vc2 = test_data["target_cppcheck"].value_counts()

print(vc2)

print("Percentage: ", (vc2[1] / vc2[0])*100, '%')


# In[11]:


test_data = pd.DataFrame(({'Text': test_data['processed_func'], 'Labels': test_data['target'], 'ASA': test_data['target_cppcheck'], 'index':test_data['index']}))
test_data


# In[12]:


model = AutoModelForSequenceClassification.from_pretrained(model_variation, num_labels=n_categories)


# In[13]:


model.resize_token_embeddings(len(tokenizer))


# In[14]:


X_test = tokenizer(
    text=test_data['Text'].tolist(),
    add_special_tokens=True,
    max_length=512,
    truncation=True,
    padding=True,
    return_tensors='pt',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)


# In[15]:


Y_test = torch.LongTensor(test_data["Labels"].tolist())
Y_test.size()


# In[16]:


batch_size = 8

test_dataset = TensorDataset(X_test["input_ids"], X_test["attention_mask"], Y_test)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)


# In[17]:


loss_fun = nn.CrossEntropyLoss()


# In[18]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[19]:


checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)


# In[ ]:


model.eval()
test_pred = []
actual_labels = []
test_loss = 0
all_scores = []
with torch.no_grad():
    for step_num, batch_data in enumerate(tqdm(test_dataloader, desc='Testing')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        
        output = model(input_ids = input_ids, attention_mask=att_mask) #, labels= labels

        loss = loss_fun(output.logits, labels) #loss = output.loss #output[0]
        test_loss += loss.item()
   
        preds = np.argmax(output.logits.cpu().detach().numpy(), axis=-1)
        all_scores.append(output.logits.cpu().detach().numpy())
        test_pred+=list(preds)
        actual_labels+=labels.cpu().numpy().tolist()
        


# In[68]:


total_test_loss = test_loss/len(test_dataloader) 

conf_matrix = confusion_matrix(actual_labels, test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
#acc = ((tp+tn)/(tp+tn+fp+fn))
#print(conf_matrix)
sn.heatmap(conf_matrix, annot=True)


# In[69]:


test_data["VP"] = test_pred
test_data


# Ground Truths

# In[70]:


print(len(test_data[test_data["Labels"] == 1]))


# Only Static Analysis

# In[71]:


print(len(test_data[test_data["ASA"] == 1]))


# In[72]:


asa_tp = len(test_data[(test_data["ASA"] == 1) & (test_data["Labels"] == 1)])
asa_tn = len(test_data[(test_data["ASA"] == 0) & (test_data["Labels"] == 0)])
asa_fp = len(test_data[(test_data["ASA"] == 1) & (test_data["Labels"] == 0)])
asa_fn = len(test_data[(test_data["ASA"] == 0) & (test_data["Labels"] == 1)])


# In[73]:


print("TP=",asa_tp)
print("TN=",asa_tn)
print("FP=",asa_fp)
print("FN=",asa_fn)

precision = (asa_tp) / (asa_tp + asa_fp)
recall = (asa_tp) / (asa_tp + asa_fn)
f1_asa = (2*precision*recall) / (precision+recall)
print("F1-score=",f1_asa * 100)


# Static Analysis and VP

# In[75]:


test_data["SAST"] = ((test_data["ASA"] == 1) & (test_data["VP"] == 1)).astype(int)

sast_tp = len(test_data[(test_data["SAST"] == 1) & (test_data["Labels"] == 1)])
sast_tn = len(test_data[(test_data["SAST"] == 0) & (test_data["Labels"] == 0)])
sast_fp = len(test_data[(test_data["SAST"] == 1) & (test_data["Labels"] == 0)])
sast_fn = len(test_data[(test_data["SAST"] == 0) & (test_data["Labels"] == 1)])

print("TP=",sast_tp)
print("TN=",sast_tn)
print("FP=",sast_fp)
print("FN=",sast_fn)

precision = (sast_tp) / (sast_tp + sast_fp)
recall = (sast_tp) / (sast_tp + sast_fn)
f1_sast = (2*precision*recall) / (precision+recall)
print("F1-score=",f1_sast * 100)


# In[76]:


test_data.to_csv('filtered_bigvul_results.csv', index=False)


# In[ ]:




