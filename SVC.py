
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
import os
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.utils import shuffle


# In[3]:
os.chdir('C:/Users/GUS9KOR/Documents/Projects/MLWARE1')
trainX = pd.read_csv('train_MLWARE1.csv')
test = pd.read_csv('test_MLWARE1.csv')


# In[4]:

train = shuffle(trainX, random_state=25)
train = train.reset_index()
train.head(10)
test.head(2)
tweets = pd.DataFrame(train.tweet)
tweets.tweet[0]
len(tweets)
train.shape


# # COUNTING word frequency

# In[11]:

total_counts = Counter()
saracstic_counts = Counter()
non_sarcastic_counts = Counter()

for word in train.tweet.values:
    total_counts.update(word.split(' '))
    
print(len(total_counts))


# In[12]:

total_counts.most_common(23)


# In[13]:

vocab = sorted(total_counts, key= total_counts.get , reverse = True)[:8000]
print(vocab[-1], total_counts[vocab[-1]])


# In[14]:

word2idx = {word :i for i, word in enumerate(vocab)}


# In[15]:

word2idx


# In[16]:

print(len(vocab))


# In[17]:

def text_to_vector(text):
    word2vec = np.zeros(len(vocab) , dtype = np.int_)
    
    for word in text.split(' '):
        idx = word2idx.get(word , None)
        if idx is None:
            continue
        else:
            word2vec[idx] = 1
            
    return np.array(word2vec)


# In[18]:

text_to_vector('the motion to the sales are in  a motion')


# In[19]:

word_vectors = np.zeros( (len(tweets),len(vocab)), dtype = np.int_)


# In[20]:

word_vectors.shape


# In[21]:

for i, (_, word) in enumerate(tweets.iterrows()):
    word_vectors[i] = text_to_vector(word[0])


# In[22]:

word_vectors[:5, :20]


# In[23]:

labels = train.label


# In[24]:

Y = (labels=='sarcastic').astype(np.int_)


# In[25]:

test_tweets = pd.DataFrame(test.tweet)
test_word2vectors = np.zeros( (len(test_tweets),len(vocab)), dtype = np.int_)
for i, (_, word) in enumerate(test_tweets.iterrows()):
    test_word2vectors[i] = text_to_vector(word[0])


# In[26]:

test_word2vectors[0]


# In[27]:




# In[32]:

X_train, X_test, y_train, y_test = train_test_split(word_vectors, Y, test_size=0.01, random_state=42)


# In[27]:

Y[0]


# ## BUILD MODEL


# In[52]:

clf = GaussianNB()


# In[53]:

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# In[ ]:

accuracy = clf.score(X_test, y_test)
print("accuracy is",accuracy) 
f1 = f1_score(y_test, y_pred)
print("F1 scor is",f1)
# In[40]:

Yac_test = clf.predict(test_word2vectors)


# In[41]:

Yac_test


# In[42]:

pred = []


# In[43]:

for i in range(len(Yac_test)):
    if(Yac_test[i]>0.5):
        pred.append('sarcastic')
    else:
        pred.append('non-sarcastic')


# In[44]:

len(pred)


# In[45]:

pred = pd.DataFrame(pred)
pred['ID']=test.ID


# In[46]:

cols=[1,0]
pred.columns = ['ID','label']


# In[47]:

pred = pred[cols]


# In[48]:

pred.columns = ['ID','label']


# In[49]:

pred


# In[50]:

pred.to_csv('GNB-1.csv',index=False)

