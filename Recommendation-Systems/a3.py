
# coding: utf-8

# In[1]:



# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


# In[2]:

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


# In[3]:

def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


# In[4]:

def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    temp = []
    for i in range(len(movies)):
        temp.append(tokenize_string(movies['genres'][i]))
    movies['tokens'] = pd.Series(temp,index=movies.index)
    return movies
    pass


# In[5]:

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    vocab = dict()
    for id,i in enumerate(sorted(set(np.concatenate(movies['tokens'].tolist())))):
        vocab[i] = id
    df = defaultdict(lambda:0)
    tf = defaultdict(lambda:0)
    maximum = defaultdict(lambda:0)
    tfidf = defaultdict(lambda:0)
    for id,token in enumerate(movies['tokens'].tolist()):
        m = []
        for t in token:
            tf[t,id] += 1
            m.append(tf[t,id])
        maximum[id] = max(m)
        for t in set(token):
            df[t] += 1
    for id,token in enumerate(movies['tokens'].tolist()):
        for t in token:
            tfidf[t,id] = tf[t,id] / maximum[id] * math.log10(len(movies)/df[t])
    feats = []
    for id,token in enumerate(movies['tokens'].tolist()):
        data = []
        row = []
        col = []
        for t in token:
            if(t in vocab.keys()):
                row.append(0)
                col.append(vocab[t])
                data.append(tfidf[t,id])
        feats.append(csr_matrix((data,(row,col)),shape=(1,len(vocab)),dtype=np.float64))
    #X = csr_matrix((data,(row,col)),dtype=np.float64)
    movies['features'] = pd.Series(feats,index=movies.index)
    #print(movies['features'])
    return tuple((movies,vocab))
    pass


# In[6]:

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


# In[7]:

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    a = a.A
    b = b.A
    return np.dot(a,b.T).sum() / (np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b))))
    pass


# In[8]:

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
<<<<<<< HEAD
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
=======

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

>>>>>>> template/master
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    rate = []
    #for u in set(ratings_test['userId']):        
        #for id,row in ratings_test[ratings_test.userId==u].iterrows():
    for u, target in zip(ratings_test['userId'], ratings_test['movieId']):
        #target = row.movieId
        cos = 0;den = 0;num = 0
        means = []
        target_feats = movies['features'][movies[movies.movieId==target].index[0]]
        for id2,row2 in ratings_train[ratings_train.userId==u].iterrows():
            means.append(row2.rating)
            cos = cosine_sim(movies['features'][movies[movies.movieId==row2.movieId].index[0]],target_feats)
            #print(type(cos))
            if cos > 0:
                num += cos * row2.rating
                den += cos
        if num == 0 or den == 0:
            rate.append(np.mean(means))
        else:
            rate.append(np.divide(num,den))
    return np.asarray(rate)
    pass


# In[9]:

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


# In[10]:

def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


# In[11]:

if __name__ == '__main__':
    main()

