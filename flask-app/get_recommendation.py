import requests
import pandas as pd
import numpy as np
import pickle
from credentials import apiKey

def nmf_recommender(query, ratings, k=5):
    "recommender based on negative matrix factorization "
    users_id = ratings.index.to_list() 
    movies = ratings.columns.to_list()
    
    # load model
    file = open('../models/nmf_model.bin',mode="rb")
    binary = file.read()
    file.close()
    nmf_model = pickle.loads(binary)
    
    # create Q matrix (movie-genre)
    Q_matrix = nmf_model.components_
    
    # create P matrix 
    new_user_ratings = pd.DataFrame(data=query,
                                columns=movies,
                                 index = ['1000'])
    
    query_imputed = new_user_ratings.fillna(0)
    P_new_user_matrix = nmf_model.transform(query_imputed)
    
    # compute recunstructed ratings for new user
    Reconstructed_new_user_ratings = pd.DataFrame(data=np.dot(P_new_user_matrix,Q_matrix),
                                              columns=movies,
                                              index=['1000'])
    # Filter out the movie that the user has rated
    Reconstructed_new_user_ratings.drop(query.keys(),axis=1,inplace=True)
    
    # get the K top movies
    topK = Reconstructed_new_user_ratings.sort_values(['1000'],
                                                axis=1, ascending=False).T.index.to_list()[:k]
    
    return topK

def get_poster(id):
    '''
    take movie id and return movie poster using OMDb API 
    '''
    if len(id) == 6 or len(id) == 5:
        link = 'http://www.omdbapi.com/?i=tt00'+id+'&apikey='+apiKey
    if len(id) == 7 :
        link = 'http://www.omdbapi.com/?i=tt0'+id+'&apikey='+apiKey
    movieInfo = requests.get(link).json()
    poster = movieInfo['Poster']
    return poster


    