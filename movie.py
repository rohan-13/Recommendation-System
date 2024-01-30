import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os

movies = pd.read_csv(r"/Users/rohanpatel/Documents/VS/Python/Movie_Rec/archive/tmdb_5000_movies.csv")
credits= pd.read_csv(r"/Users/rohanpatel/Documents/VS/Python/Movie_Rec/archive/tmdb_5000_credits.csv")

data = movies.merge(credits,on='title')

data = data[['movie_id','title','overview','genres','keywords','cast','crew']]


data.dropna(inplace=True)

#print(data.isnull().sum())

import ast #for converting str to list

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L
data['genres'] = data['genres'].apply(convert)
data['keywords'] = data['keywords'].apply(convert)

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L
data['cast'] = data['cast'].apply(convert_cast)

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
data['crew'] = data['crew'].apply(fetch_director)

data['overview'] = data['overview'].apply(lambda x:x.split())


#removes space in between words (i.e. cast and crew sections )
def remove_space(x):
    l = []
    for i in x:
        l.append(i.replace(" ", ""))
    return l

data['cast'] = data['cast'].apply(remove_space)
data['genres'] = data['genres'].apply(remove_space)
data['keywords'] = data['keywords'].apply(remove_space)
data['crew'] = data['crew'].apply(remove_space)

#Combine multiple columns into one 
data['tags'] = data['overview'] + data['genres'] + data['keywords'] + data['cast'] + data['crew']

#update the dataset to remove the individual columns put into the tag
new_df = data[['movie_id', 'title', 'tags']]

#Convert tags from list to paragraph
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

#Convert all charcters to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

#Convert all titles to lowercase
#new_df['title'] = new_df['title'].str.lower()


import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()

#convert paragraph to list of words
#PorterSteamer is then appending the list to the parargam
##Purpose is to allow searches to be more accurate (i.e love and loved will have similar results)
def stem(x):
    l = []
    for i in x.split():
        l.append(ps.stem(i))
    
    return " ".join(l)

new_df['tags'] = new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

#Give similarity score
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

#print(similarity)

#function to reccommend movie given a certain movie

def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


#check if input in valid
def run_program():
    movie = input("What movie do you want to find similar too: ")
    movie = movie.title()
    print(movie)
    if new_df['title'].eq(movie).any():
        print(recommend(movie))
    else:
        print("Invalid input, try a valid input")
        run_program()

run_program()
        
















