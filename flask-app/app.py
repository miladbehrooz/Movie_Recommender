import pandas as pd
from sklearn.impute import KNNImputer
from flask import Flask, render_template, request
from fuzzywuzzy import process
from get_recommendation import nmf_recommender, get_poster


ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
movies = pd.read_csv("../data/ml-latest-small/movies.csv")
links = pd.read_csv("../data/ml-latest-small/links.csv")
movies_links=movies.merge(links, on='movieId')
movies_links.set_index('title',inplace=True)


df = ratings.merge(movies, on='movieId')
ratings = df.pivot_table(index='userId',columns='title',values='rating')

knn_imputer = KNNImputer(n_neighbors=10)

users_id = ratings.index.to_list() 
movies = ratings.columns.to_list()
ratings = pd.DataFrame(data=knn_imputer.fit_transform(ratings),
            index=users_id,
            columns=movies)

app = Flask(import_name=__name__)

@app.route("/",methods=['POST', 'GET'])
def homepage():
    return render_template('homepage.html')


@app.route("/rating",methods=['POST', 'GET'])

def rating():
   movies_selected = request.form.to_dict()
   movies_selected = list(movies_selected.values())
   movies_selected = movies_selected[0].split(',')
   if movies_selected != ['']:
       movies_selected_real_name = [process.extractOne(x,movies)[0] for x in movies_selected ]
   else:
       movies_selected_real_name = movies_selected   
   return render_template(
        "rating.html",
        movie_list=movies_selected_real_name) 
    
    
@app.route("/recommendations",methods=['POST', 'GET'])
def results():
    user_query = request.args.to_dict()
    
    top  = nmf_recommender(user_query, ratings, k=5)
    posters = [get_poster(str(movies_links.loc[x]['imdbId'])) for x in top]
    top_posters = dict(zip(top, posters))
    return render_template(
        "results.html",
        movies=top_posters 
    )

if __name__ == '__main__':
    
    app.run(debug=True)