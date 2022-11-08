# Movie Recommender App
![demo](demo.gif)
- Used a small dataset of [MovieLens](https://grouplens.org/datasets/movielens/) (100,000 ratings applied to 9,000 movies by 600 users) and the **web-scraped movie posters** from [OMDb API](http://www.omdbapi.com/)
- Implemented the following **recommender methods**:
  - Simple Recommender (recommend the most popular movies)
  - **Non-Negative Matrix Factorization (NMF)**
  - **Collaborative Filtering**
- Built a movie **recommender app** with **Flask** - user can select favorites movies. When user rate selected movies, 5 movies based on the NMF algorithm  are recommended 
## Usage 
- Clone the git repository: ```git clone https://github.com/miladbehrooz/Movie_Recommender.git```
- Get API KEY from [OMDb API](http://www.omdbapi.com/apikey.aspx) and copy it to ```flask-app/credentials.py```
- Install the requirements: ```pip install requirements.txt```
- Run web app locally: ``` python flask-app/app.py```

