Overview

The dawn of the new era technology and information has opened floodgates of data - organized raw data. In the last two decades, the Internet has opened various markets, from eCommerce to media streaming services. For example, websites such as Netflix, Youtube, and Amazon have their businesses highly dependent on the kind of recommendation system they have in place.
Recommendation Systems are a type of information filtering systems as they improve the quality of search results and provide items that are more relevant to the search item or are related to the search history of the user. Such systems are used to predict the preference that a user would give to an item. Amazon uses recommendation systems to suggest products to customers (Users who bought this item also bought..) , YouTube uses it to decide which video to play next on autoplay, and Facebook uses it to suggest pages and people you may know.
In this project, we will see the different kinds of recommendation systems and implement them using data manipulation and Python Libraries.
Types of Recommendation Systems
There are basically 3 main types of recommendations systems –
Demographic Filtering- They offer general recommendations to every user, based on movie popularity and/or genre. The simple idea behind this system is that movies that are more critically acclaimed and are popular will have a higher probability of being enjoyed by someone from the average audience.
Content-Based Filtering- They set up correlation between items and suggest similar items given a particular item. This system uses item metadata, such as genre, description, director, actors, etc. for movies, to make these recommendations. The general idea is that someone who likes an item will also like something that is similar to it.
Collaborative Filtering- This recommender matches people with similar interests and based on this it provides recommendations. Collaborative filters do not require item metadata. The main idea behind collaborative methods is that the past user-item interactions are enough to detect similar users and/or that some users have an interest in similar items and make predictions based on these estimated relationships.
Hybrid Systems- Hybrid recommendation systems tend to bring multiple recommendation techniques together and try to get the best of multiple worlds. For example a hybrid of content based and collaborative filtering.
Implementation
We will start by creating a simple demographic filtering system, using a similar rating system used by IMDb to predict popular movies across genres. Then we will pick up a subset from our dataset (Due to limited computation power) and use cosine similarity function to implement Content-based filtering. We will then use singular vector decomposition (which became popular in 2009 from Netflix’s $1 million competition) from the Surprise library in Python to implement collaborative filtering.
For the purpose of this project, we will use “The Movies Dataset” by Rounak Banik from Kaggle. Following is the API command to add the dataset in your python notebook
kaggle datasets download -d rounakbanik/the-movies-dataset
Demographic Filtering
After loading the dataset, we can use the average ratings of the movies as the parameter to compare but using it will not be fair since a movie with a 9.1 average rating and 4 votes cannot be considered better than the movie with an 7.9 as average rating but 75 votes. we will calculate the average rating for each movie as per IMDB's weighted rating (WR) and sort them in decreasing order of their ratings -

v is the vote count for the movie
m is the minimum votes required to be listed 
R is the average rating for a movie
C is the mean vote average of all movies
We have v and R from the dataset. We can calculate C by taking the average of R for every movie. Now, we get m by taking the top 5% of most voted movies for comparison (quantile(x) function). This gives m to be 434.
We add a column of weighted ratings (WR) in the dataframe and sort it in descending order of values in WR. Taking the top 10 movies gives -

We can also get a genre-specific recommendation by extracting different genres from the above “genre” field. For example, top “Action” movies are-


Advantages of this system: Recommends better critically acclaimed movies.
Drawbacks of this system: Does not capture the taste of the user; everybody will receive the same recommendations.
Content-Based Filter
Next, for the content-based filter, we calculate Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each entry in the “overview” column of our dataset. Then we use cosine similarity scores to check how each movie is related to other movies. 
Term Frequency (TF)- It is a measure of how frequently a term, t, appears in a document.

Inverse Document Frequency (IDF)- IDF is a measure of how important a term is. We need the IDF value because computing just the TF alone is not sufficient to understand the importance of words:

The product of the two matrices is the TF-IDF matrix.
We calculate the cosine similarity matrix and return the top 10 movies similar to the input movie (Excluding the first movie). For the movie “Inception”, recommendations on the basis of “overview” of the movie are:

We use a similar method to make predictions on the basis of credits, genres, and keywords by making a metadata soup and creating a Term Frequency matrix for the same(Penalizing a movie just because members of it’s cast have appeared in many different movies doesn’t make much sense). We return the movies with the highest similarity score for a given movie. A drawback is that everyone who watched a given movie will get the same suggestions. Also, it is only capable of suggesting movies which are close to a certain movie.


Collaborative Filtering
For collaborative filtering, we will first have a sparse vector for the ratings each user has given for a movie (Hence of order users X movies). A blank represents not yet rated by the user. These are the blanks that we need to fill. So we will use singular vector decomposition from the Python library Surprise to generate two vectors whose product will approximately give us the ratings of the original user X movie matrix we had but along with it, it will also give predicted ratings for a movie that the user never saw. We will aim to minimize RMSE and make accurate predictions for ratings. Input will be userID and MovieId.
The SVD() algorithm automatically takes care of the number of features it needs to successfully achieve both tasks - minimum error in generating old entries and making accurate predictions on the missing entries.
Running our SVD algorithm 5 folds on our dataset gives an average Root Mean Square Error (RMSE) of 0.895.

If we do a train-test split of 0.25, we get an RMSE of 0.8975 on our test set.

Hybrid Recommendations -
Now we aim a little higher - we try to create hybrid recommendations based on content based and collaborative filtering. Our goal is to first generate 50 movies that are closest to the input movie. Then for those 50 movies, we take the movie ID and user ID to generate estimated ratings for the given user. We  sort the list of 50 movies, and return the top 10.
Suppose a 2 users with userID 1 and 5 watch the movie “Pirates of the Caribbean: The Curse of the Black Pearl” on Netflix. So they’ll get different recommendations - 
Input will be userID and MovieId. 


Conclusion
This project served as a beacon to develop an insight about recommendation systems. We started by defining what recommendation systems are and what role they play in our today's world. We then looked into the three main types of recommendation systems - Demographic, Content-based and Collaborative. We also defined the idea behind hybrid engines. We then went on to implement the recommendation techniques, concluding with a Hybrid recommendation engine that takes in user ID and movieID and gives personalized recommendation for movies.
Implemented Demographic filtering
This filter used Vote Averages and TMDB Vote Count to build Top Movies Charts. We created both general and genre based charts. We used IMDB’s Weighted Rating System to calculate ratings and then sorted them in descending order.
Implemented Content-Based filtering
By generating TF-IDF matrix and cosine similarity functions we built two content based engines. The first took movie overview and taglines as input and the other which took keywords and other took movie metadata to come up with predictions.
Implemented Collaborative filtering
We used the insanely powerful surprise library to create our very own collaborative filter using Singular Value Decomposition. The RMSE obtained was less than 1 (Which is good enough to make predictions) and our recommender gave estimated ratings given a user and a movie. 
Implemented a Hybrid Recommender
We combined the techniques of Content based and Collaborative filtering to create a Hybrid engine. We first shortlisted 50 most similar movies and fed them to our collaborative engine to estimate ratings for a user then sorting was done as per the scores.


References
https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101#Collaborative-Filtering-model
https://medium.com/@shengyuchen/how-tfidf-scoring-in-content-based-recommender-works-5791e36ee8da
https://towardsdatascience.com/creating-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-cc8b431618af
https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26
Adomavicius, Gediminas, and Alexander Tuzhilin. "Context-aware recommender systems." Recommender systems handbook. Springer, Boston, MA, 2011. 217-253.
https://www.ijert.org/research/recommender-systems-types-of-filtering-techniques-IJERTV3IS110197.pdf
https://www.youtube.com/watch?v=d7iIb_XVkZs
https://surprise.readthedocs.io/en/stable/
https://surprise.readthedocs.io/en/stable/getting_started.html#getting-started
https://medium.com/@m_n_malaeb/singular-value-decomposition-svd-in-recommender-systems-for-non-math-statistics-programming-4a622de653e9
https://surprise.readthedocs.io/en/stable/accuracy.html
Image courtesy - Google images, screenshots from youtube tutorials and kaggle notebooks.
