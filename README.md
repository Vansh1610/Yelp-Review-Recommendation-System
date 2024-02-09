# Yelp Restaurants Recommendation System

Welcome to my Yelp Restaurants Recommendation System project! The aim of this project is to develop a personalized recommendation system capable of predicting ratings for places based similarities between businesses. Leveraging Big Data functionalities such as Spark RDD (Resilient Distributed Datasets) and collaborative filtering algorithms, the system analyzes past user ratings and preferences to generate accurate and personalized recommendations.

# Note
The files *app.py* and *recommen.py*, are stored in a private repository and cannot be shared on a public domain due to project guidelines and requirements.


## Features

1. *Item-Item Interaction Analysis*:

    •⁠ Analyze item-item interaction data to identify similarities between items (places).
  
2. *Collaborative Filtering Algorithms*:

    •⁠ Implement item-item collaborative filtering algorithms to identify similar items based on user ratings and business features achieving an RMSE of 1.09.

3. *Model-Based Recommendation*:
   
    • Implement sophisticated machine learning models, such as XGBRegressor, to predict ratings for  businesses, achievign an RMSE of 1.

5. *Hybrid Recommendation System*:
   
    • Constructed a refined hybrid recommendation model combining above techniques by performing feature engineering, yielding RMSE of 0.9792 (validation) and 0.9798 (test).


## Techniques Used

•⁠  *Data Mining*: Utilized data mining techniques to extract meaningful insights from large datasets.

•⁠  *Spark RDD*: Leveraged Spark RDD (Resilient Distributed Datasets) for efficient and distributed data processing.

•⁠  *Flask*: Implemented the web application using Flask, a lightweight and flexible web framework for Python.

•⁠  *Collaborative Filtering*: Employed collaborative filtering algorithms to provide recommendations of places based on the preferences of similar users.

•⁠  *XGBoost*: Utilized XGBoost, a powerful machine learning library, for model-based recommendation and prediction tasks.



## Results


The model-based recommendation system effectively provides accurate ratings for businesses by users, achieving an RMSE of 1. However, the hybrid recommendation, which combines the power of both model-based and collaborative filtering with feature engineering, significantly enhances the RMSE to 0.97

## Usage

1. Clone the Git repository.
   
3. Ensure that you have Spark and PySpark installed and configured in your environment.
   
5. To visualize the project, run python3 app.py (stored in the private repository) in the command line.
   
7. First, select a user from the provided options (10 options available).
   
9. Next, select a business from the provided options (10 options available). On the business page, you have the option to select the user again.
    
11. Voilà! You now have the predicted rating for the business by the user. You also have the option to go back to the home page.
    
13. To create the model, you can run the recommen.py (stored in the private repository).

## Demo Video



https://github.com/Vansh1610/Yelp-Review-Recommendation-System/assets/63463030/ef993810-05e3-43d5-8511-e232d1313116



## Contributors

•⁠  ⁠[Vansh Rajesh Jain]
