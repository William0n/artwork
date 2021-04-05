# Tate Museum Artwork 

## Introduction 

Using a data set from the Tate Art Museum, my goal for this project is to analyze the data and see how well a random forest algorithm performs at predicting the artwork's creation year. 

## Packages and Resources Used 
Main Packages:
- Numpy
- Sklearn
- Pandas
- Matplotlib

Data source can be found on the [Tate Art Museum's](https://github.com/tategallery/collection) github page.

## Exploratory Data Analysis 

Taking a look at the plot below, it would appear that the vast majority of the artwork in this collection have heights less than 5000mm and widths less than 3000. 
To see if there might be some sort of trend with a piece of art's dimensions, a plot was made between the artwork's aspect ratio (i.e. width/height) and the year that the piece was created.  

<img src="imgs/width vs length.png"  width = 500/>

As previously suspected, there appears to be some pattern between the artworks' dimensions and their creation dates. Taking a look at the left plot below, there appears to be a slight upwards trend from the 1850s to the 2000s. This trend becomes a lot clearer when looking at the right plot which shows a more englarged image of the left plot. 

<p float="left">
<img src="imgs/aspect ratio vs time.png"  width = 410/>
<img src="imgs/aspect ratio vs time closer.png"  width = 410/>
</p>

Aside from the artworks' dimensions, I thought it might be interested to take a look at who the 5 most popular artists were in this collection. Interestingly, the most popular artist went by the name Joseph Mallord William Turner. As shown below, it would appear that more than 50% of all the artwork in the collection belongs to this artist.

<img src="imgs/top 5 artists.png"  width = 500/>

Similarly, a plot was made to see the top 10 most commonly used materials in these pieces of art.

<img src="imgs/top 10 materials.png"  width = 500/>


## Data Cleaning and Preprocessing 

The data set used for this project contained information for over 69,000 different pieces of artwork. Unfortunately, there were a few rows which had missing values, and thus, they were removed. Along with the removal of rows with missing values, the following was also applied to the data set:
  - All columns removed except for the `year`, `width`, and `height` column 
  - Created a new column based on the `medium` column called `materials`. Each number (0-10) in the column represents the top 10 materials where the value 10 represents any material not in the top 10
    - Below is a table showing the materials and their respective numbers
   - Data split into training and testing sets with a 75% and 25% split respectively

| Number | Material | 
|-------|---------------|
|0 | Graphite on paper|
|1| Oil Paint on canvas|
|2| Screenprint on paper|
|3 |Lithograph on paper |
|4 | Watercolour on paper|
|5 | Etching on paper|
|6 | Graphite & watercolour |
|7 | Ink on paper|
|8 | Intaglio print|
|9 | Photograph, gelatin silver print|
|10 | Other |    

## Modeling 

In order to predict the artwork's creation year, I have decided to use a random forest regression model from the sklearn package. The features which were used in this random forest are the artworks' `width`, `height` and the `material` used to make these pieces. Following this, the random forest consists of 600 trees, a max depth of 600, and max leaf nodes 16. 

```model_rf = RandomForestRegressor(n_estimators = 600, max_leaf_nodes = 16, max_depth = 20, n_jobs = -1)```

## Model Results 

Random Forest Model:
|                | Test          |
| -------------  | ------------- | 
| MAE            | 26.55         |  
| MSE            | 1928.34       | 

To evaluate the performance of this model, the Mean Absolute Error (MAE) and Mean Squared Error (MSE) were used. As shown above, the performance of the random forest model was not the best. It would appear that on average, the model's predictions on the test set were off by about 26 years when compared to the true values.  

### Feature Importance

Taking a quick look at the table below, something rather interesting can be seen. Prior to this, I had assumed that perhaps the most important features were the artworks' width and height. However, the table below seems to suggest that height is the least important feature to this model. 

|Feature               | Score         |
| -------------  | ------------- | 
| Width            | 0.787         |  
| Height            | 0.035       | 
| Material            | 0.177       | 

Note: Total scores of all features should be close to 1

## Improving The Model

To improve upon the previous model, the `RandomizedSearchCV` function from sklearn will be used to create a better model. The model will include some more hyperparameters which will then be tuned using a gridsearch and a 5 fold cross validation. The new model will include the following hyperparameters: `n_estimators`, `max_features`, `max_depth`, `min_sample_split`, `min_samples_leaf`, `bootstrap`. Below shows all the values that will be tested for each hyperparameter: 

```n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 4, 8]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]```



