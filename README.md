# Tate Museum Artwork 

## Introduction 

Using a data set from the Tate Art Museum, my goal for this project is to analyze the data and see how well a random forest algorithm performs at predicting the artwork's creation 
year. 

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
<img src="imgs/aspect ratio vs time.png"  width = 450/>
<img src="imgs/aspect ratio vs time closer.png"  width = 450/>
</p>

Aside from the artworks' dimensions, I thought it might be interested to take a look at who the 5 most popular artists were in this collection. Interestingly, the most popular artist went by the name Joseph Mallord William Turner. As shown below, it would appear that more than 50% of all the artwork in the collection belongs to this artist.

<img src="imgs/top 5 artists.png"  width = 500/>

Similarly, a plot was made to see the top 10 most commonly used materials in these pieces of art.

<img src="imgs/top 10 materials.png"  width = 500/>


## Data Cleaning and Preprocessing 

## Modeling 

## Model Results 

Random Forest Model:
|                | Training      | Test         | 
| -------------  | ------------- | -------------|
| MAE            |               |         | 
| MSE            |               |         | 


