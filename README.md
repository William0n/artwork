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

## Model Results 

Random Forest Model:
|                | Training      | Test         | 
| -------------  | ------------- | -------------|
| MAE            |               |         | 
| MSE            |               |         | 


