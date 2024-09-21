# Abstract 

This study aims to comprehensively analyze the factors influencing mobile app ratings. Understanding these elements is essential to improving user satisfaction and optimizing app performance. The study looks into the application of machine learning algorithms and how variables like user reviews, app cost, and genre affect app ratings. It also examines the emotional reactions of users both before and after the COVID-19 outbreak. In general, the study offers significant perspectives for marketers and app developers to enhance customer contentment and maximize app efficiency.


# Scope of the Project

This project, conducted using comprehensive datasets obtained from Google Play Store and other mobile application platforms, aims to understand the market performance, user experience, and emotional responses of mobile applications. The project will extensively analyze fundamental features such as application names, categories, rating scores, download counts, and sentiment analysis results derived from user reviews.

Key focuses of the project include:

- The study leverages datasets from Google Play Store and other mobile platforms to comprehend the market performance, user experience, and emotional responses associated with mobile applications.

- It will delve into basic features like application names, categories, rating scores, download counts, along with detailed analysis of sentiment analysis results from user reviews, aiming to provide in-depth insights into how each application is perceived among users.

- Comparative analysis will evaluate changes in user perceptions of applications before and after the Covid-19 period.


# Research Questions

- Which features play a more decisive role in influencing app scores: user reviews, app pricing, or app genre? Which features can be best utilized to enhance the accuracy of the score prediction model?

- Which machine learning techniques may be used to fully comprehend the elements influencing the app installations' rating? How do these algorithms utilize features such as emotional content, pricing policies, user reviews, and other external factors to predict app scores effectively?

- Considering the emotional changes in user evaluations before and during COVID-19, what particular machine learning and sentiment analysis approaches can be used? In what ways do these studies most accurately capture the changes in users' emotional reactions before and after the epidemic in various spheres of life?


# Related Works

The internet's growth has allowed users to share their opinions on social media and commercial websites. As a result, analyzing sentiments and emotions in text has become an important area of research. This field focuses on automatically classifying user reviews to gain insights into public sentiment.

Researchers usually classify customer reviews into three categories: positive, negative, or neutral. But, since reviews can be super positive or super negative, using a specific scale to measure how positive or negative they are could make sentiment analysis work better. (Singh et al., 2016).

One way to perform sentiment analysis is by the use of lexicons, which assign a sentiment value to each term. To enhance accuracy in calculating sentiment values from basic summation and mean methods, Jurek et al. (2015) suggested a normalization function. Lexicon-based methods can be categorized into two types: dictionary-based and corpus-based approaches.

In dictionary-based methods, a list of initial words is created and then expanded with words that have similar or opposite meanings (Schouten and Frasincar, 2015). On the other hand, corpus-based methods involve identifying sentiment words that are specific to a particular subject based on their usage in context (Bernabé-Moreno et al., 2020). Another approach suggested by Cho et al. (2014) is a three-step method to improve how polarity is determined based on context, as well as making dictionaries more adaptable for different domains.

Within the field of text classification, researchers have developed several techniques. For example, Dai et al. (2007) used an iterative Expectation-Maximization algorithm to transfer a Naïve Bayes classifier from one domain to another. This method allowed them to apply the classifier to a new context effectively. In their 2008 study, Gao et al. employed a combination of multiple classifiers trained on various source domains to classify target documents by assessing their similarity to a clustering of the target documents.

In cross-domain classification or domain adaptation, text categorization is essential. It was Pan and Yang (2010) who suggested this concept where knowledge can be transferred between two domains that have different distributions but the same labels.

Techniques like Latent Dirichlet Allocation (Blei et al., 2003) or Latent Semantic Indexing (Weigend et al., 1999) uncover hidden correlations among words, thereby enhancing document representations. More recent approaches extract semantic information from terms by utilizing external knowledge bases such as WordNet (Scott and Matwin, 1998) or Wikipedia (Gabrilovich and Markovitch, 2007).

Another approach to sensitivity analysis involves machine learning algorithms, where data sets are classified into training and test sets for model training and analysis. Supervised classification algorithms such as Naïve Bayes, Support Vector Machine (SVM), and decision trees are often used (Gamon, 2004). Bučar et al. (2018) developed a sentiment lexicon and labeled news corpora to analyze sentiments in Slovene texts. They found that Naïve Bayes performed better than SVM. Tiwari et al. (2020) used SVM, Naïve Bayes, and maximum entropy algorithms with n-gram feature extraction on a dataset of movie reviews. They observed that accuracy decreased as the n-gram values increased.

In different research papers, ensemble methods have been explored to tackle the hurdles of sentiment analysis by utilizing mathematical and statistical approaches like Gaussian distributions. However, these models are frequently seen as theoretical and lack real-world application (Buche et al., 2013). On the other hand, in a thorough exploration of machine learning, a separate study employed a variety of methods such as decision trees and neural networks to forecast app rankings by considering numerous features of the apps (Suleman et al., 2019). 

Ratings are crucial because they directly impact an app's visibility and success. Apps with higher ratings are­ more likely to show up in the Google­ Play Store and attract new people­ to try the app. Sentime­nt analysis explores an intriguing realm: de­coding the nuanced expre­ssions embedded within re­views, including the intricate subte­xt conveyed through emojis. Emojis can help share­ feelings that words alone might not show. Studie­s show that emojis can share how people­ feel, and can help pre­dict ratings they might give to an app. Analyzing these alongside textual reviews offers a richer, more dimensional understanding of user opinions (Martens and Johann, 2017).  


# About Preprocessing Data

**Loading Data**

The initial step in the project involves loading the data from the provided CSV files. Utilized four main datasets, each containing specific information about Google Play Store applications and their user reviews. Below are the details of the files used and their contents:

1.	googleplaystore.csv:
o	This file contains comprehensive information about various applications available on the Google Play Store. The data includes details such as app names, categories, ratings, number of reviews, size, installs, type, price, content rating, genres, last updated date, current version, and Android version required.
2.	googleplaystore_user_reviews.csv:
o	This file includes user reviews for the applications listed in the googleplaystore.csv. The data includes the app name, the review text, the sentiment (positive, negative, or neutral), and the sentiment polarity and subjectivity scores.
3.	Apps.csv:
o	Similar to googleplaystore.csv, this file contains information about various applications, from the Covid-19 period. It includes similar attributes such as app names, categories, ratings, number of reviews, size, installs, type, price, content rating, genres, last updated date, current version, and Android version required.
4.	Reviews.csv:
o	This file includes user reviews for the applications listed in Apps.csv. It contains details such as the app name, the review text, the translated review text, the sentiment (positive, negative, or neutral), and the sentiment polarity and subjectivity scores.

After loading the data, merged the datasets to create a comprehensive DataFrame containing all relevant information about the apps and their reviews. The merging process involves creating three main DataFrames: mergedMain, mergedChild, and mergedAll.

1. mergedMain
Combines app data from dfParentApps with their respective user reviews from dfParentReviews.
2. mergedChild
Combines app data from dfChildApps with their respective user reviews from dfChildReviews.
3. mergedAll
Combines the previously merged DataFrame mergedMain with selected columns from mergedChild to include additional review information.


**Handling Outliers**

Managing the outliers the process is straightforward. For every column, the first quartile (Q1) and the third quartile (Q3) which represent the 25th and 75th percentiles were computed. Subsequently, the Interquartile Range (IQR) as the difference between Q3 and Q1, for a better understanding of the spread of the middle 50% of the data. The acceptable range is then determined by computing the upper and lower bounds. Which are set to be '±1.5 x IQR'. Any values outside this range are considered outliers and are replaced with NaN to prevent them from skewing the analysis. 

**Data Cleaning and Formatting**

To prepare the dataset for analysis, several cleaning and formatting steps are performed. First, the 'Size' column was standardized by replacing 'Varies with device' entries with NaN, removing 'M' suffixes for megabytes and converting 'k' suffixes for kilobytes to scientific notation (e-3), ensuring all values were numeric. Rows with missing 'Review Date' entries were dropped to maintain data integrity. Additionally, rows with NaN or non-numeric values in the 'Size' column were also removed to ensure consistency. Numeric columns like 'Installs' were cleaned by removing special characters and converting them to integers, while 'Price' underwent transformations to remove dollar signs and convert to floats. 'Size' and 'Rating' columns were explicitly converted to float types for uniformity. 'Review Date' was converted to datetime format and reformatted to 'dd-mm-yyyy' to standardize its presentation. Finally, missing values in numeric columns were filled with zeros to facilitate accurate analysis. 


**Scaling Numeric Features and Encoding Categorical Variables**

Numeric features such as 'Rating', 'Reviews', 'Size', 'Installs', 'Sentiment_Polarity', and 'Sentiment_Subjectivity' are standardized using StandardScaler() from scikit-learn. Ensuring that features with varying scales contribute equally without dominance due to the higher magnitude, is achieved by standartizing the range of numeric data through scaling. This transformation centers the data around zero and scales it to have unit variance.

Categorical variables including 'Category', 'Type', 'Content Rating', and 'Genres' are transformed into numerical labels using LabelEncoder(). Each unique category within these columns is assigned a unique integer, allowing categorical data to be effectively utilized in machine learning algorithms that require numeric inputs.

The scaled numeric features and encoded categorical variables are merged along the columns axis. The resulting DataFrame "final_df" integrates both types of transformed data, ensuring that all features are now in a format suitable for machine learning tasks.

Utilizing this combined dataset "final_df", analysis and predictions may now be performed without any need for additional preprocessing.

**Splitting the Data for Training and Testing**

The final dataset is first splitted into input features (x) and target value (y). The input features (X) encompass all columns in final_df except for 'Rating', which serves as our target variable.

The parameter "test_size=0.2" specifies that 20% of the data will be reserved for testing, while the remaining 80% will be allocated for training the machine learning models. The "random_state=42" parameter ensures reproducibility by fixing the random seed, thereby ensuring consistent results across different executions.

The training set (X_train and y_train) is used to capture the patterns and the relationships within the data. Models learn from pairing the input features (X_train) and known target values (y_train).

The test set (X_test and y_test) serves as an independent dataset used to evaluate the trained models' performance. It simulates real-world scenarios where models encounter new, blind, previously unseen data.

**Exploratory Data Analysis (EDA)**

For understanding the basic characteristics of the dataset and gain initial insights, basic statistical analysis is performed. 

The "describe()" function generates statistics such as count, mean, standard deviation, minimum, quartiles, and maximum for numerical columns in the dataset. With the information gathered us can understand the data range, evaluate of the quality of the data and detect possible outliers. The underlying pattern, overall distribution, central tendency of the data is overviewed.

**Basic Statistics**

```plaintext
            Rating         Size      Installs   Price  Sentiment_Polarity
count  6534.000000  6534.000000  6.534000e+03  6534.0         2896.000000
mean      4.453336    34.826752  1.048096e+08     0.0            0.308423
std       0.188545    21.970818  2.266175e+08     0.0            0.282917
min       4.100000     4.400000  5.000000e+05     0.0           -0.250000
25%       4.400000    15.000000  1.000000e+07     0.0            0.000000
50%       4.400000    37.000000  5.000000e+07     0.0            0.308333
75%       4.500000    53.000000  1.000000e+08     0.0            0.500000
max       4.900000    77.000000  1.000000e+09     0.0            1.000000

       Sentiment_Subjectivity
count             2896.000000
mean                 0.415136
std                  0.279582
min                  0.000000
25%                  0.000000
50%                  0.502976
75%                  0.600000
max                  0.900000
```
Based on the descriptive statistics provided for various app features, we can derive several meaningful insights:

**Rating Distribution:**

High Satisfaction: The mean rating of 4.45 and the relatively tight standard deviation of 0.19 suggest that users are generally very satisfied with the apps. The ratings are closely clustered around the mean, indicating consistent user experience across different apps.

**App Size:**

Wide Range: App sizes vary significantly, with a range from 4.4 MB to 77 MB. The standard deviation of 21.97 MB indicates considerable variance in app sizes. The high variance points to a diverse set of apps, from lightweight to very large ones.

**Install Counts:**

Right-Skewed Distribution: The mean number of installs is approximately 104.81 million, which is significantly higher than the median of 50 million. This indicates a right-skewed distribution, where a small number of extremely popular apps have very high install counts, inflating the average.
High Variance: With a standard deviation of 226.62 million installs, the data shows high variance, reflecting the disparity between widely popular apps and those with fewer installs.

**Sentiment Polarity:**

Balanced Sentiments: The mean sentiment polarity of 0.31 and the median at the same value indicate a balanced distribution of positive and negative sentiments. However, the standard deviation of 0.28 shows there is some variability in user sentiments.
Diverse Opinions: The range from -0.25 to 1.0 suggests that user opinions are diverse, with some apps receiving highly positive feedback and others receiving more mixed or even negative feedback.

**Sentiment Subjectivity:**

Moderate Subjectivity: The mean sentiment subjectivity of 0.42 and a median of 0.50 imply that reviews tend to be moderately subjective. This indicates that while many reviews are based on personal opinion, there is also a substantial amount of objective feedback.

![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/f20c4202-fc48-4331-b91e-1bd9435e4ed7)

*Figure 1.1: Correlation Between Numerical Features*

Based on the correlation matrix provided, we can make the following observations:

High Correlation Between Installs and Reviews: Installs and Reviews are highly correlated with a correlation coefficient close to 1. This indicates that as the number of installs increases, the number of reviews also increases proportionally.

Significant Relationship Between Size and Reviews: There is a notable correlation between app size and the number of reviews, with a correlation coefficient of 0.60. This suggests that larger apps tend to have more reviews.

Negative Correlation Between Rating and App Size: Since the project's aim is to predict ratings, it is important to note that app size has a negative correlation with ratings (correlation coefficient of -0.34). This relationship has been further analyzed in subsequent steps.

![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/c3b5f5b0-25fb-4ace-a07d-b4cbed17e740)
*Figure 1.2: Pair Plot of Numeric Features*

![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/a05682c7-e983-41c1-ae4a-b76e12206d2e)
*Figure 1.3: Mean Rating by Category*

The category-based analysis reveals that most app categories have remarkably high and consistent mean ratings, generally around 4 or above, indicating widespread user satisfaction across various types of apps. This trend suggests that regardless of the app category, users tend to rate their experiences favorably. Notably, the "GAME" category stands out with the highest mean rating, highlighting the particularly positive reception of gaming apps. This could be attributed to the engaging and entertaining nature of these apps. 

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/a0dab2e3-1e56-4244-80c6-3d8cbe7435d9)

*Figure 1.4: Distribution of Ratings*

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/a1cea90a-618d-47e8-8008-4df2a72f8cb0)

*Figure 1.5: Distribution of App Sizes*

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/cddfc667-2f25-405e-950b-b2bbd1f150fd)

Figure 1.6: Rating vs App Size*

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/a7cdbdc7-5ecc-41a5-b7b7-ed54faa4529b)

*Figure 1.7: Regression Line of Rating and App Size*

The scatter plot with a regression line and the correlation matrix provide valuable insights into the relationship between app size and rating. The scatter plot, enhanced with a regression line, indicates a slight negative trend, where larger app sizes tend to have slightly lower ratings. This observation is quantified in the correlation matrix, which shows a negative correlation coefficient of -0.34 between app size and rating. This suggests a moderate inverse relationship, meaning as the size of the app increases, the rating tends to decrease.

From the scatter plot, it's evident that there is no strong, consistent pattern, but the general trend supports the negative correlation. The regression line further emphasizes this trend, indicating that larger apps might face challenges that impact user ratings, such as performance issues, longer download times, or higher storage requirements.

# Clustering Analysis

**K-Means Clustering**

K-Means clustering was employed to identify distinct groups of apps based on their size and rating. This method was chosen for its efficiency and simplicity in partitioning the data into a predefined number of clusters. By varying the number of clusters (3, 4, and 5), the aim was to understand how different segmentation strategies affect the clustering of apps. K-Means clustering helps uncover patterns and similarities among apps, providing insights into how app size and rating interact and allowing the identification of groups of apps with similar characteristics.

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/1386ac05-b2c4-47b9-8a8f-13efb160c8be)

*Figure 1.8: K-Means Clustering of App Size with 3 Clusters*

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/3b95169b-e57b-4383-aceb-2497fd70b668)

*Figure 1.9: K-Means Clustering of App Size with 4 Clusters*

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/3bf70011-f440-4e54-a780-4f3a679f351b)

*Figure 1.10 K-Means Clustering of App Size with 5 Clusters*

The visualizations of the clustering results with 3, 4, and 5 clusters showed distinct groupings of apps based on their size and rating. These clusters revealed that apps with similar sizes and ratings tend to be grouped together, offering valuable insights into the relationship between these two variables.

**Including More Features in Clustering**

To enhance the clustering analysis, additional features such as 'Reviews', 'Installs', and 'Price' were included along with 'Size' and 'Rating'. These features were selected to provide a more comprehensive view of app characteristics that influence user satisfaction and engagement. By standardizing these features using the StandardScaler, it was ensured that each feature contributes equally to the clustering process, preventing any single feature from dominating due to its scale.

After standardizing the features, K-Means clustering with three clusters was applied to the dataset. To visualize the results of this multi-dimensional clustering, Principal Component Analysis (PCA) was used to reduce the dimensions of the data to two principal components. PCA helps in simplifying the complexity of high-dimensional data while retaining its variance, making it easier to visualize and interpret.

The scatter plot shows the clustering results, with each point representing an app and colored according to its assigned cluster. The axes represent the first two principal components derived from PCA, which capture the most significant variance in the data.

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/b70101f2-3548-4363-892a-6619c2266d3e)

*Figure 1.11: K-Means Clustering with More Features*

**Automated Clustering**

The purpose to automate the clustering process was driven by the need to gain deeper insights into the relationships between app characteristics, specifically 'Size', 'Rating', 'Reviews', 'Installs', and 'Price'. The previous analyses, such as the scatter plots and correlation matrices, indicated that these features play significant roles in determining app performance and user satisfaction. However, those analyses were limited to pairwise relationships and did not capture the multi-dimensional interactions between these features.

Step 1: Check for Missing Values

The initial step involved checking for missing values in the dataset to ensure data quality and completeness. Identifying and addressing missing values is crucial because they can significantly impact the accuracy and reliability of clustering results. This step ensures that the data used for clustering is complete and representative.

Step 2: Select Relevant Features and Drop Rows with Missing Values

Relevant features such as 'Size', 'Rating', 'Reviews', 'Installs', and 'Price' were selected for clustering. Rows with missing values in these selected features were dropped. This step was necessary to maintain the integrity of the dataset and ensure that the clustering algorithm receives consistent and complete data for analysis.

Step 3: Impute Missing Values

For any remaining missing values, a mean imputation strategy was employed. This approach fills missing values with the mean value of the respective feature, ensuring that no data points are excluded from the analysis. This step helps in maintaining a complete dataset without introducing significant bias.

Step 4: Train-Test Split

The dataset was split into training and test sets to evaluate the performance of the clustering model. A 70-30 split was chosen to ensure that a sufficient amount of data was available for both training the model and testing its effectiveness. This step is essential for validating the model and ensuring that it performs well on unseen data.

Step 5: Standardize the Features

Standardizing the features was performed to ensure that each feature contributes equally to the clustering process. Without standardization, features with larger ranges could dominate the clustering process. Standardization transforms the data to have a mean of zero and a standard deviation of one, creating a level playing field for all features.
```plaintext
Train set size: 4573
Test set size: 1961
```

Step 6: Clustering

The optimal number of clusters was determined using the elbow method, which involves plotting the inertia (sum of squared distances of samples to their closest cluster center) against the number of clusters. This method helps in identifying the point where adding more clusters no longer significantly reduces the inertia, indicating the optimal number of clusters. K-Means clustering was then performed with the chosen number of clusters, and the results were added as cluster labels to the dataset.

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/393366bf-6c0d-40ec-9f3f-d030fd02d67f)

*Figure 1.12: Elbow Method*
```plaintext
Cluster
0    2391
2    1923
1     259
```

Step 7: Visualize Clusters

Principal Component Analysis (PCA) was used to reduce the dimensionality of the data for visualization purposes. PCA simplifies the complexity of high-dimensional data while retaining its variance. The clusters were visualized in a 2D plot using the first two principal components, making it easier to interpret and understand the clustering results.

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/d1a51c9c-96a5-4f0f-a435-cba54b3a126e)

*Figure 1.13: Train Set Clusters*


Step 8: Predict Clusters for Test Set

The trained K-Means model was used to predict cluster labels for the test set. This step ensured that the model's performance could be evaluated on unseen data, providing insights into its generalizability and effectiveness.

![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/cea8a928-e2ff-43a6-833b-b61b1e0b1576)

*Figure 1.14: Test Set Clusters*

```plaintext
Cluster
0    1044
2     801
1     116
```

Step 9: Evaluate Clustering Performance

The silhouette score was calculated for both the training and test sets to evaluate the quality of the clusters. The silhouette score measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined and more cohesive clusters. This metric was used to validate the clustering results and ensure that the model effectively captured the underlying structure of the data.
```plaintext
Silhouette Score for Train Set: 0.6197158136999724
Silhouette Score for Test Set: 0.6117722860087376
```

# The score prediction model and Features


- When machine learning models are evaluated, especially when the results obtained from Random Forest , Linear Regression and Gradient Boosting Regressor models are examined, it is clearly seen that factors such as user comments (Reviews) and application type (Genres) affect application scores. It has been determined that especially in the Random Forest Regressor model, the "Reviews" and "Genres" features are decisive and these features are highly effective in score prediction. Therefore, user reviews and app type are among the most effective features for predicting app scores. These features can be optimally used to improve the accuracy of the score prediction model.
  
![Ekran görüntüsü 2024-06-17 154326](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/7b79ca6f-91f5-4068-9d35-e6d51d37cca6)  
*Figure 2.1: Feature Importance for App Ratings*

- On correlation matrix below, it is clearly seen that there is strongest correlation is the positive correlation between "Genre" and "Reviews". These results show that the most effective features for predicting app ratings are user reviews and app type.
  
![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/f2490c71-90d4-433e-81f7-a0c38a2e714f)
*Figure 2.2: Correlation Matrix for Features*

- User reviews are valuable because they provide direct feedback about the quality of the app and user satisfaction. Additionally, the app type also plays an important role in score estimation as it reflects the app's overall category and target audience. App pricing may also be effective in score estimation, but the effect of this factor was seen to be less pronounced compared to others in the models used in this study.

- While the Linear Regression model performed reasonably well with a mean square error of 0.0226 and an R-squared value of 0.5023 on the test set, the Gradient Boosting Regressor model achieved very low error rates and a high R-squared value of 0.9991, predicting application scores extremely accurately. It was seen that he did. Additionally, it was remarkable that the Random Forest Regressor model achieved accuracy scores of up to 100% in the training and test sets, which tends to overfit the training data, showing that especially the "Reviews" (75.654%) and "Genres" (24.346%) features were decisive in score prediction.
  
![Ekran görüntüsü 2024-06-17 154033](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/6c406f68-84d7-46d7-a45d-f9c408b99602)  
*Figure 2.3: Actual vs Predicted Ratings (Linear Regression)*

![Ekran görüntüsü 2024-06-17 154119](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/8210408c-89d4-4f34-bfcd-e8c33c6d7d5b)  
*Figure 2.4: Actual vs Predicted Ratings (Gradient Boosting Regressor)*

![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/cf4936dd-ed8b-4894-95be-2473f83199f4)
*Figure 2.5: Actual vs Predicted Ratings (Random Forest)*

- When different SVM Models are implemented, such output is observed;  

```plaintext
         Linear Kernel SVM Results:
Accuracy: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         9
           1       1.00      1.00      1.00      3441
           2       1.00      1.00      1.00       128
           3       1.00      1.00      1.00       275
           4       1.00      1.00      1.00      8842
           5       1.00      1.00      1.00       853
           6       1.00      1.00      1.00        48
           7       1.00      1.00      1.00      9924
           8       1.00      1.00      1.00       227
           9       1.00      1.00      1.00         7

    accuracy                           1.00     23754
   macro avg       1.00      1.00      1.00     23754
weighted avg       1.00      1.00      1.00     23754

Polynomial Kernel SVM Results:
Accuracy: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         9
           1       1.00      1.00      1.00      3441
           2       1.00      1.00      1.00       128
           3       1.00      1.00      1.00       275
           4       1.00      1.00      1.00      8842
           5       1.00      1.00      1.00       853
           6       1.00      1.00      1.00        48
           7       1.00      1.00      1.00      9924
           8       1.00      1.00      1.00       227
           9       1.00      1.00      1.00         7

    accuracy                           1.00     23754
   macro avg       1.00      1.00      1.00     23754
weighted avg       1.00      1.00      1.00     23754

RBF Kernel SVM Results:
Accuracy: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         9
           1       1.00      1.00      1.00      3441
           2       1.00      1.00      1.00       128
           3       1.00      1.00      1.00       275
           4       1.00      1.00      1.00      8842
           5       1.00      1.00      1.00       853
           6       1.00      1.00      1.00        48
           7       1.00      1.00      1.00      9924
           8       1.00      1.00      1.00       227
           9       1.00      1.00      1.00         7

    accuracy                           1.00     23754
   macro avg       1.00      1.00      1.00     23754
weighted avg       1.00      1.00      1.00     23754

```

When we look below, the class distribution between rating values of apps is given. It appears that the target variable "Rating" has only 10 unique values, and data is distributed quite unevenly. The most common values are 4.7 and 4.4, while the other values are much less frequent. This situation may explain why SVM models are overfitting and showing perfect performance across all classes. 

```plaintext
Class Distribution 
4.7    49931
4.4    44268
4.1    16806
4.5     4215
4.3     1354
4.8     1188
4.2      685
4.6      248
4.0       50
4.9       22
Name: Rating, dtype: int64
```



# Emotional Studies: Before and After the Pandemic

The studies presented in the graphs on “SentimentAnalysisGraphs.ipynb” capture the changes in users' emotional reactions before and after the COVID-19 pandemic in several significant ways.By reading the datasets, column names were standardized and datasets were merged. Sentiment analysis was performed on the “content” column using TextBlob from the merged dataset. This analysis resulted in the calculation of sensitivity, polarity, and subjectivity values ​​for each review. The data was divided into pre-COVID-19 and post-COVID-19 periods and saved in separate CSV files. Both CSV files included sentiment analysis results based on the "content" column. This process aimed to analyze sentiment in app reviews and their changes over time. Here's everything in detail:

- **Pre-COVID Sentiment Distribution**: The pie chart indicates that before the pandemic, a majority of the sentiments were positive (65.1%), with negative sentiments at 26.1%, and neutral sentiments at 8.9%.  
- **Post-COVID Sentiment Distribution**: The second pie chart shows a slight decrease in positive sentiments to 61%, a small decrease in negative sentiments to 24.4%, and an increase in neutral sentiments to 14.6%.

![Ekran görüntüsü 2024-06-17 124639](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/0b46f365-c378-4ef4-958b-a819b8dd258f)

*Figure 2.6: Sentiment Distribution Pre-COVID-19*

![Ekran görüntüsü 2024-06-17 124647](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/34a469e9-4f1a-4e2f-9be4-2ef9a2fcfbf6)

*Figure 2.7: Sentiment Distribution Post-COVID-19*

- The emotion distribution bar chart comparing pre-,and post-COVID reveals an obvious decrease in positive emotions and a little decline in negative emotions, with a notable increase in neutral sentiments following the COVID-19 pandemic. This implies that consumers' emotional responses during the epidemic were less judgmental or unsure.

![Ekran görüntüsü 2024-06-17 124615](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/a7438008-0c94-4303-b2d1-d1f638f19cd4)  
*Figure 2.8: Sentiment Distribution Pre- and Post-COVID-19*

- The line graph shows the evolution of sentiment trends over time; it shows an obvious decrease in positive emotions prior to COVID-19 and an increase in neutral feelings in response, while negative sentiments stay mostly unchanged. This pattern shows a change in user responses, which may have been impacted by the pandemic's stress and uncertainty on a worldwide scale.

![Ekran görüntüsü 2024-06-17 124703](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/d57afbe2-74bc-4a23-9da3-e15b7a9c401a)  
*Figure 2.9: Sentiment Trends in Pre- and Post-COVID-19*

- **Pre-COVID Genre Usage**: The bar chart shows high usage of Education and Photography apps before the pandemic, with other genres having significantly lower usage rates.  
- **During COVID Genre Usage**: There's a noticeable shift in genre usage during the pandemic, with Education and Photography remaining dominant, but there's a slight decrease in their proportion, indicating a diversification in app usage.

![Ekran görüntüsü 2024-06-17 132444](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/99834023-aa77-4f5d-8633-5adea258bc9e)  
*Figure 2.10: Pre-COVID Genre Usage*

![Ekran görüntüsü 2024-06-17 124727](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/461a2d21-c1b9-4359-af50-c386203f9849)  
*Figure 2.11: During COVID Genre Usage*

- The comparison bar chart illustrates how consumption trends for different app genres changed. For instance, during the pandemic, there was a noticeable rise in the use of Tools and Art-Design apps, indicating changes in user requirements and activities. In contrast, Education and Photography apps continued to have significant utilization

![Ekran görüntüsü 2024-06-17 124739](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/2e554eb2-1dd1-4698-9950-949943f1c1ff)  
*Figure 2.12: Genre Usage Comparision*

- Before and during the epidemic, the stacked bar chart offers a thorough analysis of the sentiment distribution across several app categories. Although there is an obvious increase in neutral opinions across all categories throughout the pandemic, areas with significant positive sentiments include Tools, Health & Fitness, and Education. This shows the need to apps that are in Tools, Health & Fitness, and Education genres increased during pandemic. Also, these apps made lives easier and changed the dynamics of education, workout, and business life on daily basis. 

![Ekran görüntüsü 2024-06-17 124801](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/2fc3fbaf-74d4-498d-9994-294be79a2ccf)  
*Figure 2.13: Sentiment Distribution by Category*  

# Additional Analysis on User Behaviour
The visuals presented on “SomeAdditionalAnalaysis.ipynb” capture the patterns in users' behaviours on Google Play Apps in several significant ways.

- The first scatter plot (Installs vs Reviews) visualizes how the number of installs relates to the number of reviews an app receives. This helps us see if there's a correlation between popularity (installs) and user engagement (reviews). As seen on correlation matrix (Figure 1.1) ,there is high correlation between Install numbers and Reviews. Also, when we look for the distribution of Install variable, dataset has only 8 unique values, numbers are not continuous valued. That is why on Installs aspect, there is no high distribution on graphic.
  The number of reviews for apps with one billion installations ranges extensively, from a few thousand to more than five million. This shows that even if there is a broad correlation between high install numbers and high review counts, there is still a significant amount of variety in the activity with which individuals evaluate these well-known apps. The quantity of reviews is comparatively low for apps with mid-range install numbers (0.2 billion to 0.8 billion), indicating that not all moderately successful apps produce a significant volume of user reviews.

```plaintext
"Installs" Unique Values
[    500000    5000000   10000000 1000000000  100000000   50000000
    1000000  500000000]
```
![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/e04a8d43-f5b5-4dc5-ac14-7c48d782e617)

*Figure 3.1: Installs vs Reviews*


- The second scatter plot (Rating vs Installs) examines how app ratings are distributed based on the number of installs. This plot helps us understand if highly installed apps tend to have higher ratings or if there's no clear relationship. Same thing with the previous graphic happens on "Ratings vs Installs". Because that Install values are not distributed continously, again there is no high distribution on graphic. We can observe from graphic that apps that have received ratings of 4.6 or above are often less installed. This implies that the most downloaded apps aren't always the ones with the highest ratings.
![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/f9e1086e-3596-4c69-839c-addb5f40fa03)

*Figure 3.2: Ratings vs Installs*

- The pie chart displays the distribution of reviews based on sentiment (Sentiment). It shows the percentage of reviews categorized as positive, negative, or neutral, providing insights into user sentiment towards the apps. It looks like, users tend to comment on positive aspects of apps. Positive emotion dominates overall, which is encouraging for those mentioned apps. On the other hand, a significant amount of negative feedback indicates that developers should solve certain problems or flaws.
![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/5b63a2b0-3130-461f-a5a9-512a32257c23)

*Figure 3.3: Percentage of Reviews by Sentiment*

- The pie chart below shows that out of all the users who installed the apps in the dataset, about 4.73% of them also left a review. This metric provides insight into user engagement and can be used to gauge how actively users provide feedback after installing an app.
![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/6db23312-062d-4745-9b68-4c05d9b39004)

*Figure 3.4: Percentage of Users Who Left Reviews*

- The analysis of the box plot reveals that apps rated for "Teen" audiences show the most consistency, with tightly clustered ratings around the median. This suggests that apps targeted at teenagers tend to meet user expectations more uniformly compared to those in the "Everyone" category. In contrast, the "Everyone" category exhibits greater variability in ratings, likely due to the broad and diverse user base, which results in a wider range of feedback and ratings. This variability highlights the challenge of satisfying a more general audience compared to a more specific demographic. Despite these differences in variability, the median ratings across all content ratings are relatively similar, hovering around 4.4. This indicates a general trend of high user satisfaction across different content ratings, though the degree of variability differs.
![image](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/e5f29eeb-c2e4-4b55-ad3b-2df773b68f27)
*Figure 3.5: Ratings by Content Rating*


# Future Works

To improve the project that includes score prediction models and insights, the following steps can be implemented to make the scope of the project more comprehensive:

- **Data Augmentation:**
Data augmentation artificially increases the training set by creating modified copies of a dataset using existing data. Generating synthetic data to augment the training set, especially for underrepresented categories or features, might make our model more reliable and stronger.

- **Stacking:**
Combining predictions from multiple models (e.g., Random Forest, Gradient Boosting, Linear Regression) would create a more robust and accurate ensemble model.

- **Topic Modeling:**
Latent Dirichlet Allocation (LDA) is a probabilistic model that generates a set of topics, each represented by a distribution over words, for a given corpus of documents. Implementing LDA to identify key themes and topics within user reviews would significantly improve our predictions.

- **Deep Learning Models:**
To improve the model, implementing learning architectures such as LSTM and Transformer models might be effective for handling sequential and text data.

- **Emotion Detection:**
Using emotion detection algorithms to classify reviews into specific emotional categories (e.g., joy, anger, sadness) would enhance the project’s comprehensiveness in emotion studies.

- **User Segmentation:**
Applying clustering algorithms (e.g., K-means) to segment users based on their reviews, ratings, and installation behaviors is preferable to gain deeper insights into user behavior and preferences through segment specific analytics. This would allow the development of personalized recommendations and targeted marketing strategies based on user segments.


# Datasets Used


1. Google Play Store Apps
   
https://www.kaggle.com/datasets/lava18/google-play-store-apps?select=googleplaystore.csv

2. Google Play Store Apps - User Reviews

https://www.kaggle.com/datasets/lava18/google-play-store-apps?select=googleplaystore_user_reviews.csv

3. Google Play Store Apps Reviews (+110K Comment)-Apps

https://www.kaggle.com/datasets/mehdislim01/google-play-store-apps-reviews-110k-comment?select=Apps.csv

4. Google Play Store Apps Reviews (+110K Comment)-Reviews

https://www.kaggle.com/datasets/mehdislim01/google-play-store-apps-reviews-110k-comment?select=Reviews.csv

## REFERENCES 

A. Buche, D. Chandak, and A. Zadgaonkar, Opinion mining and analysis: a survey, arXiv preprintarXiv:1307.3336, 2013

Bernabé-Moreno J, Tejeda-Lorente A, Herce-Zelaya J, Porcel C, Herrera-Viedma E (2020) A context-aware embeddings supported method to extract a fuzzy sentiment polarity dictionary. Knowledge-Based Systems 190:105236.

Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet allocation. The Journal of Machine Learning research, 3:993–1022.

Bučar, Jože & Žnidaršič, Martin & Povh, Janez. (2018). Annotated news corpora and a lexicon for sentiment analysis in Slovene. Language Resources and Evaluation. 52. 10.1007/s10579-018-9413-3.

Cho H, Kim S, Lee J, Lee JS (2014) Data-driven integration of multiple sentiment dictionaries for lexicon-based sentiment classification of product reviews. Knowledge-Based Systems 71:61–71.

C. Shin, J.-H. Hong, and A. K. Dey, ‘‘Understanding and prediction of
mobile application usage for smart phones,’’ in Proc. ACM Conf. Ubiquitous Comput. (UbiComp), 2012, pp. 173–182.

D. Martens and T. Johann, On the emotion of users in appreviews, in Proc. IEEE/ACM Int. Workshop Emotion Awareness Softw. Eng. (Buenos Aires, Argentina), May 2017

Dai, W., Xue, G.-R., Yang, Q., and Yu, Y. (2007). Transferring naive bayes classifiers for text classification. In Proceedings of the AAAI ’07, 22nd national conference on Artificial intelligence, pages 540–545.

Gabrilovich, E. and Markovitch, S. (2007). Computing semantic relatedness using Wikipedia-based explicit semantic analysis. In Proceedings of the 20th International Joint Conference on Artificial Intelligence, volume 7, pages 1606–1611.

Gamon M (2004) Sentiment classification on customer feedback data: noisy data, large feature vectors, and the role of linguistic analysis. In: COLING 2004: Proceedings of the 20th international conference on computational linguistics, pp 841–847

Gao, Yan & Mas, Jean. (2008). A comparison of the performance of pixel based and object based classifications over images with various spatial resolutions. Online Journal of Earth Science. 2. 27-35.

Jurek-Loughrey, Anna & Mulvenna, Maurice & Bi, Yaxin. (2015). Improved lexicon-based sentiment analysis for social media analytics. Security Informatics. 4. 10.1186/s13388-015-0024-x.

 K. Huang, C. Zhang, X. Ma, and G. Chen, ‘‘Predicting mobile application usage using contextual information,’’ in Proc. ACM Conf. UbiquitousComput. (UbiComp), 2012, pp. 1059–1065.

 M. Suleman, A. Malik, and S. S.Hussain,Google play storeapp ranking prediction using machine learning algorithm, Urdu News Headline, Text Classification by Using Different Machine Learning Algorithms, 2019.

Pan, S. J., Kwok, J. T., and Yang, Q. (2008). Transfer learning via dimensionality reduction. In Proceedings of the AAAI ’08, 23rd national conference on Artificial intelligence, pages 677–682.


S. J. Pan and Q. Yang, "A Survey on Transfer Learning," in IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, Oct. 2010, doi: 10.1109/TKDE.2009.191.

Sam Scott and Stan Matwin. 1998. Text Classification Using WordNet Hypernyms. In Usage of WordNet in Natural Language Processing Systems.

Schouten, Kim & Frasincar, Flavius. (2015). Survey on Aspect-Level Sentiment Analysis. IEEE Transactions on Knowledge and Data Engineering. 28. 1-1. 10.1109/TKDE.2015.2485209.


Singh, Mangal & Nafis, Md Tabrez & Mani, Neel. (2016). Sentiment Analysis and Similarity Evaluation for Heterogeneous-Domain Product Reviews. International Journal of Computer Applications. 144. 16-19. 10.5120/ijca2016910112.

Tiwari P, Mishra BK, Kumar S, Kumar V (2020) Implementation of n-gram methodology for rotten tomatoes review dataset sentiment analysis. In: Cognitive analytics: concepts, methodologies, tools, and applications, IGI Global, pp 689–701.

X. Zou, W. Zhang, S. Li, and G. Pan, ‘‘Prophet: What app you wish touse next,’’ in Proc. ACM Conf. Pervasive Ubiquitous Comput. AdjunctPublication, 2013, pp. 167–170.

Weigend, A. S., Wiener, E. D., and Pedersen, J. O. (1999). Exploiting hierarchy in text categorization. Information Retrieval, 1(3):193–216.
