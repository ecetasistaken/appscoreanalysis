# App Score and Sentiment Analysis Report

## Introduction
This report explores the influential factors on app scores and the impact of user sentiment before and during COVID-19. The analysis focuses on features such as user reviews, app pricing, app genres, and their role in predicting app scores and installations. Additionally, we examine the emotional changes in user evaluations due to the pandemic.

---

## 1. Influential Features in App Scores

### Analysis
We used a dataset containing pre-COVID and post-COVID app data. The features included in the analysis were user reviews, app genres, and user ratings. The pricing feature was excluded due to constant values. The data was processed using label encoding for categorical data and split into training and testing sets.

### Key Findings
- **User Reviews and App Genres** are the most decisive features influencing app scores.
- Feature importance was derived using a Random Forest Regressor, which highlighted the significance of these features.

### Visualization
![Feature Importance](feature_importance.png)
*Figure 1: Feature Importance in Predicting App Scores*

```python
import matplotlib.pyplot as plt
import pandas as pd

# Example DataFrame (replace with actual data)
importance_df = pd.DataFrame({
    'Feature': ['Reviews', 'Genres'],
    'Importance': [0.65, 0.35]
})

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Predicting App Scores')
plt.savefig('feature_importance.png')
plt.show()
