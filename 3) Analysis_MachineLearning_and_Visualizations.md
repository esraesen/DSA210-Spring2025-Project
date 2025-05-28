# Analysis and Visualizations 

This document contains the statistical analysis and visualizations for the Playtime Matters project. The goal is to explore the relationship between game playtime, genres, prices, and popularity on Steam.

## Hypothesis Testing

The hypothesis test in this project is whether there is a significant relationship between playtime length and game popularity.

- **Null Hypothesis (H0):** There is no significant relationship between playtime length (`average_completion_time`) and game popularity (`positive_ratings`). An increase in playtime does not significantly affect popularity.
- **Alternative Hypothesis (H1):** There is a significant relationship between playtime length and game popularity. As playtime increases, game popularity also increases.

### Results
A Pearson correlation analysis was conducted to test the relationship between `average_completion_time` and `positive_ratings`. The results are as follows:

- **Pearson Correlation Coefficient:** 0.165
- **P-value:** 9.86e-40

Since the p-value is less than 0.05, the relationship is statistically significant. The correlation coefficient of 0.165 indicates a weak positive relationship: as playtime increases, popularity also increases slightly. Therefore, the null hypothesis (H0) is rejected, and the alternative hypothesis (H1) is supported. However, the weak correlation suggests that playtime has a limited practical impact on popularity, and other factors may play a larger role.

### Code
The following Python code was used to perform the Pearson correlation analysis:

```python
import pandas as pd
from scipy.stats import pearsonr

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Read the dataset
df = pd.read_csv('/content/drive/My Drive/DSA210/project/clean_merged_data.csv')

# Perform Pearson correlation test between average_completion_time and positive_ratings
correlation, p_value = pearsonr(df['average_completion_time'], df['positive_ratings'])

# Print the results
print("Pearson Correlation Coefficient:", correlation)
print("P-value:", p_value)

# Interpret the results
if p_value < 0.05:
    print("P-value < 0.05, the relationship is statistically significant.")
    if correlation > 0:
        print("There is a positive relationship: As playtime increases, popularity also increases.")
    elif correlation < 0:
        print("There is a negative relationship: As playtime increases, popularity decreases.")
    else:
        print("There is no relationship.")
else:
    print("P-value >= 0.05, the relationship is not statistically significant.")
```

## Visualizations

Three visualizations were created to further explore the relationships between playtime, popularity, and genres.

### Visualization 1: Scatter Plot (Average Completion Time vs Positive Ratings)

![image](https://github.com/user-attachments/assets/05fc8d32-6fb5-43ab-bbec-fa11f27488c9)

A scatter plot was created to visualize the relationship between `average_completion_time` and `positive_ratings`. The plot shows a slight upward trend, but with significant spread, confirming the weak positive relationship identified in the correlation analysis (coefficient: 0.165). Most games have completion times under 200 hours, with positive ratings generally below 50,000. A few outliers with longer completion times (up to 800 hours) have higher ratings (up to 300,000), but these are rare. This indicates that while longer playtimes are associated with higher popularity, the effect is not strong.

### Visualization 2: Bar Chart (Average Completion Time by Genre)
A bar chart was created to show the average completion time for the top 5 genres:

- **Strategy:** 17.5 hours
- **Adventure:** 9.5 hours
- **Action:** 9.0 hours
- **Indie:** 8.5 hours
- **Casual:** 7.5 hours
  
![image](https://github.com/user-attachments/assets/45f5ebd3-96eb-4973-8476-15b41ba1bb09)

Strategy games have the longest average completion time at 17.5 hours, suggesting that this genre typically involves more complex and time-intensive gameplay. In contrast, Casual games have the shortest completion time at 7.5 hours, indicating shorter and lighter gameplay experiences. Action, Indie, and Adventure genres fall in between, with completion times ranging from 8.5 to 9.5 hours.

### Visualization 3: Bar Chart (Average Ownership by Genre)
A bar chart was created to show the average number of owners for the top 5 genres:

- **Action:** 200,000 owners
- **Strategy:** 175,000 owners
- **Indie:** 100,000 owners
- **Adventure:** 75,000 owners
- **Casual:** 50,000 owners

![image](https://github.com/user-attachments/assets/c10ca8ae-d2a2-4caa-8940-03f3589b4ec7)

Action games have the highest average ownership at 200,000, indicating that this genre is highly popular on Steam. Strategy games follow closely with 175,000 owners, also showing strong popularity. Indie and Adventure genres have moderate ownership numbers at 100,000 and 75,000, respectively. Casual games have the lowest ownership at 50,000, suggesting they appeal to a smaller, more niche audience.

### Code for Visualizations
The following Python code was used to create the visualizations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization 1: Scatter Plot (Average Completion Time vs Positive Ratings)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='average_completion_time', y='positive_ratings', data=df, alpha=0.5)
plt.title('Average Completion Time vs Positive Ratings')
plt.xlabel('Average Completion Time (hours)')
plt.ylabel('Positive Ratings')
plt.savefig('/content/drive/My Drive/DSA210/project/scatter_completion_vs_ratings.png')
plt.close()

# Visualization 2: Bar Chart (Average Completion Time by Genre)
# Process the genres column to get the most common genres
df['genres'] = df['genres'].str.split(';')  # Split genres
genres_exploded = df.explode('genres')  # Explode genres into separate rows
top_genres = genres_exploded['genres'].value_counts().head(5).index  # Get the top 5 genres

# Filter the data to include only the top 5 genres
filtered_df = genres_exploded[genres_exploded['genres'].isin(top_genres)]

# Calculate the average completion time by genre
genre_completion = filtered_df.groupby('genres')['average_completion_time'].mean().sort_values()

# Create a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='average_completion_time', y=genre_completion.index, data=genre_completion.reset_index())
plt.title('Average Completion Time by Genre (Top 5 Genres)')
plt.xlabel('Average Completion Time (hours)')
plt.ylabel('Genre')
plt.savefig('/content/drive/My Drive/DSA210/project/bar_genre_completion.png')
plt.close()

# Visualization 3: Bar Chart (Average Ownership by Genre)
# Convert the 'owners' column to numerical values by taking the midpoint of the range
# Example: "50000-100000" -> 75000
def convert_owners_to_numeric(owners_str):
    if isinstance(owners_str, str):
        # Split the range into lower and upper bounds
        bounds = owners_str.replace(',', '').split('-')
        lower = int(bounds[0])
        upper = int(bounds[1])
        # Calculate the midpoint
        return (lower + upper) / 2
    return 0

# Filter the data to include only the top 5 genres and create a copy
filtered_df = genres_exploded[genres_exploded['genres'].isin(top_genres)].copy()

# Apply the conversion to the owners column using .loc
filtered_df.loc[:, 'owners_numeric'] = filtered_df['owners'].apply(convert_owners_to_numeric)

# Calculate the average number of owners by genre
genre_ownership = filtered_df.groupby('genres')['owners_numeric'].mean().sort_values()

# Create a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='owners_numeric', y=genre_ownership.index, data=genre_ownership.reset_index())
plt.title('Average Ownership by Genre (Top 5 Genres)')
plt.xlabel('Average Number of Owners')
plt.ylabel('Genre')
plt.savefig('/content/drive/My Drive/DSA210/project/bar_genre_ownership.png')
plt.close()
```

# Machine Learning

## Machine Learning Process

### Step 1: Data Preparation
- **Description:** Loaded and cleaned the dataset (`clean_merged_data.csv`). Converted categorical `genres` to numerical using one-hot encoding and normalized `owners` to `owners_numeric`.
- **Code:**
  ```python
  # Import necessary libraries
  import pandas as pd
  import numpy as np

  # Mount Google Drive
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)

  # Load dataset
  df = pd.read_csv('/content/drive/My Drive/DSA210/project/clean_merged_data.csv')

  # Convert genres to one-hot encoding
  df['genres'] = df['genres'].str.split(';')
  df_exploded = df.explode('genres')
  df_encoded = pd.get_dummies(df_exploded, columns=['genres'], prefix='genres')
  df_encoded = df_encoded.groupby(df_encoded.index).sum().reset_index()

  # Convert owners to numeric
  def convert_owners_to_numeric(owners_str):
      if isinstance(owners_str, str):
          bounds = owners_str.replace(',', '').split('-')
          lower = int(bounds[0])
          upper = int(bounds[1])
          return (lower + upper) / 2 / 1000000  # Normalize to millions
      return 0

  df_encoded['owners_numeric'] = df_encoded['owners'].apply(convert_owners_to_numeric)

  # Drop missing values
  df_encoded = df_encoded.dropna()
  ```

### Step 2: Model Training and Evaluation
- **Description:** Split data into training and test sets, trained a Decision Tree Regressor, and evaluated its performance using R² and RMSE metrics.
- **Code:**
  ```python
  # Import machine learning libraries
  from sklearn.model_selection import train_test_split
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import r2_score, mean_squared_error

  # Define features and target
  features = ['average_completion_time', 'owners_numeric'] + [col for col in df_encoded.columns if col.startswith('genres_')]
  X = df_encoded[features]
  y = df_encoded['positive_ratings']

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Decision Tree model
  model = DecisionTreeRegressor(max_depth=10, random_state=42)
  model.fit(X_train, y_train)

  # Predict and evaluate
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  print(f"R² Score: {r2:.3f}")
  print(f"RMSE: {rmse:.0f}")
  ```

### Step 3: Feature Importance Visualization
- **Description:** Created a bar chart to show the importance of features in the model, with `owners_numeric` and `average_completion_time` being the most significant.
- **Code:**
  ```python
  # Import plotting library
  import matplotlib.pyplot as plt
  import os

  # Define output directory
  output_dir = '/content/drive/My Drive/DSA210/project/'
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  # Create feature importance plot
  feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
  feature_importance = feature_importance[feature_importance['importance'] > 0].sort_values(by='importance', ascending=False).head(10)

  plt.figure(figsize=(12, 8))
  plt.barh(feature_importance['feature'], feature_importance['importance'])
  plt.title('Feature Importance in Decision Tree Model (Top 10, Non-Zero)', fontsize=14)
  plt.xlabel('Importance', fontsize=12)
  plt.ylabel('Feature', fontsize=12)
  plt.yticks(fontsize=10)
  plt.tight_layout()
  feature_importance_path = os.path.join(output_dir, 'feature_importance_decision_tree.png')
  plt.savefig(feature_importance_path, bbox_inches='tight')
  plt.close()
  ```

  - **Graph:**
  - ![image](https://github.com/user-attachments/assets/fe7ef9d8-111f-4596-9ac0-8f1d9f9227cd)

  - **Comment:** `owners_numeric` (0.642) and `average_completion_time` (0.195) are the most influential features, indicating that user base and playtime significantly affect positive ratings.

### Step 4: Actual vs Predicted Visualization
- **Description:** Generated a scatter plot to compare actual and predicted `positive_ratings`, showing model performance with R² = 0.574 and RMSE = 15416.
- **Code:**
  ```python
  # Create actual vs predicted plot
  plt.figure(figsize=(10, 8))
  plt.scatter(y_test, y_pred, alpha=0.3, s=30, color='blue')
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
  plt.title('Actual vs Predicted Positive Ratings', fontsize=14)
  plt.xlabel('Actual Positive Ratings', fontsize=12)
  plt.ylabel('Predicted Positive Ratings', fontsize=12)
  plt.text(0.05, 0.95, f'R²: {r2:.3f}\nRMSE: {rmse:.0f}', transform=plt.gca().transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
  plt.tight_layout()
  actual_vs_predicted_path = os.path.join(output_dir, 'actual_vs_predicted_decision_tree.png')
  plt.savefig(actual_vs_predicted_path, bbox_inches='tight')
  plt.close()
  ```

  - **Graph:**
  - ![image](https://github.com/user-attachments/assets/481d734d-a5f4-47d6-b316-f35806022382)

  - **Comment:** The model performs well for low ratings but struggles with higher values (e.g., 400,000+), as shown by deviations from the red line.

### Step 5: Cross-Validation
- **Description:** Performed 5-fold cross-validation to assess the Decision Tree model's generalization ability. Calculated R² and RMSE scores for each fold and visualized the results to evaluate consistency.
- **Code:**
  ```python
    from sklearn.model_selection import cross_val_score
    
    # Perform 5-fold cross-validation for R² score
    cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("Cross-Validation R² Scores:", cv_r2_scores)
    print("Average R²:", cv_r2_scores.mean())
    print("Standard Deviation of R²:", cv_r2_scores.std())
    
    # Perform 5-fold cross-validation for RMSE
    def rmse_scorer(model, X, y):
        return np.sqrt(mean_squared_error(y, model.predict(X)))
    
    cv_mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_mse_scores)
    print("Cross-Validation RMSE Scores:", cv_rmse_scores)
    print("Average RMSE:", cv_rmse_scores.mean())
    print("Standard Deviation of RMSE:", cv_rmse_scores.std())
    
    # Create bar plot for Cross-Validation R² and RMSE scores
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² Scores Bar Plot
    ax[0].bar(range(1, 6), cv_r2_scores, color='#66B2FF', alpha=0.7)
    ax[0].set_title('Cross-Validation R² Scores per Fold', fontsize=12)
    ax[0].set_xlabel('Fold', fontsize=10)
    ax[0].set_ylabel('R² Score', fontsize=10)
    ax[0].axhline(y=cv_r2_scores.mean(), color='red', linestyle='--', label=f'Average R²: {cv_r2_scores.mean():.3f}')
    ax[0].legend()
    
    # RMSE Scores Bar Plot
    ax[1].bar(range(1, 6), cv_rmse_scores, color='#FF9999', alpha=0.7)
    ax[1].set_title('Cross-Validation RMSE Scores per Fold', fontsize=12)
    ax[1].set_xlabel('Fold', fontsize=10)
    ax[1].set_ylabel('RMSE', fontsize=10)
    ax[1].axhline(y=cv_rmse_scores.mean(), color='red', linestyle='--', label=f'Average RMSE: {cv_rmse_scores.mean():.0f}')
    ax[1].legend()
    
    plt.tight_layout()
    cv_scores_path = os.path.join(output_dir, 'cross_validation_scores.png')
    plt.savefig(cv_scores_path, bbox_inches='tight')
    plt.close()
  ```

  - **Results:**
    - Cross-Validation R² Scores: [-0.25363327, 0.85779482, 0.59771106, 0.0316324, 0.60434295]
    - Average R²: 0.368
    - Standard Deviation of R²: 0.412
    - Cross-Validation RMSE Scores: [52100.70, 12197.87, 11885.41, 6664.88, 6003.70]
    - Average RMSE: 17770
    - Standard Deviation of RMSE: 17355
    
 - **Graph:**
  - ![image](https://github.com/user-attachments/assets/a9989614-9b71-42b0-925f-5c9e34ea9960)

  - **Comment:** The cross-validation results show an average R² of 0.368, lower than the single split (0.574), indicating potential overfitting. The high variability in R² (std dev: 0.412) and RMSE (std dev: 17,355) across folds suggests inconsistent performance, possibly due to data imbalance or limited features.

### Step 6: Random Forest Model and Cross-Validation
- **Description:** Trained a Random Forest Regressor and evaluated it using 5-fold cross-validation to compare with Decision Tree. Random Forest was chosen to reduce overfitting through ensemble learning.
- **Code:**
  ```python
    from sklearn.ensemble import RandomForestRegressor
    
    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # Perform 5-fold cross-validation for R² score
    rf_cv_r2_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print("Random Forest Cross-Validation R² Scores:", rf_cv_r2_scores)
    print("Average R²:", rf_cv_r2_scores.mean())
    print("Standard Deviation of R²:", rf_cv_r2_scores.std())
    
    # Perform 5-fold cross-validation for RMSE
    rf_cv_mse_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
    rf_cv_rmse_scores = np.sqrt(-rf_cv_mse_scores)
    print("Random Forest Cross-Validation RMSE Scores:", rf_cv_rmse_scores)
    print("Average RMSE:", rf_cv_rmse_scores.mean())
    print("Standard Deviation of RMSE:", rf_cv_rmse_scores.std())
    
    # Create bar plot for Random Forest Cross-Validation R² and RMSE scores
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² Scores Bar Plot
    ax[0].bar(range(1, 6), rf_cv_r2_scores, color='#66B2FF', alpha=0.7)
    ax[0].set_title('Random Forest Cross-Validation R² Scores per Fold', fontsize=12)
    ax[0].set_xlabel('Fold', fontsize=10)
    ax[0].set_ylabel('R² Score', fontsize=10)
    ax[0].axhline(y=rf_cv_r2_scores.mean(), color='red', linestyle='--', label=f'Average R²: {rf_cv_r2_scores.mean():.3f}')
    ax[0].legend()
    
    # RMSE Scores Bar Plot
    ax[1].bar(range(1, 6), rf_cv_rmse_scores, color='#FF9999', alpha=0.7)
    ax[1].set_title('Random Forest Cross-Validation RMSE Scores per Fold', fontsize=12)
    ax[1].set_xlabel('Fold', fontsize=10)
    ax[1].set_ylabel('RMSE', fontsize=10)
    ax[1].axhline(y=rf_cv_rmse_scores.mean(), color='red', linestyle='--', label=f'Average RMSE: {rf_cv_rmse_scores.mean():.0f}')
    ax[1].legend()
    
    plt.tight_layout()
    rf_cv_scores_path = os.path.join(output_dir, 'random_forest_cv_scores.png')
    plt.savefig(rf_cv_scores_path, bbox_inches='tight')
    plt.close()
  ```
    - **Results:**
    - Random Forest Cross-Validation R² Scores: [0.39804556, 0.59055053, 0.67240391, 0.41721839, 0.31479105]
    - Average R²: 0.479
    - Standard Deviation of R²: 0.132
    - Random Forest Cross-Validation RMSE Scores: [36102.74, 20697.91, 10725.42, 5170.41, 7900.80]
    - Average RMSE: 16119
    - Standard Deviation of RMSE: 11286

  - **Graph:**
  - ![image](https://github.com/user-attachments/assets/b6bc23cb-7408-4884-99bb-0b554286f37c)


  - **Comment:** Random Forest shows an average R² of 0.479 and RMSE of 16,119, with lower variability (std dev R²: 0.132, RMSE: 11,286) compared to Decision Tree, indicating better generalization.



## Conclusion
- **Summary:** The Decision Tree model achieved an R² of 0.574 on a single train-test split, but cross-validation showed a lower average R² of 0.368. Random Forest performed better with an average R² of 0.479 and RMSE of 16,119. Key features like owners_numeric and average_completion_time drive predictions in both models.
- **Model Choice Rationale:** Decision Tree was chosen for its interpretability, while Random Forest was added to improve generalization through ensemble learning.
- **Model Comparison:** Random Forest outperformed Decision Tree with a higher average R² (0.479 vs 0.368) and lower RMSE (16,119 vs 17,770). It also showed lower variability (std dev R²: 0.132 vs 0.412; RMSE: 11,286 vs 17,355), suggesting better consistency due to its ensemble approach.
- **Future Improvements:** Hyperparameter tuning (e.g., max_depth, n_estimators) or collecting more data could further enhance performance.
