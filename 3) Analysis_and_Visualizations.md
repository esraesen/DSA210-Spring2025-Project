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

*Visualizations will be added in the next step to further explore the relationships between playtime, popularity, and genres.*
