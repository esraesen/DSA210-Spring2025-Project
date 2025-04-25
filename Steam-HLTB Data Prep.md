# Steam-HLTB Data Prep - DSA210

This document outlines the steps to prepare the Steam and HLTB datasets, including merging, cleaning, handling missing values, and describing the columns.

## Data Preparation Steps

### 1. Merging the Datasets

I merged the Steam dataset (`steam_store_games.csv`) and the HLTB dataset (`howlongtobeat_data.csv`) using the `name` and `title` columns, creating `merged_data.csv`.

```python
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load datasets
steam_games = pd.read_csv("/content/drive/My Drive/DSA210/project/steam_store_games.csv")
howlongtobeat_data = pd.read_csv("/content/drive/My Drive/DSA210/project/howlongtobeat_data.csv")

# Check the datasets
print("First 5 rows of the Steam dataset:")
print(steam_games.head())
print("\nFirst 5 rows of the HLTB dataset:")
print(howlongtobeat_data.head())

# Merge the datasets using 'name' from Steam and 'title' from HLTB
merged_df = pd.merge(steam_games, howlongtobeat_data, how='inner', left_on='name', right_on='title')

# Check the merged dataset
print("\nFirst 5 rows of the merged dataset:")
print(merged_df.head())

# Save merged dataset
merged_df.to_csv('/content/drive/My Drive/DSA210/project/merged_data.csv', index=False)
print("\nMerged dataset saved as 'merged_data.csv'.")
```

This code merges the datasets using an inner join on the `name` and `title` columns and saves the result as `merged_data.csv`.

### 2. Removing Unnecessary Columns

I removed columns not needed for analysis, such as `appid`, `english`, `developer`, `publisher`, `genres_x`, and others.

```python
# Drop unnecessary columns
columns_to_drop = [
    'appid', 'english', 'developer', 'publisher', 'genres_x',
    'achievements', 'id', 'title', 'type', 'genres_y',
    'release_na', 'release_eu', 'release_jp', 'categories',
    'coop', 'versus', 'publishers',
    'platforms_x', 'platforms_y'
]
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')
```

This removes irrelevant columns to simplify the dataset.

### 3. Renaming Columns

I renamed `steamspy_tags` to `genres` for clarity.

```python
# Rename 'steamspy_tags' to 'genres'
merged_df = merged_df.rename(columns={'steamspy_tags': 'genres'})
```

This renames the `steamspy_tags` column to `genres` for better readability.

### 4. Converting and Formatting Time Columns

#### 4.1. Ensuring Numeric Values

I converted time columns (`average_playtime`, `median_playtime`, `main_story`, `main_plus_extras`, `completionist`, `all_styles`) to numeric values.

```python
# Convert time columns to numeric
time_columns = ['average_playtime', 'median_playtime', 'main_story', 'main_plus_extras', 'completionist', 'all_styles']
for col in time_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
```

This ensures time columns are numeric, converting invalid values to `NaN`.

#### 4.2. Steam Playtime Columns

I converted `average_playtime` and `median_playtime` from minutes to hours and rounded to 2 decimal places.

```python
# Convert Steam playtime columns to hours and round to 2 decimals
merged_df['average_playtime'] = (merged_df['average_playtime'] / 60).round(2)
merged_df['median_playtime'] = (merged_df['median_playtime'] / 60).round(2)
```

This converts Steam playtime columns from minutes to hours and rounds them.

#### 4.3. HLTB Playtime Columns

I rounded the HLTB time columns to 2 decimal places.

```python
# Round HLTB time columns to 2 decimals
hltb_time_columns = ['main_story', 'main_plus_extras', 'completionist', 'all_styles']
for col in hltb_time_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].round(2)
```

This rounds HLTB time columns for consistency.

### 5. Handling Missing Values

I checked for missing values and removed columns with more than 50% missing data (`main_story`, `main_plus_extras`, `completionist`). Then, I removed rows with missing values in the remaining time columns.

```python
# Check for missing values
print("\nMissing Values in。各 Column:")
print(merged_df.isnull().sum())
print("\nPercentage of Missing Values in Each Column:")
print(merged_df.isnull().sum() / len(merged_df) * 100)

# Drop columns with more than 50% missing values
missing_percentage = merged_df.isnull().sum() / len(merged_df) * 100
columns_to_drop_high_missing = missing_percentage[missing_percentage > 50].index
merged_df = merged_df.drop(columns=columns_to_drop_high_missing)
print("\nColumns Dropped Due to High Missing Values (>50%):")
print(columns_to_drop_high_missing.tolist())

# Update time_columns after dropping high-missing columns
time_columns = [col for col in time_columns if col in merged_df.columns]
```

This removes columns like `main_story` (60.64% missing), `main_plus_extras` (70.28% missing), and `completionist` (63.09% missing) due to high missing value rates, keeping `all_styles` (48.13% missing).

### 6. Renaming HLTB Time Column

I renamed the `all_styles` column to `average_completion_time` for clarity.

```python
# Rename 'all_styles' to 'average_completion_time'
if 'all_styles' in merged_df.columns:
    merged_df = merged_df.rename(columns={'all_styles': 'average_completion_time'})
    # Update time_columns to reflect the new column name
    time_columns = ['average_playtime', 'median_playtime', 'average_completion_time']
```

This renames the `all_styles` column to `average_completion_time` to better reflect its meaning.

### 7. Removing Rows with Missing Values

I removed rows with missing values in the remaining time columns (`average_playtime`, `median_playtime`, `average_completion_time`).

```python
# Drop rows with missing values in remaining time columns
merged_df = merged_df.dropna(subset=time_columns)

# Check for missing values after dropping rows
print("\nMissing Values After Dropping Rows:")
print(merged_df.isnull().sum())
```

This removes rows with missing values in the remaining time columns, reducing the dataset from 12,089 rows to 6,271 rows.

### 8. Final Dataset Inspection and Saving

I inspected the final dataset and saved it as `clean_merged_data.csv`.

```python
# Check the dataset
print("\nUpdated Dataset - First 5 Rows:")
print(merged_df.head())
print("\nDataset Shape:", merged_df.shape)
print("\nColumns in the Dataset:", merged_df.columns.tolist())

# Save the cleaned dataset
merged_df.to_csv('/content/drive/My Drive/DSA210/project/clean_merged_data.csv', index=False, float_format='%.2f')
print("\nUpdated dataset saved as 'clean_merged_data.csv'.")
```

This saves the cleaned dataset with floating-point numbers rounded to 2 decimal places.

## Final Columns and Descriptions

The cleaned dataset (`clean_merged_data.csv`) contains the following columns:

| **Column**               | **Description**                                      | **Source** |
|---------------------------|-----------------------------------------------------|------------|
| `name`                   | Name of the game                                    | Steam      |
| `release_date`           | Release date of the game (YYYY-MM-DD)              | Steam      |
| `required_age`           | Minimum age required to play the game              | Steam      |
| `genres`                 | Game genres (e.g., "RPG;Indie;Simulation")         | Steam      |
| `positive_ratings`       | Number of positive reviews on Steam                | Steam      |
| `negative_ratings`       | Number of negative reviews on Steam                | Steam      |
| `average_playtime`       | Average playtime on Steam (hours)                  | Steam      |
| `median_playtime`        | Median playtime on Steam (hours)                   | Steam      |
| `owners`                 | Estimated number of owners (e.g., "50000-100000")  | Steam      |
| `price`                  | Price of the game on Steam (USD)                   | Steam      |
| `average_completion_time`| Average playtime across all play styles (hours)    | HLTB       |
| `developers`             | Developers of the game                             | HLTB       |

- The final dataset has 6,271 rows and 12 columns, reduced from 12,089 rows after removing rows with missing values.


## Code

The complete script for the merging and cleaning process is available in `clean_merged_data.py`:

