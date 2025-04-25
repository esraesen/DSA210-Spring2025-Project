# Steam-HLTB Data Prep - DSA210

This document details the steps I took to prepare the Steam and HLTB datasets, including merging, cleaning, handling missing values, and describing the columns.

## Data Preparation Steps

### 1. Merging the Datasets

I merged the Steam dataset (`steam_store_games.csv`) and the HLTB dataset (`howlongtobeat_data.csv`) using the `name` column from the Steam dataset and the `title` column from the HLTB dataset, creating `merged_data.csv`.

```python
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load datasets
steam_games = pd.read_csv("/content/drive/My Drive/DSA210/project/steam_store_games.csv")  # Steam dataset
howlongtobeat_data = pd.read_csv("/content/drive/My Drive/DSA210/project/howlongtobeat_data.csv")  # HLTB dataset

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

This code mounts Google Drive to access the datasets, loads the two datasets, merges them into a single file using the `name` and `title` columns as the key with an inner join (keeping only the rows that match in both datasets), and saves the merged dataset as `merged_data.csv`. I also printed the first 5 rows of each dataset and the merged dataset to verify the data.

### 2. Removing Unnecessary Columns

I removed columns that were not needed for analysis:

- `appid`, `english`, `developer`, `publisher`, `genres_x`, `achievements`, `id`, `title`, `type`, `genres_y`, `release_na`, `release_eu`, `release_jp`, `categories`, `coop`, `versus`, `publishers`, `platforms_x`, `platforms_y`

Additionally, I decided to keep only the `all_styles` column from the HLTB time-related columns and removed the others (`main_story`, `main_plus_extras`, `completionist`) to simplify the analysis and reduce the impact of missing values.

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

# Remove unnecessary HLTB time columns (keep only all_styles)
hltb_columns_to_drop = ['main_story', 'main_plus_extras', 'completionist']
merged_df = merged_df.drop(columns=hltb_columns_to_drop, errors='ignore')
```

This code removes the specified columns that are irrelevant for the analysis. The `errors='ignore'` parameter ensures that the code does not fail if a column is not found. I also removed the HLTB time columns (`main_story`, `main_plus_extras`, `completionist`) to focus on `all_styles`, as it provides an average playtime across all play styles and has fewer missing values compared to the other HLTB columns.

### 3. Renaming Columns

I renamed `steamspy_tags` to `genres` for clarity.

```python
# Rename 'steamspy_tags' to 'genres'
merged_df = merged_df.rename(columns={'steamspy_tags': 'genres'})
```

This renames the `steamspy_tags` column to `genres` for better readability and consistency, as it represents the game genres.

### 4. Converting and Formatting Time Columns

#### 4.1. Ensuring Numeric Values

I converted all time-related columns (`average_playtime`, `median_playtime`, `all_styles`) to numeric values to ensure they can be processed correctly.

```python
# Convert time columns to numeric
time_columns = ['average_playtime', 'median_playtime', 'all_styles']
for col in time_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
```

This ensures that all time columns are numeric. The `errors='coerce'` parameter converts non-numeric values to `NaN`, which helps identify missing or invalid data.

#### 4.2. Steam Playtime Columns

I converted `average_playtime` and `median_playtime` from minutes to hours and rounded them to 2 decimal places.

```python
# Convert Steam playtime columns to hours and round to 2 decimals
merged_df['average_playtime'] = (merged_df['average_playtime'] / 60).round(2)
merged_df['median_playtime'] = (merged_df['median_playtime'] / 60).round(2)
```

This converts the Steam playtime columns from minutes to hours (by dividing by 60) and rounds the results to 2 decimal places for consistency and readability.

#### 4.3. HLTB Playtime Column

I rounded the `all_styles` column (already in hours) to 2 decimal places.

```python
# Round HLTB time columns to 2 decimals
merged_df['all_styles'] = merged_df['all_styles'].round(2)
```

This ensures that the `all_styles` column is rounded to 2 decimal places, maintaining consistency with the Steam playtime columns.

### 5. Handling Missing Values

I checked for missing values in the dataset and found that the `all_styles` column had a significant amount of missing data (48.13%, or 5,818 out of 12,089 rows). The `main_story`, `main_plus_extras`, and `completionist` columns had even higher missing value rates (60.64%, 70.28%, and 63.09%, respectively), which is why I decided to remove them and focus on `all_styles`.

To handle the missing values in `all_styles`, I removed the rows where `all_styles` was missing, as filling them with estimated values (e.g., median) could distort the analysis given the high percentage of missing data.

```python
# Check for missing values
print("\nMissing Values in Each Column:")
print(merged_df.isnull().sum())
print("\nPercentage of Missing Values in Each Column:")
print(merged_df.isnull().sum() / len(merged_df) * 100)

# Drop rows with missing values in all_styles
merged_df = merged_df.dropna(subset=['all_styles'])

# Check for missing values after dropping rows
print("\nMissing Values After Dropping Rows:")
print(merged_df.isnull().sum())
```

This code first checks the number and percentage of missing values in each column. Since `all_styles` had 48.13% missing values, I removed the rows where `all_styles` was missing using `dropna()`. After this step, the dataset was reduced from 12,089 rows to 6,271 rows, but `all_styles` now has no missing values. I also checked the missing values again to confirm that the operation was successful.

### 6. Final Dataset Inspection and Saving

I inspected the final dataset and saved it as `clean_merged_data.csv`, ensuring that floating-point numbers are saved with 2 decimal places.

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

This code prints the first 5 rows of the final dataset, its shape (number of rows and columns), and the list of columns to verify the cleaning process. The dataset is then saved as `clean_merged_data.csv` with floating-point numbers formatted to 2 decimal places.

## Final Columns and Descriptions

The cleaned dataset (`clean_merged_data.csv`) contains the following columns after all processing steps:

| **Column**         | **Description**                                      | **Source** |
|---------------------|-----------------------------------------------------|------------|
| `name`             | Name of the game                                    | Steam      |
| `release_date`     | Release date of the game (YYYY-MM-DD)              | Steam      |
| `required_age`     | Minimum age required to play the game (e.g., 0 for all ages) | Steam      |
| `genres`           | Game genres (e.g., "RPG;Indie;Simulation")         | Steam      |
| `positive_ratings` | Number of positive reviews on Steam                | Steam      |
| `negative_ratings` | Number of negative reviews on Steam                | Steam      |
| `average_playtime` | Average playtime on Steam (hours, rounded to 2 decimals) | Steam      |
| `median_playtime`  | Median playtime on Steam (hours, rounded to 2 decimals) | Steam      |
| `owners`           | Estimated number of owners (e.g., "50000-100000")  | Steam      |
| `price`            | Price of the game on Steam (USD)                   | Steam      |
| `all_styles`       | Average playtime across all play styles (hours, rounded to 2 decimals) | HLTB       |
| `developers`       | Developers of the game                             | HLTB       |

- **Notes on Columns:**
  - The `main_story`, `main_plus_extras`, and `completionist` columns were removed due to their high missing value rates (60.64%, 70.28%, and 63.09%, respectively).
  - The `all_styles` column was kept as it had the lowest missing value rate among the HLTB time columns (48.13%), and rows with missing `all_styles` values were removed.
  - The final dataset has 6,271 rows and 12 columns, reduced from the initial 12,089 rows due to the removal of rows with missing `all_styles` values.

## Sample Data

A sample of the time-related columns in the final dataset:

| **average_playtime** | **median_playtime** | **all_styles** |
|----------------------|---------------------|----------------|
| 4.62                 | 1.03                | 4.50           |
| 3.12                 | 0.57                | 4.50           |
| 4.30                 | 3.07                | 4.50           |
| 10.40                | 6.92                | 6.00           |
| 2.92                 | 0.17                | 4.50           |

## Code

The complete script for the merging and cleaning process is available in `clean_merged_data.py`:
