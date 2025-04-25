# Steam-HLTB Data Prep - DSA210

This document details the steps I took to prepare the Steam and HLTB datasets, including merging, cleaning, and describing the columns.

## Data Preparation Steps

### 1. Merging the Datasets

I merged the Steam dataset (`steam_data.csv`) and the HLTB dataset (`hltb_data.csv`) using the `name` column, creating `merged_data.csv`.

```python
import pandas as pd

# Load datasets
steam_df = pd.read_csv("/content/drive/My Drive/DSA210/project/steam_data.csv")
hltb_df = pd.read_csv("/content/drive/My Drive/DSA210/project/hltb_data.csv")

# Merge on 'name' column
merged_df = pd.merge(steam_df, hltb_df, on='name', how='inner')

# Save merged dataset
merged_df.to_csv('/content/drive/My Drive/DSA210/project/merged_data.csv', index=False)
```

*This code loads the two datasets and merges them into a single file using the `name` column as the key.*

### 2. Removing Unnecessary Columns

I removed columns that were not needed for analysis:

- `appid`, `english`, `developer`, `publisher`, `genres_x`, `achievements`, `id`, `title`, `type`, `genres_y`, `release_na`, `release_eu`, `release_jp`, `categories`, `coop`, `versus`, `publishers`, `platforms_x`, `platforms_y`

```python
columns_to_drop = [
    'appid', 'english', 'developer', 'publisher', 'genres_x', 
    'achievements', 'id', 'title', 'type', 'genres_y', 
    'release_na', 'release_eu', 'release_jp', 'categories', 
    'coop', 'versus', 'publishers',
    'platforms_x', 'platforms_y'
]
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')
```

*This code removes the specified columns that are irrelevant for the analysis.*

### 3. Renaming Columns

I renamed `steamspy_tags` to `genres` for clarity.

```python
merged_df = merged_df.rename(columns={'steamspy_tags': 'genres'})
```

*This renames the `steamspy_tags` column to `genres` for better readability.*

### 4. Converting and Formatting Time Columns

#### 4.1. Steam Playtime Columns

Converted `average_playtime` and `median_playtime` from minutes to hours, rounded to 2 decimal places.

```python
merged_df['average_playtime'] = (merged_df['average_playtime'] / 60).round(2)
merged_df['median_playtime'] = (merged_df['median_playtime'] / 60).round(2)
```

*This converts Steam playtime columns from minutes to hours and rounds to 2 decimal places.*

#### 4.2. HLTB Playtime Columns

Rounded `main_story`, `main_plus_extras`, `completionist`, and `all_styles` (already in hours) to 2 decimal places.

```python
hltb_time_columns = ['main_story', 'main_plus_extras', 'completionist', 'all_styles']
for col in hltb_time_columns:
    merged_df[col] = merged_df[col].round(2)
```

*This ensures HLTB playtime columns are rounded to 2 decimal places for consistency.*

#### 4.3. Ensuring Numeric Values

Converted all time columns to numeric using `pd.to_numeric()`. Saved the dataset with 2 decimal places using `float_format='%.2f'`.

```python
time_columns = ['average_playtime', 'median_playtime', 'main_story', 'main_plus_extras', 'completionist', 'all_styles']
for col in time_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Save with formatted floats
merged_df.to_csv('/content/drive/My Drive/DSA210/project/updated_merged_data.csv', index=False, float_format='%.2f')
```

*This ensures all time columns are numeric and saves the dataset with 2 decimal places.*

## Final Columns and Descriptions

The cleaned dataset (`updated_merged_data.csv`) contains the following columns:

| Column             | Description                                                                    | Source |
|--------------------|--------------------------------------------------------------------------------|--------|
| `name`             | Name of the game                                                               | Steam  |
| `release_date`     | Release date of the game (YYYY-MM-DD)                                          | Steam  |
| `required_age`     | Minimum age required to play the game (e.g., 0 for all ages)                   | Steam  |
| `genres`           | Game genres (e.g., "RPG;Indie;Simulation")                                     | Steam  |
| `positive_ratings` | Number of positive reviews on Steam                                            | Steam  |
| `negative_ratings` | Number of negative reviews on Steam                                            | Steam  |
| `average_playtime` | Average playtime on Steam (hours, rounded to 2 decimals)                       | Steam  |
| `median_playtime`  | Median playtime on Steam (hours, rounded to 2 decimals)                        | Steam  |
| `owners`           | Estimated number of owners (e.g., "50000-100000")                              | Steam  |
| `price`            | Price of the game on Steam (USD)                                               | Steam  |
| `main_story`       | Time to complete the main story (hours, rounded to 2 decimals)                 | HLTB   |
| `main_plus_extras` | Time to complete the main story and extras (hours, rounded to 2 decimals)      | HLTB   |
| `completionist`    | Time to 100% complete the game (hours, rounded to 2 decimals)                  | HLTB   |
| `all_styles`       | Average playtime across all play styles (hours, rounded to 2 decimals)         | HLTB   |
| `developers`       | Developers of the game                                                         | Steam  |

### Sample Data

A sample of the time-related columns:

| average_playtime | median_playtime | main_story | main_plus_extras | completionist | all_styles |
|------------------|-----------------|------------|------------------|---------------|------------|
| 4.62             | 1.03            | 5.50       | 6.00             | 7.50          | 6.00       |
| 3.12             | 0.57            | 0.00       | 4.00             | 5.00          | 4.00       |
| 10.40            | 6.92            | 12.00      | 14.00            | 15.00         | 12.50      |

## Code

The script for the merging and cleaning process: [clean_merged_data.py](clean_merged_data.py)
