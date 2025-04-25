# Playtime Matters - DSA210 Project Proposal

## Project Overview
Steam games will be examined to investigate how game genres such as action and RPG, as well as prices, relate to their playtime lengths and popularity. Two datasets from Kaggle will be utilized: a Steam dataset for game details (genres, prices, estimated ownership) and a HowLongToBeat (HLTB) dataset for completion times. These datasets will be merged, and the connection between play duration and player engagement will be analyzed through simple visualizations and statistical methods. The HLTB dataset includes games available on various platforms, so the focus will be on those also available on Steam. The goal is to understand how game length influences popularity and pricing trends on Steam.

## Objectives
- **Analyze Game Genres:** The distribution of genres across Steam games will be explored.
- **Measure the Playtime:** Average completion times by genre will be calculated for Steam games.
- **Evaluate Popularity:** The relationship between Steam ownership numbers and playtime lengths will be assessed.
- **Identify Trends:** Patterns linking genres, prices, playtime, and popularity will be identified.

## Motivation
As someone who enjoys games, I’ve always wondered what makes a game so popular on Steam. Some games offer quick, fun moments, while others take dozens of hours to complete. Does the time a game takes to play affect how much players like it? Curious about what makes some games stand out, this project will leverage data science to explore the connection between gameplay length and popularity on Steam.

## Dataset

### Steam Game Data
- **Source:** Kaggle (“Steam Store Games” - [https://www.kaggle.com/datasets/nikdavis/steam-store-games](https://www.kaggle.com/datasets/nikdavis/steam-store-games)).
- **Content:** Game name, genre, price (USD), estimated owners.
- **Sample Data:** “Stardew Valley, Simulation, $14.99, 1M-2M owners”, “The Witcher 3: Wild Hunt, RPG, $39.99, 2M-5M owners”.

### HowLongToBeat Data
- **Source:** Kaggle
- **Content:** Game name, completion time (hours); the dataset covers games across multiple platforms, but I will focus on those available on Steam.
- **Sample Data:** “Stardew Valley, 93.5 hours”, “The Witcher 3: Wild Hunt, 103 hours”.
- **Purpose:** Analyze the influence of playtime lengths for overlapping games on popularity to enrich Steam data.

## Collection Plan

### Steam Game Data
The dataset will be downloaded as a CSV from Kaggle, loaded into Python using pandas, and prepared for analysis.

### HowLongToBeat Data
This dataset will also be downloaded as a CSV from Kaggle, loaded into Python using pandas, and merged with the Steam data based on matching game names.

## Analysis Plan

### Data Preparation
The two datasets will be merged by matching game names, including only games present in both datasets. The merged dataset will be cleaned by removing irrelevant columns, converting playtime columns from minutes to hours, and addressing missing values by dropping columns and rows with excessive missing data. The cleaned dataset will be saved as `clean_merged_data.csv` for further analysis.

### Exploratory Data Analysis (EDA)
- **Bar Charts:** Average price and ownership numbers by genre will be visualized.
- **Scatter Plots:** Ownership vs. completion time (`average_completion_time`), with points colored by genre, will be plotted.
- **Bar Charts:** Average completion time by genre will be shown.

### Statistical Analysis
- **Correlation Analysis:** Relationships between price, ownership numbers, and completion time will be examined.

### Visualization
- **Scatter Plots:** Genre-specific trends in playtime and popularity will be explored.
- **Bar Charts:** Top 5 genres by ownership and their average playtime will be compared.

## Report

### Introduction
The relationship between game genres, playtime lengths, prices, and popularity on Steam will be analyzed in this project. The main objective is to determine how these factors influence player engagement, as measured by ownership numbers and positive ratings. The hypothesis that longer playtimes lead to higher popularity will be tested. Patterns will be identified to understand what drives game popularity on Steam, potentially informing better game selection or pricing strategies for players and developers.

### Hypothesis
- **Null Hypothesis (H0):** There is no significant relationship between playtime length (`average_completion_time`) and game popularity (`positive_ratings`). An increase in playtime does not significantly affect popularity.
- **Alternative Hypothesis (H1):** There is a significant relationship between playtime length and game popularity. As playtime increases, game popularity also increases.
