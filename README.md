# Playtime Matters-DSA210ProjectProposal
 
## Project Overview
Steam games will be examined to investigate how game genres such as action and RPG, as well as prices, relate to their playtime lengths and popularity. Two datasets from Kaggle will be utilized: a Steam dataset for game details (genres, prices, estimated ownership) and a HowLongToBeat dataset for completion times. These datasets will be merged, and the connection between play duration and player engagement will be analyzed through simple visualizations and statistical methods. Since the HowLongToBeat dataset includes games available on various platforms, the focus will be placed on those available on Steam, the platform where many of these games are already present. The goal is to understand how game length influences popularity and pricing trends on Steam.

## Objectives
- **Analyze Game Genres**: The distribution of genres across Steam games will be explored.
- **Measure the Playtime**: Average completion times (Main + Extra) by genre will be calculated for Steam games.
- **Evaluate Popularity**: The relationship between Steam ownership numbers and playtime lengths will be assessed.
- **Identify Trends**: Patterns linking genres, prices, playtime, and popularity will be identified.

## **Motivation**
As someone who enjoys games, I’ve always wondered what makes a game so popular on Steam. Some games offer quick, fun moments, while others take dozens of hours to complete. Does the time a game takes to play affect how much players like it? Curious about what makes some games stand out, this project will leverage data science to explore the connection between gameplay length and popularity on Steam.

## Dataset
- **Steam Game Data**:
  - **Source**: Kaggle (“Steam Store Games” - https://www.kaggle.com/datasets/nikdavis/steam-store-games).
  - **Content**: Game name, genre, price (USD), estimated owners.
  - **Sample Data**: “Stardew Valley, Simulation, $14.99, 1M-2M owners”, “The Witcher 3: Wild Hunt, RPG, $39.99, 2M-5M owners”.
- **HowLongToBeat Data**:
  - **Source**: Kaggle
  - **Content**: Game name, completion time (Main + Extras, hours); the dataset covers games across multiple platforms, but I will focus on those available on Steam.
  - **Sample Data**: “Stardew Valley, 93.5 hours (Main + Extras)”, “The Witcher 3: Wild Hunt, 103 hours (Main + Extras)”.
  - **Purpose**: The goal is to analyze the influence of newly scraped playtime lengths (Main + Extras) for overlapping games on popularity in order to enrich Steam data.
      
## Collection Plan
- **Steam Game Data**:
  - The dataset will be downloaded as a CSV from Kaggle, loaded into Python using pandas, and prepared for analysis.
- **HowLongToBeat Data**:
  - This dataset will also be downloaded as a CSV from Kaggle, loaded into Python using pandas, and merged with the Steam data based on matching game names.

    
## **Analysis Plan**

### **Data Preparation**
The two datasets will be merged by matching game names, ensuring that only games present in both datasets are included in the analysis. Games without a defined endpoint (e.g., competitive multiplayer or battle royale titles like Dead by Daylight, CS:GO, PUBG) will be excluded, and the focus will be placed on narrative-driven or completable games. After merging, the merged dataset will be cleaned by standardizing formats, handling missing values, removing duplicates, and addressing any inconsistencies in the data.

### **Exploratory Data Analysis (EDA)**
- **Bar Charts**: Average price and ownership numbers by genre will be visualized.
- **Scatter Plots**: Ownership vs. completion time (Main + Extras), with points colored by genre, will be plotted.
- **Bar Charts**: Average completion time by genre will be shown.

### **Statistical Analysis**
- **Correlation Analysis**: Relationships between price, ownership numbers, and completion time will be examined.

### **Visualization**
- **Scatter Plots**: Genre-specific trends in playtime and popularity will be explored.
- **Bar Charts**: Top 5 genres by ownership and their average playtime will be compared.
