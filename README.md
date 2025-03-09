# Playtime Matters -DSA210ProjectProposal

## Project Overview
This project examines Steam games to investigate how game genres (e.g., action, RPG) and prices relate to their playtime lengths and popularity. Using a Steam dataset from Kaggle for game details (genres, prices, estimated ownership) and a HowLongToBeat dataset for completion times, the connection between play duration and player engagement will be analyzed through simple visualizations and statistical methods. Since HowLongToBeat covers games across platforms, the analysis will focus on games available on Steam, leveraging their significant overlap. The goal is to understand how game length influences popularity and pricing trends on Steam.

## Objectives
- **Analyze Game Genres**: The distribution of genres across Steam games will be explored.
- **Quantify Playtime**: Average completion times (Main + Extra) by genre will be calculated for Steam games.
- **Evaluate Popularity**: The relationship between Steam ownership numbers and playtime lengths will be assessed.
- **Identify Trends**: Patterns linking genres, prices, playtime, and popularity will be identified.

## **Motivation**
As a gamer, I’ve always wondered what drives a game’s success on Steam. Some titles demand dozens of hours to complete, while others offer quick, bite-sized experiences. Does the time commitment of a game influence its appeal to players? This project leverages data science to uncover how playtime lengths connect to popularity on Steam, fueled by my passion for gaming and curiosity about what makes certain games stand out.

## Dataset
- **Steam Game Data**:
  - **Source**: Kaggle (“Steam Store Games” - https://www.kaggle.com/datasets/nikdavis/steam-store-games).
  - **Content**: Game name, genre, price (USD), estimated owners.
  - **Sample Data**: “Stardew Valley, Simulation, $14.99, 1M-2M owners”, “The Witcher 3: Wild Hunt, RPG, $39.99, 2M-5M owners”.
- **HowLongToBeat Data**:
  - **Source**: Web scraping from HowLongToBeat (https://howlongtobeat.com).
  - **Content**: Game name, completion time (Main + Extras, hours); data will be collected directly from the site, covering games across multiple platforms with a focus on those available on Steam.
  - **Sample Data**: “Stardew Valley, 93.5 hours (Main + Extras)”, “The Witcher 3: Wild Hunt, 103 hours (Main + Extras)”.
  - **Purpose**: To enrich Steam data with freshly scraped playtime lengths (Main + Extras) for overlapping games and analyze their impact on popularity.
      
## Collection Plan
- **Steam Game Data**:
  - Dataset will be downloaded as a CSV from Kaggle, loaded into Python with `pandas`, and cleaned (e.g., removing duplicates, handling missing values).
- **HowLongToBeat Data**:
  - Data will be scraped from HowLongToBeat (https://howlongtobeat.com) using Python (e.g., `requests` and `BeautifulSoup`), targeting completion times (Main + Extras) for games available on Steam, then merged with Steam data using `pandas` on game names.
    
## **Analysis Plan**

### **Data Preparation**
Data will be cleaned (e.g., standardizing formats, handling missing values) and merged to include only Steam-available games. **Games without a defined endpoint (e.g., competitive multiplayer or battle royale titles like *Dead by Daylight*, *CS:GO*, *PUBG*) will be excluded**, focusing on narrative-driven or completable games. Scraped HowLongToBeat data (Main + Extras) will be aligned with Steam data using game names.

### **Exploratory Data Analysis (EDA)**
- **Bar Charts**: Average price and ownership numbers by genre will be visualized.
- **Scatter Plots**: Ownership vs. completion time (Main + Extras), with points colored by genre, will be plotted.
- **Bar Charts**: Average completion time by genre will be shown.

### **Statistical Analysis**
- **Correlation Analysis**: Relationships between price, ownership numbers, and completion time will be examined.

### **Visualization**
- **Scatter Plots**: Genre-specific trends in playtime and popularity will be explored.
- **Bar Charts**: Top 5 genres by ownership and their average playtime will be compared.
