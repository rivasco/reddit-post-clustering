# Reddit Post Clustering

This project scrapes posts related to trips on Reddit, preprocesses them, and stores the preprocessed posts in a database. It also generates and stores embeddings based on preprocessed posts in the database. An automation pipeline is also created for automatically repeating this process every 5 minutes. Finally, a clustering algorithm is developed to cluster these posts based on their embeddings and generate corresponding cluster plots to show clustering results and keywords in each cluster.

## Features
- Scrapes posts from Reddit  
- Preprocesses posts  
- Generates embeddings of posts  
- Stores posts and embeddings in database  
- Automation pipeline for scraping posts and updating database  
- Clusters posts using embeddings  
- Generates cluster plots  

## Usage

Create and activate a virtual environment:  
`python -m venv venv`  
`source venv/bin/activate`  

Install dependencies:  
`pip install -r requirements.txt`  

Put Reddit API file in the same directory:  
`.env`  

Run scrape and preprocess script (replace database configuration with your own):  
`python reddit_scraper.py`  

Run embedding and automation script (replace database configuration with your own):  
`python automation.py --interval 10`  
`python automation.py --message "I like traveling"`  

Run clustering script:  
`python clustering.py`
