# Reddit Data Crawler

Scrape Reddit for public opinions on AI Tools based on post and comments metadata (e.g. upvotes) and export results.

## Requirements:

- beautifulsoup
- numpy
- pandas
- urllib
- requests
- openpyxl

## Setup

- To change which subreddits to scrape from, modify the list "SUBREDDITS" at the top of the script.
- To change the filter used to sort the results (e.g. best, new etc), modify the "sort_type" at the last line of the script

## Usage

To run the script:

```
py reddit_upvoted_comments_crawler.py
```
