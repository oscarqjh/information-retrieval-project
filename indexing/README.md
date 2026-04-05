# Group 26 – AI Opinion Search Engine

A web-based opinion search engine that allows users to search and analyse public opinions about Artificial Intelligence topics.

---

## Project Structure

```
opinion_search/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend UI
├── index_to_solr.py        # One-time script to index CSV data into Solr
├── opinions.csv            # Crawled and cleaned dataset (32,648 records)
└── README.md               # This file
```

---

## Requirements

- Python 3.8 or above
- Apache Solr 9.10.1
- Java 17 or above (required by Solr)

---

## Installation & Setup

### Step 1 – Install Python dependencies

```bash
pip3 install flask requests pandas
```

### Step 2 – Download and start Apache Solr

1. Download Solr 9.10.1 from https://solr.apache.org/downloads.html
2. Unzip and navigate into the folder:

```bash
tar -xzf solr-9.10.1.tgz
cd solr-9.10.1
```

3. Start Solr:

```bash
bin/solr start
```

4. Create the core:

```bash
bin/solr create -c opinions
```

### Step 3 – Set up the Solr schema

Run the following commands one by one in your terminal:

```bash
curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"text","type":"text_en","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"cleaned_text","type":"text_en","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"author","type":"string","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"platform","type":"string","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"sentiment_label","type":"string","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"sentiment_score","type":"pfloat","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"likes","type":"pint","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"reposts","type":"pint","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"created_at","type":"string","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"query","type":"string","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"subjectivity_label","type":"string","stored":true}}'

curl -X POST -H 'Content-type:application/json' \
  'http://localhost:8983/solr/opinions/schema' \
  -d '{"add-field": {"name":"polarity_label","type":"string","stored":true}}'
```

### Step 4 – Index the data

Make sure `opinions.csv` is in the same folder as `index_to_solr.py`, then run:

```bash
python3 index_to_solr.py
```

You should see all 33 batches indexed successfully.

### Step 5 – Run the Flask app

```bash
cd opinion_search
python3 app.py
```

### Step 6 – Open the search engine

Open your browser and go to:

```
http://0.0.0.0:5000
```

---

## Features

- **Keyword search** – Search any AI-related term (e.g. ChatGPT, machine learning)
- **Timeline search** – Filter results by date range
- **Sentiment filter** – Filter by Positive, Neutral, or Negative sentiment
- **Sort options** – Sort by Most Relevant, Newest, Oldest, Most Liked, or Most Reposted
- **Sentiment pie chart** – Real-time sentiment breakdown for every query
- **Pagination** – Navigate through results page by page
- **Query speed** – Displays response time in milliseconds for every search

---

## Notes

- Solr must be running before starting the Flask app
- Every time you restart your machine, run `bin/solr start` before `python3 app.py`
- The dataset contains 32,648 records crawled from Bluesky on the topic of Artificial Intelligence
