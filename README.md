# Group 26 -- AI Opinion Search Engine

A web-based opinion search engine that scrapes, classifies, indexes, and searches public opinions about Artificial Intelligence from Bluesky and Reddit.

## Pipeline Overview

```
1. Scrape          Collect posts from Bluesky (AT Protocol) and Reddit
       |
2. Clean           NLP preprocessing (HTML, contractions, emoji, lemmatization)
       |
3. Filter          Rule-based + ML zero-shot relevance classification
       |
4. Analyze         VADER sentiment scoring
       |
5. Export           CSV/JSON output with optional filters
       |
6. Classify        Train & run hierarchical subjectivity/polarity + sarcasm detection
       |
7. Index           Index classified data into Apache Solr
       |
8. Search          Flask web app for querying opinions
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Apache Solr 9.10.1
- Java 17+ (required by Solr)

## Setup

```bash
# Clone and install
git clone <repo-url>
cd information-retrieval-project
uv sync

# Configure credentials
cp .env.example .env  # fill in your Bluesky credentials
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BSKY_HANDLE` | Your Bluesky handle (e.g. `user.bsky.social`) |
| `BSKY_PASSWORD` | Bluesky app password |

---

## Step 1 -- Scrape

### Bluesky

Scrape posts using 40 curated preset queries via the AT Protocol SDK:

```bash
# Scrape using preset queries with reply threads
uv run opinion-scraper preset --with-replies

# Or scrape specific queries
uv run opinion-scraper scrape -q "ChatGPT" -q "Claude AI" -n 100 --with-replies
```

| Option | Default | Description |
|--------|---------|-------------|
| `-q, --query` | `"AI tools"` | Search query (repeatable) |
| `-n, --max-results` | `100` | Max results per query |
| `--with-replies` | `False` | Also scrape reply threads |
| `--min-replies` | `0` | Min reply count to fetch thread |
| `--reply-depth` | `6` | Max reply depth |

### Reddit

Scrape Reddit comments from AI-related subreddits using the standalone crawler script:

```bash
python scripts/reddit_upvoted_comments_crawler.py
```

**Configuration:**

- To change which subreddits to scrape from, modify the `SUBREDDITS` list at the top of the script
- To change the sort filter (e.g. best, new), modify `sort_type` at the last line of the script

**Additional dependencies** (if not already installed): `beautifulsoup4`, `requests`, `openpyxl`

---

## Step 2 -- Clean

Run the NLP text preprocessing pipeline: HTML stripping, contraction expansion, emoji conversion, lowercasing, URL/number removal, stop word removal, and lemmatization. Rejects bot posts and entries too short after cleaning.

```bash
uv run opinion-scraper clean
```

---

## Step 3 -- Filter

Run ML relevance classification on unfiltered opinions. Uses `cleaned_text` when available.

```bash
uv run opinion-scraper filter
```

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold` | `0.5` | Minimum confidence to accept classification |
| `--model` | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | HuggingFace model |
| `--batch-size` | `64` | Inference batch size |

---

## Step 4 -- Analyze

Run VADER sentiment analysis on unanalyzed opinions:

```bash
uv run opinion-scraper analyze
```

---

## Step 5 -- Export

Export the cleaned and filtered data to CSV for classifier training and downstream use:

```bash
# Export relevant opinions to CSV
uv run opinion-scraper export -f csv -o opinions.csv --relevant-only --clean-text-only

# Export with filters
uv run opinion-scraper export -f csv -o opinions.csv --relevant-only --exclude-rejected -p reddit
```

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format` | (required) | `csv` or `json` |
| `-o, --output` | (required) | Output file path |
| `-s, --sentiment` | `all` | Filter: `all`, `positive`, `negative`, `neutral` |
| `-p, --platform` | `all` | Filter: `all`, `bluesky`, `reddit` |
| `--relevant-only` | `False` | Exclude spam/off-topic posts |
| `--exclude-rejected` | `False` | Exclude rejected posts |
| `--clean-text-only` | `False` | Use cleaned text in the `text` column |

---

## Step 6 -- Classify

Classifiers are trained on the exported cleaned data, then applied to annotate the full dataset.

### Hierarchical opinion classification

Uses finetuned RoBERTa-base models:

- **Stage 1:** `neutral` vs `opinionated` (subjectivity detection)
- **Stage 2:** `positive` vs `negative` for opinionated posts only; neutral subjectivity automatically receives neutral polarity

```bash
uv run opinion-scraper classify --model-type finetuned
```

### Training and Evaluating the Classifiers

The classification package under `src/opinion_scraper/classification/` supports three workflows:

#### Ablation study

```bash
uv run opinion-scraper run-hierarchical-ablation \
  --output-dir artifacts/ablation \
  --base-model MoritzLaurer/deberta-v3-large-zeroshot-v2.0
```

Compares:

- `baseline_hierarchical_finetuned` -- hierarchical design + fine-tuning
- `ablation_no_finetuning` -- hierarchical zero-shot NLI without fine-tuning
- `ablation_no_hierarchy` -- single flat 3-class classifier

#### Sarcasm evaluation

```bash
uv run opinion-scraper evaluate-sarcasm-classifier \
  --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0
```

#### Annotate the full CSV

Hierarchical subjectivity/polarity:

```bash
uv run opinion-scraper annotate-hierarchical \
  --subjectivity-model artifacts/ablation/baseline_hierarchical_finetuned/subjectivity \
  --polarity-model artifacts/ablation/baseline_hierarchical_finetuned/polarity \
  --csv-path data/all_opinions.csv \
  --force
```

Sarcasm:

```bash
uv run opinion-scraper annotate-sarcasm \
  --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
  --csv-path data/all_opinions.csv \
  --force
```

Both annotation commands default to GPU (`--device 0`) and write metrics to `data/all_opinions.csv.*.metrics.json`.

---

## Step 7 -- Index into Solr

### Install and start Solr

```bash
# Download Solr 9.10.1 from https://solr.apache.org/downloads.html
tar -xzf solr-9.10.1.tgz
cd solr-9.10.1

# Start Solr and create the core
bin/solr start
bin/solr create -c opinions
```

### Set up the Solr schema

```bash
# Add all required fields
for field in \
  '{"add-field": {"name":"text","type":"text_en","stored":true}}' \
  '{"add-field": {"name":"cleaned_text","type":"text_en","stored":true}}' \
  '{"add-field": {"name":"author","type":"string","stored":true}}' \
  '{"add-field": {"name":"platform","type":"string","stored":true}}' \
  '{"add-field": {"name":"sentiment_label","type":"string","stored":true}}' \
  '{"add-field": {"name":"sentiment_score","type":"pfloat","stored":true}}' \
  '{"add-field": {"name":"likes","type":"pint","stored":true}}' \
  '{"add-field": {"name":"reposts","type":"pint","stored":true}}' \
  '{"add-field": {"name":"created_at","type":"string","stored":true}}' \
  '{"add-field": {"name":"query","type":"string","stored":true}}' \
  '{"add-field": {"name":"subjectivity_label","type":"string","stored":true}}' \
  '{"add-field": {"name":"polarity_label","type":"string","stored":true}}'; do
  curl -X POST -H 'Content-type:application/json' \
    'http://localhost:8983/solr/opinions/schema' -d "$field"
done
```

### Index the data

Make sure `opinions.csv` is in the `indexing/` folder, then run:

```bash
cd indexing
python3 index_to_solr.py
```

---

## Step 8 -- Run the Search Engine

```bash
cd indexing
python3 app.py
```

Open your browser at `http://0.0.0.0:5000`.

### Search Features

- **Keyword search** -- Search any AI-related term (e.g. ChatGPT, machine learning)
- **Timeline search** -- Filter results by date range
- **Sentiment filter** -- Filter by Positive, Neutral, or Negative sentiment
- **Sort options** -- Sort by Most Relevant, Newest, Oldest, Most Liked, or Most Reposted
- **Sentiment pie chart** -- Real-time sentiment breakdown for every query
- **Pagination** -- Navigate through results page by page
- **Query speed** -- Displays response time in milliseconds

> **Note:** Solr must be running before starting the Flask app. Every time you restart your machine, run `bin/solr start` before `python3 app.py`.

---

## Project Structure

```
src/opinion_scraper/
  cli.py              # Click CLI entry point
  config.py           # ScraperConfig with preset queries
  storage.py          # SQLite storage + Opinion dataclass
  cleaner.py          # NLP text preprocessing pipeline
  analysis.py         # VADER sentiment analysis
  filter.py           # Rule-based spam/noise filter
  relevance.py        # ML zero-shot relevance classifier
  classification/     # Hierarchical classification, sarcasm detection, ablation, annotation
  export.py           # CSV/JSON export
  scraper/
    base.py           # Abstract base scraper
    bluesky.py        # Bluesky scraper (AT Protocol)

scripts/
  reddit_upvoted_comments_crawler.py   # Reddit comment scraper
  inter_annotator_agreement_calculator.py
  evaluate_pretrained.py               # Pretrained model evaluation
  finetune_models.py                   # Model finetuning

indexing/
  app.py              # Flask backend
  index_to_solr.py    # Solr indexing script
  templates/
    index.html        # Frontend UI

tests/                # Tests covering scraping, filtering, classification, and export
```

## Testing

```bash
uv run pytest tests/ -v
```

## Global CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `opinions.db` | Path to SQLite database |
