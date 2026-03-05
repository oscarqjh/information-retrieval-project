# Opinion Scraper

Scrape Bluesky for public opinions on AI tools, then filter, analyze sentiment, and export results.

## Features

- **Bluesky scraping** — via AT Protocol SDK with 40 curated preset queries
- **Threaded reply scraping** — fetch reply threads with configurable depth
- **Two-layer relevance filtering**
  - Rule-based (inline during scrape): min text length, language check, URL/hashtag density, keyword blocklist, near-duplicate detection
  - ML-powered (post-scrape): zero-shot classification using HuggingFace transformers on GPU
- **Text cleaning** — NLP preprocessing pipeline (HTML removal, contraction expansion, emoji conversion, stop word removal, lemmatization, bot detection)
- **Sentiment analysis** — VADER-based positive/negative/neutral scoring
- **Deduplication** — `post_id` primary key prevents duplicate entries across runs
- **Export** — CSV and JSON output with optional filtering by sentiment and relevance

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

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

## Usage

### Reproducing the Bluesky AI Opinions Experiment

```bash
# 1. Scrape Bluesky using preset queries with reply threads
uv run opinion-scraper preset --with-replies

# 2. Clean — NLP text preprocessing (HTML, contractions, stopwords, lemmatization)
uv run opinion-scraper clean

# 3. Filter — ML relevance classification (zero-shot on GPU)
uv run opinion-scraper filter

# 4. Analyze — VADER sentiment scoring
uv run opinion-scraper analyze

# 5. Report — view results (relevant posts only)
uv run opinion-scraper report --relevant-only

# 6. Export — relevant opinions to CSV
uv run opinion-scraper export -f csv -o opinions.csv --relevant-only --clean-text-only
```

### Custom Queries

```bash
# Scrape specific queries
uv run opinion-scraper scrape -q "ChatGPT" -q "Claude AI" -n 100 --with-replies

# Run the rest of the pipeline
uv run opinion-scraper clean
uv run opinion-scraper filter
uv run opinion-scraper analyze
uv run opinion-scraper export -f csv -o opinions.csv --relevant-only
```

### CLI Reference

#### `scrape`

Scrape opinions from Bluesky.

| Option | Default | Description |
|--------|---------|-------------|
| `-q, --query` | `"AI tools"` | Search query (repeatable) |
| `-n, --max-results` | `100` | Max results per query |
| `--with-replies` | `False` | Also scrape reply threads |
| `--min-replies` | `0` | Min reply count to fetch thread |
| `--reply-depth` | `6` | Max reply depth |

#### `clean`

Clean and preprocess opinion text using an NLP pipeline: HTML stripping, contraction expansion, emoji conversion, lowercasing, URL/number removal, stop word removal, and lemmatization. Rejects bot posts and entries too short after cleaning.

#### `filter`

Run ML relevance classification on unfiltered opinions. Uses `cleaned_text` when available.

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold` | `0.5` | Minimum confidence to accept classification |
| `--model` | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | HuggingFace model |
| `--batch-size` | `64` | Inference batch size |

#### `analyze`

Run VADER sentiment analysis on unanalyzed opinions.

#### `report`

Display sentiment analysis report.

| Option | Default | Description |
|--------|---------|-------------|
| `--relevant-only` | `False` | Exclude spam/off-topic posts |

#### `export`

Export opinions to CSV or JSON.

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format` | (required) | `csv` or `json` |
| `-o, --output` | (required) | Output file path |
| `-s, --sentiment` | `all` | Filter: `all`, `positive`, `negative`, `neutral` |
| `-p, --platform` | `all` | Filter: `all`, `bluesky` |
| `--relevant-only` | `False` | Exclude spam/off-topic posts |
| `--clean-text-only` | `False` | Use cleaned text in the `text` column, drop `cleaned_text`/`clean_status` columns |

#### `preset`

Scrape using curated AI opinion queries (40 queries).

| Option | Default | Description |
|--------|---------|-------------|
| `--with-replies` | `False` | Also scrape reply threads |
| `--min-replies` | `0` | Min reply count to fetch thread |
| `--reply-depth` | `6` | Max reply depth |

### Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `opinions.db` | Path to SQLite database |

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
  export.py           # CSV/JSON export
  scraper/
    base.py           # Abstract base scraper
    bluesky.py        # Bluesky scraper (AT Protocol)

tests/                # 59 tests covering all modules
```

## Pipeline Architecture

```
scrape (--with-replies)
  ├── Fetch posts from Bluesky search API
  ├── Rule-based filter (Layer 1) — drops spam inline
  ├── Store surviving posts to SQLite
  ├── [if --with-replies] Fetch reply threads
  ├── Rule-based filter on replies
  └── Store replies (is_reply=True, parent_post_id set)
         │
         v
clean
  ├── Strip HTML, expand contractions, convert emojis
  ├── Lowercase, remove URLs/numbers/punctuation
  ├── Tokenize, remove stop words, lemmatize
  ├── Reject bot posts and too-short entries
  └── Store cleaned_text + clean_status to DB
         │
         v
filter
  ├── Load opinions where relevance_label IS NULL
  ├── Zero-shot classification on GPU (uses cleaned_text)
  └── Write relevance_score + relevance_label to DB
         │
         v
analyze
  └── VADER sentiment scoring (uses original text)
         │
         v
report / export
  ├── --relevant-only: exclude spam/off_topic
  └── --sentiment: filter by sentiment label
```

## Testing

```bash
uv run pytest tests/ -v
```
