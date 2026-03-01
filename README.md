# Opinion Scraper

Scrape Twitter/X and Bluesky for public opinions on AI tools, then filter, analyze sentiment, and export results.

## Features

- **Multi-platform scraping** — Twitter/X (via twscrape) and Bluesky (via AT Protocol SDK)
- **Threaded reply scraping** — fetch reply threads with configurable depth
- **Two-layer relevance filtering**
  - Rule-based (inline during scrape): min text length, language check, URL/hashtag density, keyword blocklist, near-duplicate detection
  - ML-powered (post-scrape): zero-shot classification using HuggingFace transformers on GPU
- **Sentiment analysis** — VADER-based positive/negative/neutral scoring
- **Deduplication** — `post_id` primary key prevents duplicate entries across runs
- **Export** — CSV and JSON output with optional filtering by sentiment and relevance
- **Preset queries** — curated query sets for AI tool opinions (platform-specific for Twitter vs Bluesky)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- GPU with CUDA support (for ML relevance filtering; 4x H100 recommended)

## Setup

```bash
# Clone and install
git clone <repo-url>
cd information-retrieval-project
uv sync

# Configure credentials
cp .env.example .env  # if available, or create .env manually
```

### Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `BSKY_HANDLE` | Bluesky | Your Bluesky handle (e.g. `user.bsky.social`) |
| `BSKY_PASSWORD` | Bluesky | Bluesky app password |

Twitter/X uses twscrape which manages its own account pool — see [twscrape docs](https://github.com/vladkens/twscrape) for setup.

## Usage

### Full Pipeline

```bash
# 1. Scrape — collect opinions from social media
uv run opinion-scraper scrape -q "ChatGPT" -n 100 -p bluesky

# 2. Filter — ML relevance classification (GPU)
uv run opinion-scraper filter

# 3. Analyze — VADER sentiment scoring
uv run opinion-scraper analyze

# 4. Report — view results
uv run opinion-scraper report --relevant-only

# 5. Export — save to file
uv run opinion-scraper export -f csv -o opinions.csv --relevant-only
```

### Using Preset Queries

Run with curated AI tool opinion queries (14 queries per platform):

```bash
# All platforms
uv run opinion-scraper preset

# Single platform
uv run opinion-scraper preset -p bluesky
uv run opinion-scraper preset -p twitter
```

### CLI Reference

#### `scrape`

Scrape opinions from social media platforms.

| Option | Default | Description |
|--------|---------|-------------|
| `-q, --query` | `"AI tools"` | Search query (repeatable) |
| `-n, --max-results` | `100` | Max results per query |
| `-p, --platform` | `all` | `twitter`, `bluesky`, or `all` |
| `--with-replies` | `False` | Also scrape reply threads |
| `--min-replies` | `0` | Min reply count to fetch thread |
| `--reply-depth` | `6` | Max reply depth (Bluesky only) |

#### `filter`

Run ML relevance classification on unfiltered opinions.

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
| `--relevant-only` | `False` | Exclude spam/off-topic posts |

#### `preset`

Scrape using curated AI opinion queries.

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --platform` | `all` | `twitter`, `bluesky`, or `all` |

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
  analysis.py         # VADER sentiment analysis
  filter.py           # Rule-based spam/noise filter
  relevance.py        # ML zero-shot relevance classifier
  export.py           # CSV/JSON export
  scraper/
    base.py           # Abstract base scraper
    twitter.py        # Twitter/X scraper (twscrape)
    bluesky.py        # Bluesky scraper (AT Protocol)

tests/                # 47 tests covering all modules
```

## Pipeline Architecture

```
scrape (--with-replies)
  ├── Fetch posts from platform search APIs
  ├── Rule-based filter (Layer 1) — drops spam inline
  ├── Store surviving posts to SQLite
  ├── [if --with-replies] Fetch reply threads
  ├── Rule-based filter on replies
  └── Store replies (is_reply=True, parent_post_id set)
         │
         v
filter
  ├── Load opinions where relevance_label IS NULL
  ├── Zero-shot classification on GPU (batched)
  └── Write relevance_score + relevance_label to DB
         │
         v
analyze
  └── VADER sentiment scoring
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
