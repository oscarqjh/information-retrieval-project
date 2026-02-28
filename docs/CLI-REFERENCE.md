# Opinion Scraper CLI Reference

## Overview

```
uv run opinion-scraper [OPTIONS] COMMAND [ARGS]
```

A CLI tool that scrapes Twitter/X and Bluesky for public opinions on AI tools, analyzes sentiment, and exports results.

### Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db TEXT` | `opinions.db` | Path to SQLite database file |
| `--help` | | Show help and exit |

```bash
# Use a custom database
uv run opinion-scraper --db my_data.db scrape -q "AI tools" -p bluesky
```

---

## Commands

### `scrape`

Scrape opinions from Twitter/X and/or Bluesky.

```
uv run opinion-scraper scrape [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--query TEXT` | `-q` | `"AI tools"` | Search query. Can be specified multiple times. |
| `--max-results INT` | `-n` | `100` | Maximum results to collect per query. |
| `--platform CHOICE` | `-p` | `all` | Platform to scrape: `twitter`, `bluesky`, or `all`. |

**Examples:**

```bash
# Basic scrape from Bluesky
uv run opinion-scraper scrape -q "AI tools" -n 20 -p bluesky

# Multiple queries
uv run opinion-scraper scrape -q "ChatGPT" -q "Claude AI" -n 50 -p bluesky

# Scrape from all platforms
uv run opinion-scraper scrape -q "AI tools" -n 100 -p all

# Twitter only
uv run opinion-scraper scrape -q "AI tools" -p twitter
```

**Platform credentials:**

- **Bluesky** reads `BSKY_HANDLE` and `BSKY_PASSWORD` from `.env` (or environment variables). Falls back to interactive prompt if not set.
- **Twitter/X** requires accounts added to the twscrape pool (see Setup below).

**Rate limiting:** Requests are automatically paced with randomized 1-3 second delays between pagination batches to mimic human behavior.

---

### `analyze`

Run VADER sentiment analysis on all unanalyzed opinions in the database.

```
uv run opinion-scraper analyze
```

No options. Processes all opinions that haven't been scored yet and updates the database.

Each opinion gets:
- **sentiment_score**: float from -1.0 (most negative) to +1.0 (most positive)
- **sentiment_label**: `positive` (score >= 0.05), `negative` (score <= -0.05), or `neutral`

**Example:**

```bash
uv run opinion-scraper scrape -q "AI tools" -n 50 -p bluesky
uv run opinion-scraper analyze
```

---

### `report`

Display a sentiment analysis summary in the terminal.

```
uv run opinion-scraper report
```

No options. Shows:
- Total opinions collected per platform
- Sentiment breakdown (positive/negative/neutral counts and percentages)
- Average sentiment score
- Sample positive and negative opinions

**Example output:**

```
=== Opinion Scraper Report ===

Total opinions: 50
  bluesky: 50

Sentiment breakdown:
  Positive: 28 (56.0%)
  Negative: 12 (24.0%)
  Neutral:  10 (20.0%)
  Average score: 0.1523

Sample positive opinions:
  [bluesky] @alice.bsky.social: AI tools have genuinely improved my workflow...

Sample negative opinions:
  [bluesky] @bob.bsky.social: These AI tools are overhyped garbage...
```

---

### `export`

Export opinions from the database to CSV or JSON files.

```
uv run opinion-scraper export [OPTIONS]
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--format CHOICE` | `-f` | Yes | | Output format: `csv` or `json`. |
| `--output PATH` | `-o` | Yes | | Output file path. |
| `--sentiment CHOICE` | `-s` | No | `all` | Filter: `all`, `positive`, `negative`, or `neutral`. |

**Examples:**

```bash
# Export everything to CSV
uv run opinion-scraper export -f csv -o opinions.csv

# Export to JSON
uv run opinion-scraper export -f json -o opinions.json

# Export only negative opinions
uv run opinion-scraper export -f csv -o negative.csv -s negative

# Export positive opinions to JSON
uv run opinion-scraper export -f json -o positive.json -s positive
```

**CSV columns:** `platform`, `post_id`, `author`, `text`, `created_at`, `query`, `likes`, `reposts`, `sentiment_score`, `sentiment_label`

---

### `preset`

Scrape using pre-configured queries designed to capture AI tool opinions.

```
uv run opinion-scraper preset
```

No options. Runs `scrape` with these 4 queries (200 results each, all platforms):

| Query | What it finds |
|-------|---------------|
| `"AI tools" lang:en -is:retweet` | Broad mentions |
| `("AI tools" OR "AI assistants") ("I think" OR "my experience" OR "opinion")` | Personal opinions |
| `(ChatGPT OR Claude OR Gemini OR Copilot) (love OR hate OR amazing OR terrible)` | Specific tool sentiment |
| `"generative AI" (overrated OR underrated OR "game changer")` | Strong opinions |

---

## Setup

### Bluesky

1. Create an App Password at [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords)
2. Create a `.env` file in the project root:

```
BSKY_HANDLE=yourname.bsky.social
BSKY_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

### Twitter/X

Twitter requires accounts added to twscrape's pool:

```bash
uv run python -c "
import asyncio
from opinion_scraper.scraper.twitter import TwitterScraper

async def setup():
    scraper = TwitterScraper()
    await scraper.add_account('username', 'password', 'email', 'email_password')

asyncio.run(setup())
"
```

---

## Typical Workflow

```bash
# 1. Scrape
uv run opinion-scraper scrape -q "AI tools" -n 50 -p bluesky

# 2. Analyze
uv run opinion-scraper analyze

# 3. View results
uv run opinion-scraper report

# 4. Export
uv run opinion-scraper export -f csv -o results.csv

# 5. Reset (delete database and start fresh)
rm opinions.db
```

---

## Database

All data is stored in a local SQLite file (`opinions.db` by default). Duplicate posts are automatically ignored based on post ID.

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `platform` | TEXT | `twitter` or `bluesky` |
| `post_id` | TEXT (PK) | Unique post identifier |
| `author` | TEXT | Username/handle |
| `text` | TEXT | Post content |
| `created_at` | TEXT | ISO 8601 timestamp |
| `query` | TEXT | Search query that found this post |
| `likes` | INTEGER | Like/favorite count |
| `reposts` | INTEGER | Repost/retweet count |
| `sentiment_score` | REAL | VADER compound score (-1.0 to 1.0) |
| `sentiment_label` | TEXT | `positive`, `negative`, or `neutral` |
