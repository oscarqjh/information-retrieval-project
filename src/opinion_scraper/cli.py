"""CLI entry point for the opinion scraper."""

import asyncio
import os

import click
from dotenv import load_dotenv

load_dotenv()

from opinion_scraper.analysis import SentimentAnalyzer
from opinion_scraper.config import ScraperConfig
from opinion_scraper.scraper.bluesky import BlueskyScraper
from opinion_scraper.storage import OpinionStore


@click.group()
@click.option("--db", default="opinions.db", help="Path to SQLite database.")
@click.pass_context
def main(ctx, db):
    """Scrape social media for opinions on AI tools."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


@main.command()
@click.option("--query", "-q", multiple=True, default=["AI tools"], help="Search query.")
@click.option("--max-results", "-n", default=100, help="Max results per query.")
@click.option("--with-replies", is_flag=True, default=False, help="Also scrape reply threads.")
@click.option("--min-replies", default=0, help="Min reply count to fetch thread (0 = all).")
@click.option("--reply-depth", default=6, help="Max reply depth for Bluesky threads.")
@click.pass_context
def scrape(ctx, query, max_results, with_replies, min_replies, reply_depth):
    """Scrape opinions from Bluesky."""
    store = OpinionStore(ctx.obj["db"])
    config = ScraperConfig(search_queries=list(query), max_results=max_results)

    async def run():
        from opinion_scraper.filter import RuleFilter

        total = 0
        rule_filter = RuleFilter()
        handle = os.environ.get("BSKY_HANDLE") or click.prompt("Bluesky handle")
        password = os.environ.get("BSKY_PASSWORD") or click.prompt("Bluesky password", hide_input=True)
        click.echo("Scraping Bluesky...")
        scraper = BlueskyScraper(handle=handle, password=password)
        try:
            scraper._ensure_login()
        except Exception as e:
            click.echo(f"  Failed to connect to Bluesky: {e}", err=True)
            return
        for q in config.search_queries:
            with click.progressbar(length=config.max_results, label=f"  [{q}]") as bar:
                opinions = await scraper.scrape(
                    q, config.max_results, on_progress=bar.update,
                    rule_filter=rule_filter,
                )
            store.save_batch(opinions)
            total += len(opinions)
            if with_replies:
                reply_candidates = [op for op in opinions
                                    if not (min_replies > 0 and getattr(op, '_reply_count', 0) < min_replies)]
                reply_total = 0
                with click.progressbar(reply_candidates, label=f"  Replies [{q}]") as bar:
                    for op in bar:
                        try:
                            replies = await scraper.scrape_replies(
                                post_uri=op._original_uri,
                                parent_post_id=op.post_id,
                                query=q,
                                depth=reply_depth,
                                rule_filter=rule_filter,
                            )
                            store.save_batch(replies)
                            reply_total += len(replies)
                        except Exception:
                            pass  # Skip failed thread fetches
                        await scraper._random_delay()
                click.echo(f"  Collected {reply_total} replies")
                total += reply_total

        click.echo(f"\nTotal: {total} opinions saved to {ctx.obj['db']}")

    asyncio.run(run())


@main.command()
@click.option("--force", is_flag=True, default=False, help="Re-analyze all opinions.")
@click.pass_context
def analyze(ctx, force):
    """Run sentiment analysis on unanalyzed opinions."""
    store = OpinionStore(ctx.obj["db"])
    if force:
        store.reset_sentiment()
    analyzer = SentimentAnalyzer()

    unanalyzed = store.get_unanalyzed()
    if not unanalyzed:
        click.echo("No unanalyzed opinions found.")
        return

    with click.progressbar(unanalyzed, label="Analyzing opinions") as bar:
        for opinion in bar:
            analyzer.analyze(opinion)
            store.update_sentiment(opinion.post_id, opinion.sentiment_score, opinion.sentiment_label)

    click.echo("Done. Run 'opinion-scraper report' to see results.")


@main.command()
@click.option("--force", is_flag=True, default=False, help="Re-clean all opinions.")
@click.pass_context
def clean(ctx, force):
    """Clean and preprocess opinion text."""
    from opinion_scraper.cleaner import TextCleaner

    store = OpinionStore(ctx.obj["db"])
    if force:
        store.reset_cleaned()
    cleaner = TextCleaner()

    uncleaned = store.get_uncleaned()
    if not uncleaned:
        click.echo("No uncleaned opinions found.")
        return

    # First pass: detect duplicate texts (spam bots posting identical content)
    from hashlib import md5
    text_hashes: dict[str, str] = {}  # hash -> first post_id
    duplicate_ids: set[str] = set()
    for opinion in uncleaned:
        h = md5(opinion.text.strip().lower().encode()).hexdigest()
        if h in text_hashes:
            duplicate_ids.add(opinion.post_id)
        else:
            text_hashes[h] = opinion.post_id

    stats = {"cleaned": 0, "too_short": 0, "bot": 0, "duplicate": 0}
    with click.progressbar(uncleaned, label="Cleaning opinions") as bar:
        for opinion in bar:
            if opinion.post_id in duplicate_ids:
                store.update_cleaned(opinion.post_id, None, "duplicate")
                stats["duplicate"] += 1
                continue
            cleaned_text, status = cleaner.clean(opinion.text, opinion.author)
            store.update_cleaned(opinion.post_id, cleaned_text, status)
            stats[status] += 1

    click.echo(f"\nResults: {stats['cleaned']} cleaned, "
               f"{stats['too_short']} too short, {stats['bot']} bot, "
               f"{stats['duplicate']} duplicate")


@main.command(name="filter")
@click.option("--threshold", default=0.5, help="Minimum confidence threshold.")
@click.option("--model", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", help="HuggingFace model name.")
@click.option("--batch-size", default=64, help="Inference batch size.")
@click.option("--force", is_flag=True, default=False, help="Re-classify all opinions.")
@click.pass_context
def filter_cmd(ctx, threshold, model, batch_size, force):
    """Run ML relevance classification on unfiltered opinions."""
    from opinion_scraper.relevance import RelevanceClassifier

    store = OpinionStore(ctx.obj["db"])
    if force:
        store.reset_relevance()
    unfiltered = store.get_unfiltered()

    if not unfiltered:
        click.echo("No unfiltered opinions found.")
        return

    click.echo(f"Loading model: {model}")
    classifier = RelevanceClassifier(model_name=model, batch_size=batch_size)

    texts = [o.cleaned_text or o.text for o in unfiltered]
    click.echo(f"Classifying {len(texts)} opinions...")

    with click.progressbar(range(0, len(texts), batch_size), label="Filtering") as bar:
        for i in bar:
            batch_texts = texts[i:i + batch_size]
            batch_opinions = unfiltered[i:i + batch_size]
            results = classifier.classify_batch(batch_texts)
            for opinion, (score, label) in zip(batch_opinions, results):
                store.update_relevance(opinion.post_id, score, label)

    # Summary
    all_opinions = store.get_all()
    relevant = sum(1 for o in all_opinions if o.relevance_label == "relevant")
    spam = sum(1 for o in all_opinions if o.relevance_label == "spam")
    off_topic = sum(1 for o in all_opinions if o.relevance_label == "off_topic")
    click.echo(f"\nResults: {relevant} relevant, {spam} spam, {off_topic} off-topic")
    click.echo("Done. Run 'opinion-scraper report' to see results.")


@main.command()
@click.option("--model-type", "-m", type=click.Choice(["finetuned", "pretrained"]), default="finetuned", help="Use finetuned or pretrained model.")
@click.option("--batch-size", default=32, help="Inference batch size.")
@click.option("--force", is_flag=True, default=False, help="Re-classify all opinions.")
@click.option("--platform", "-p", type=click.Choice(["all", "bluesky", "reddit"]), default="all", help="Filter by platform.")
@click.pass_context
def classify(ctx, model_type, batch_size, force, platform):
    """Classify opinions for subjectivity and polarity.

    Pipeline: subjectivity first, then polarity only for opinionated posts.
    Neutral subjectivity → polarity automatically set to neutral.
    Opinionated → polarity classified as positive or negative (no neutral).
    """
    import torch
    from transformers import pipeline as hf_pipeline

    store = OpinionStore(ctx.obj["db"])
    device = 0 if torch.cuda.is_available() else -1

    subjectivity_model = {
        "finetuned": "models/subjectivity_detection/best_model",
        "pretrained": "facebook/bart-large-mnli",
    }[model_type]

    polarity_model = {
        "finetuned": "models/polarity_detection/best_model",
        "pretrained": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    }[model_type]

    if force:
        store.reset_classification("subjectivity")
        store.reset_classification("polarity")

    # ── Step 1: Subjectivity classification ──────────────────────────
    unclassified = store.get_unclassified("subjectivity")
    if platform != "all":
        unclassified = [o for o in unclassified if o.platform == platform]

    if unclassified:
        click.echo(f"[subjectivity] Loading model: {subjectivity_model}")
        click.echo(f"[subjectivity] Classifying {len(unclassified)} opinions...")

        if model_type == "pretrained":
            classifier = hf_pipeline("zero-shot-classification", model=subjectivity_model, device=device, batch_size=batch_size)
            labels = ["neutral", "opinionated"]
            with click.progressbar(range(0, len(unclassified), batch_size), label="  [subjectivity]") as bar:
                for i in bar:
                    batch = unclassified[i:i + batch_size]
                    texts = [o.cleaned_text or o.text for o in batch]
                    results = classifier(texts, candidate_labels=labels, batch_size=batch_size)
                    if isinstance(results, dict):
                        results = [results]
                    updates = [(o.post_id, r["labels"][0]) for o, r in zip(batch, results)]
                    store.update_classification_batch(updates, "subjectivity")
        else:
            classifier = hf_pipeline("text-classification", model=subjectivity_model, device=device, batch_size=batch_size, truncation=True, max_length=512)
            with click.progressbar(range(0, len(unclassified), batch_size), label="  [subjectivity]") as bar:
                for i in bar:
                    batch = unclassified[i:i + batch_size]
                    texts = [o.cleaned_text or o.text for o in batch]
                    results = classifier(texts, batch_size=batch_size)
                    updates = [(o.post_id, r["label"].lower()) for o, r in zip(batch, results)]
                    store.update_classification_batch(updates, "subjectivity")
    else:
        click.echo("[subjectivity] No unclassified opinions found.")

    # ── Step 2: Set neutral polarity for neutral subjectivity ────────
    all_opinions = store.get_all()
    if platform != "all":
        all_opinions = [o for o in all_opinions if o.platform == platform]

    neutral_no_polarity = [o for o in all_opinions if o.subjectivity_label == "neutral" and o.polarity_label is None]
    if neutral_no_polarity:
        updates = [(o.post_id, "neutral") for o in neutral_no_polarity]
        store.update_classification_batch(updates, "polarity")
        click.echo(f"[polarity] Set {len(neutral_no_polarity)} neutral-subjectivity posts to neutral polarity.")

    # ── Step 3: Polarity classification for opinionated posts only ───
    opinionated_unclassified = [o for o in all_opinions if o.subjectivity_label == "opinionated" and o.polarity_label is None]

    if opinionated_unclassified:
        click.echo(f"[polarity] Loading model: {polarity_model}")
        click.echo(f"[polarity] Classifying {len(opinionated_unclassified)} opinionated opinions (positive/negative only)...")

        if model_type == "pretrained":
            classifier = hf_pipeline("text-classification", model=polarity_model, device=device, batch_size=batch_size, truncation=True, max_length=512, top_k=None)
        else:
            classifier = hf_pipeline("text-classification", model=polarity_model, device=device, batch_size=batch_size, truncation=True, max_length=512, top_k=None)

        with click.progressbar(range(0, len(opinionated_unclassified), batch_size), label="  [polarity]") as bar:
            for i in bar:
                batch = opinionated_unclassified[i:i + batch_size]
                texts = [o.cleaned_text or o.text for o in batch]
                results = classifier(texts, batch_size=batch_size)
                updates = []
                for o, scores in zip(batch, results):
                    # Pick highest scoring between positive and negative only
                    score_map = {r["label"].lower(): r["score"] for r in scores}
                    pos_score = score_map.get("positive", 0)
                    neg_score = score_map.get("negative", 0)
                    label = "positive" if pos_score >= neg_score else "negative"
                    updates.append((o.post_id, label))
                store.update_classification_batch(updates, "polarity")
    else:
        click.echo("[polarity] No opinionated opinions to classify.")

    # ── Summary ──────────────────────────────────────────────────────
    from collections import Counter
    all_opinions = store.get_all()
    if platform != "all":
        all_opinions = [o for o in all_opinions if o.platform == platform]
    sub_dist = Counter(o.subjectivity_label for o in all_opinions if o.subjectivity_label)
    pol_dist = Counter(o.polarity_label for o in all_opinions if o.polarity_label)
    click.echo(f"\n[subjectivity] Results: {dict(sub_dist)}")
    click.echo(f"[polarity] Results: {dict(pol_dist)}")


@main.command()
@click.option("--relevant-only", is_flag=True, default=False, help="Exclude spam/off-topic posts.")
@click.pass_context
def report(ctx, relevant_only):
    """Display sentiment analysis report."""
    store = OpinionStore(ctx.obj["db"])
    analyzer = SentimentAnalyzer()
    all_opinions = store.get_all()

    if relevant_only:
        all_opinions = [o for o in all_opinions if o.relevance_label in (None, "relevant")]

    if not all_opinions:
        click.echo("No data. Run 'opinion-scraper scrape' first.")
        return

    summary = analyzer.summarize(all_opinions)
    counts: dict[str, int] = {}
    for o in all_opinions:
        counts[o.platform] = counts.get(o.platform, 0) + 1

    click.echo("\n=== Opinion Scraper Report ===\n")
    click.echo(f"Total opinions: {summary['total']}")
    for platform, count in sorted(counts.items()):
        click.echo(f"  {platform}: {count}")
    click.echo(f"\nSentiment breakdown:")
    click.echo(f"  Positive: {summary['positive']} ({summary['positive']/max(summary['total'],1)*100:.1f}%)")
    click.echo(f"  Negative: {summary['negative']} ({summary['negative']/max(summary['total'],1)*100:.1f}%)")
    click.echo(f"  Neutral:  {summary['neutral']} ({summary['neutral']/max(summary['total'],1)*100:.1f}%)")
    click.echo(f"  Average score: {summary['avg_score']}")

    # Show sample opinions by sentiment
    for label in ("positive", "negative"):
        samples = [o for o in all_opinions if o.sentiment_label == label][:3]
        if samples:
            click.echo(f"\nSample {label} opinions:")
            for s in samples:
                click.echo(f"  [{s.platform}] @{s.author}: {s.text[:100]}...")


@main.command()
@click.option("--format", "-f", type=click.Choice(["csv", "json", "jsonl"]), required=True, help="Export format.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output file path.")
@click.option("--sentiment", "-s", type=click.Choice(["all", "positive", "negative", "neutral"]), default="all", help="Filter by sentiment.")
@click.option("--platform", "-p", type=click.Choice(["all", "bluesky", "reddit"]), default="all", help="Filter by platform.")
@click.option("--relevant-only", is_flag=True, default=False, help="Exclude spam/off-topic posts.")
@click.option("--clean-text-only", is_flag=True, default=False, help="Use cleaned text in the text column, drop cleaned_text/clean_status columns.")
@click.option("--exclude-rejected", is_flag=True, default=False, help="Exclude bot and too-short posts.")
@click.pass_context
def export(ctx, format, output, sentiment, platform, relevant_only, clean_text_only, exclude_rejected):
    """Export opinions to CSV or JSON."""
    from opinion_scraper.export import OpinionExporter

    store = OpinionStore(ctx.obj["db"])
    all_opinions = store.get_all()

    if not all_opinions:
        click.echo("No data. Run 'opinion-scraper scrape' first.")
        return

    if platform != "all":
        all_opinions = [o for o in all_opinions if o.platform == platform]

    if sentiment != "all":
        all_opinions = [o for o in all_opinions if o.sentiment_label == sentiment]

    if relevant_only:
        all_opinions = [o for o in all_opinions if o.relevance_label in (None, "relevant")]

    if exclude_rejected or clean_text_only:
        all_opinions = [o for o in all_opinions if o.clean_status not in ("bot", "too_short", "duplicate")]

    exporter = OpinionExporter()
    if format == "csv":
        exporter.to_csv(all_opinions, output, clean_text_only)
    elif format == "jsonl":
        exporter.to_jsonl(all_opinions, output, clean_text_only)
    else:
        exporter.to_json(all_opinions, output, clean_text_only)

    click.echo(f"Exported {len(all_opinions)} opinions to {output}")


@main.command()
@click.option("--with-replies", is_flag=True, default=False, help="Also scrape reply threads.")
@click.option("--min-replies", default=0, help="Min reply count to fetch thread (0 = all).")
@click.option("--reply-depth", default=6, help="Max reply depth for Bluesky threads.")
@click.pass_context
def preset(ctx, with_replies, min_replies, reply_depth):
    """Scrape using the AI opinions preset queries."""
    config = ScraperConfig.ai_opinions_preset()
    queries = config.search_queries
    click.echo(f"Using {len(queries)} preset queries:")
    for q in queries:
        click.echo(f"  - {q}")
    ctx.invoke(scrape, query=queries, max_results=config.max_results,
               with_replies=with_replies, min_replies=min_replies, reply_depth=reply_depth)
