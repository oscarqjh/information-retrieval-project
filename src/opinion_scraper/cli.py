"""CLI entry point for the opinion scraper."""

import asyncio
import os

import click
from dotenv import load_dotenv

load_dotenv()

from opinion_scraper.analysis import SentimentAnalyzer
from opinion_scraper.config import ScraperConfig
from opinion_scraper.scraper.bluesky import BlueskyScraper
from opinion_scraper.scraper.twitter import TwitterScraper
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
@click.option("--platform", "-p", type=click.Choice(["twitter", "bluesky", "all"]), default="all")
@click.option("--with-replies", is_flag=True, default=False, help="Also scrape reply threads.")
@click.option("--min-replies", default=0, help="Min reply count to fetch thread (0 = all).")
@click.option("--reply-depth", default=6, help="Max reply depth for Bluesky threads.")
@click.pass_context
def scrape(ctx, query, max_results, platform, with_replies, min_replies, reply_depth):
    """Scrape opinions from social media platforms."""
    store = OpinionStore(ctx.obj["db"])
    config = ScraperConfig(search_queries=list(query), max_results=max_results)

    async def run():
        from opinion_scraper.filter import RuleFilter

        total = 0
        rule_filter = RuleFilter()
        if platform in ("twitter", "all"):
            click.echo("Scraping Twitter/X...")
            scraper = TwitterScraper()
            for q in config.search_queries:
                with click.progressbar(length=config.max_results, label=f"  [{q}]") as bar:
                    opinions = await scraper.scrape(
                        q, config.max_results, on_progress=bar.update,
                        rule_filter=rule_filter,
                    )
                store.save_batch(opinions)
                total += len(opinions)
                if with_replies:
                    reply_total = 0
                    with click.progressbar(opinions, label=f"  Replies [{q}]") as bar:
                        for op in bar:
                            try:
                                tweet_id = int(op.post_id)
                                replies = await scraper.scrape_replies(
                                    tweet_id=tweet_id, query=q,
                                    rule_filter=rule_filter,
                                )
                                store.save_batch(replies)
                                reply_total += len(replies)
                            except Exception:
                                pass  # Skip failed thread fetches
                            await scraper._random_delay()
                    click.echo(f"  Collected {reply_total} replies")
                    total += reply_total

        if platform in ("bluesky", "all"):
            handle = os.environ.get("BSKY_HANDLE") or click.prompt("Bluesky handle")
            password = os.environ.get("BSKY_PASSWORD") or click.prompt("Bluesky password", hide_input=True)
            click.echo("Scraping Bluesky...")
            scraper = BlueskyScraper(handle=handle, password=password)
            try:
                scraper._ensure_login()
            except Exception as e:
                click.echo(f"  Failed to connect to Bluesky: {e}", err=True)
                scraper = None
            if scraper is not None:
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
@click.pass_context
def analyze(ctx):
    """Run sentiment analysis on unanalyzed opinions."""
    store = OpinionStore(ctx.obj["db"])
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


@main.command(name="filter")
@click.option("--threshold", default=0.5, help="Minimum confidence threshold.")
@click.option("--model", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", help="HuggingFace model name.")
@click.option("--batch-size", default=64, help="Inference batch size.")
@click.pass_context
def filter_cmd(ctx, threshold, model, batch_size):
    """Run ML relevance classification on unfiltered opinions."""
    from opinion_scraper.relevance import RelevanceClassifier

    store = OpinionStore(ctx.obj["db"])
    unfiltered = store.get_unfiltered()

    if not unfiltered:
        click.echo("No unfiltered opinions found.")
        return

    click.echo(f"Loading model: {model}")
    classifier = RelevanceClassifier(model_name=model, batch_size=batch_size)

    texts = [o.text for o in unfiltered]
    click.echo(f"Classifying {len(texts)} opinions...")

    with click.progressbar(range(0, len(texts), batch_size), label="Filtering") as bar:
        for i in bar:
            batch_texts = texts[i:i + batch_size]
            batch_opinions = unfiltered[i:i + batch_size]
            results = classifier.classify_batch(batch_texts)
            for opinion, (score, label) in zip(batch_opinions, results):
                if score >= threshold:
                    store.update_relevance(opinion.post_id, score, label)

    # Summary
    all_opinions = store.get_all()
    relevant = sum(1 for o in all_opinions if o.relevance_label == "relevant")
    spam = sum(1 for o in all_opinions if o.relevance_label == "spam")
    off_topic = sum(1 for o in all_opinions if o.relevance_label == "off_topic")
    click.echo(f"\nResults: {relevant} relevant, {spam} spam, {off_topic} off-topic")
    click.echo("Done. Run 'opinion-scraper report' to see results.")


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
    counts = store.count_by_platform()

    click.echo("\n=== Opinion Scraper Report ===\n")
    click.echo(f"Total opinions: {summary['total']}")
    for platform, count in counts.items():
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
@click.option("--format", "-f", type=click.Choice(["csv", "json"]), required=True, help="Export format.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output file path.")
@click.option("--sentiment", "-s", type=click.Choice(["all", "positive", "negative", "neutral"]), default="all", help="Filter by sentiment.")
@click.option("--platform", "-p", type=click.Choice(["all", "twitter", "bluesky"]), default="all", help="Filter by platform.")
@click.option("--relevant-only", is_flag=True, default=False, help="Exclude spam/off-topic posts.")
@click.pass_context
def export(ctx, format, output, sentiment, platform, relevant_only):
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

    exporter = OpinionExporter()
    if format == "csv":
        exporter.to_csv(all_opinions, output)
    else:
        exporter.to_json(all_opinions, output)

    click.echo(f"Exported {len(all_opinions)} opinions to {output}")


@main.command()
@click.option("--platform", "-p", type=click.Choice(["twitter", "bluesky", "all"]), default="all")
@click.option("--with-replies", is_flag=True, default=False, help="Also scrape reply threads.")
@click.option("--min-replies", default=0, help="Min reply count to fetch thread (0 = all).")
@click.option("--reply-depth", default=6, help="Max reply depth for Bluesky threads.")
@click.pass_context
def preset(ctx, platform, with_replies, min_replies, reply_depth):
    """Scrape using the AI opinions preset queries."""
    config = ScraperConfig.ai_opinions_preset()
    invoke_kwargs = dict(max_results=config.max_results, with_replies=with_replies,
                         min_replies=min_replies, reply_depth=reply_depth)

    if platform == "all":
        click.echo(f"Using {len(config.search_queries)} preset queries for Twitter:")
        for q in config.search_queries:
            click.echo(f"  - {q}")
        ctx.invoke(scrape, query=config.search_queries, platform="twitter", **invoke_kwargs)

        bsky_queries = config.bluesky_search_queries or config.search_queries
        click.echo(f"\nUsing {len(bsky_queries)} preset queries for Bluesky:")
        for q in bsky_queries:
            click.echo(f"  - {q}")
        ctx.invoke(scrape, query=bsky_queries, platform="bluesky", **invoke_kwargs)
    else:
        if platform == "bluesky" and config.bluesky_search_queries:
            queries = config.bluesky_search_queries
        else:
            queries = config.search_queries
        click.echo(f"Using {len(queries)} preset queries:")
        for q in queries:
            click.echo(f"  - {q}")
        ctx.invoke(scrape, query=queries, platform=platform, **invoke_kwargs)
