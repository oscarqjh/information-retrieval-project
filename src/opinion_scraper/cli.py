"""CLI entry point for the opinion scraper."""

import asyncio

import click

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
@click.pass_context
def scrape(ctx, query, max_results, platform):
    """Scrape opinions from social media platforms."""
    store = OpinionStore(ctx.obj["db"])
    config = ScraperConfig(search_queries=list(query), max_results=max_results)

    async def run():
        total = 0
        if platform in ("twitter", "all"):
            click.echo("Scraping Twitter/X...")
            scraper = TwitterScraper()
            for q in config.search_queries:
                opinions = await scraper.scrape(q, config.max_results)
                store.save_batch(opinions)
                total += len(opinions)
                click.echo(f"  [{q}] collected {len(opinions)} tweets")

        if platform in ("bluesky", "all"):
            handle = click.prompt("Bluesky handle")
            password = click.prompt("Bluesky password", hide_input=True)
            scraper = BlueskyScraper(handle=handle, password=password)
            for q in config.search_queries:
                opinions = await scraper.scrape(q, config.max_results)
                store.save_batch(opinions)
                total += len(opinions)
                click.echo(f"  [{q}] collected {len(opinions)} posts")

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

    click.echo(f"Analyzing {len(unanalyzed)} opinions...")
    for opinion in unanalyzed:
        analyzer.analyze(opinion)
        store.update_sentiment(opinion.post_id, opinion.sentiment_score, opinion.sentiment_label)

    click.echo("Done. Run 'opinion-scraper report' to see results.")


@main.command()
@click.pass_context
def report(ctx):
    """Display sentiment analysis report."""
    store = OpinionStore(ctx.obj["db"])
    analyzer = SentimentAnalyzer()
    all_opinions = store.get_all()

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
@click.pass_context
def export(ctx, format, output, sentiment):
    """Export opinions to CSV or JSON."""
    from opinion_scraper.export import OpinionExporter

    store = OpinionStore(ctx.obj["db"])
    all_opinions = store.get_all()

    if not all_opinions:
        click.echo("No data. Run 'opinion-scraper scrape' first.")
        return

    if sentiment != "all":
        all_opinions = [o for o in all_opinions if o.sentiment_label == sentiment]

    exporter = OpinionExporter()
    if format == "csv":
        exporter.to_csv(all_opinions, output)
    else:
        exporter.to_json(all_opinions, output)

    click.echo(f"Exported {len(all_opinions)} opinions to {output}")


@main.command()
@click.pass_context
def preset(ctx):
    """Scrape using the AI opinions preset queries."""
    config = ScraperConfig.ai_opinions_preset()
    click.echo(f"Using {len(config.search_queries)} preset queries:")
    for q in config.search_queries:
        click.echo(f"  - {q}")
    ctx.invoke(scrape, query=config.search_queries, max_results=config.max_results, platform="all")
