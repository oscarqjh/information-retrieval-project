"""CLI entry point for the opinion scraper."""

import asyncio
import json
import os
from dataclasses import asdict

import click
from dotenv import load_dotenv

load_dotenv()

from opinion_scraper.analysis import SentimentAnalyzer
from opinion_scraper.classification.constants import DEFAULT_SARCASM_MODEL
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
@click.option("--task", "-t", type=click.Choice(["subjectivity", "polarity", "both"]), default="both", help="Which task to run.")
@click.option("--model-type", "-m", type=click.Choice(["finetuned", "pretrained"]), default="finetuned", help="Use finetuned or pretrained model.")
@click.option("--batch-size", default=32, help="Inference batch size.")
@click.option("--force", is_flag=True, default=False, help="Re-classify all opinions.")
@click.option("--platform", "-p", type=click.Choice(["all", "bluesky", "reddit"]), default="all", help="Filter by platform.")
@click.pass_context
def classify(ctx, task, model_type, batch_size, force, platform):
    """Classify opinions for subjectivity and/or polarity using ML models."""
    import torch
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

    store = OpinionStore(ctx.obj["db"])
    device = 0 if torch.cuda.is_available() else -1
    tasks = ["subjectivity", "polarity"] if task == "both" else [task]

    model_configs = {
        "subjectivity": {
            "finetuned": "models/subjectivity_detection/best_model",
            "pretrained": "facebook/bart-large-mnli",
            "labels": ["neutral", "opinionated"],
        },
        "polarity": {
            "finetuned": "models/polarity_detection/best_model",
            "pretrained": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "labels": ["positive", "neutral", "negative"],
        },
    }

    for t in tasks:
        config = model_configs[t]
        model_path = config[model_type]

        if force:
            store.reset_classification(t)

        unclassified = store.get_unclassified(t)
        if platform != "all":
            unclassified = [o for o in unclassified if o.platform == platform]

        if not unclassified:
            click.echo(f"[{t}] No unclassified opinions found.")
            continue

        click.echo(f"[{t}] Loading model: {model_path}")
        click.echo(f"[{t}] Classifying {len(unclassified)} opinions...")

        if model_type == "pretrained" and t == "subjectivity":
            # Zero-shot classification for subjectivity
            classifier = pipeline("zero-shot-classification", model=model_path, device=device, batch_size=batch_size)
            with click.progressbar(range(0, len(unclassified), batch_size), label=f"  [{t}]") as bar:
                for i in bar:
                    batch = unclassified[i:i + batch_size]
                    texts = [o.cleaned_text or o.text for o in batch]
                    results = classifier(texts, candidate_labels=config["labels"], batch_size=batch_size)
                    if isinstance(results, dict):
                        results = [results]
                    updates = [(o.post_id, r["labels"][0]) for o, r in zip(batch, results)]
                    store.update_classification_batch(updates, t)
        else:
            # Direct classification (finetuned models or pretrained sentiment)
            classifier = pipeline("text-classification", model=model_path, device=device, batch_size=batch_size, truncation=True, max_length=512)
            with click.progressbar(range(0, len(unclassified), batch_size), label=f"  [{t}]") as bar:
                for i in bar:
                    batch = unclassified[i:i + batch_size]
                    texts = [o.cleaned_text or o.text for o in batch]
                    results = classifier(texts, batch_size=batch_size)
                    updates = [(o.post_id, r["label"].lower()) for o, r in zip(batch, results)]
                    store.update_classification_batch(updates, t)

        # Summary
        all_opinions = store.get_all()
        if platform != "all":
            all_opinions = [o for o in all_opinions if o.platform == platform]
        col_val = [getattr(o, f"{t}_label") for o in all_opinions]
        from collections import Counter
        dist = Counter(v for v in col_val if v is not None)
        click.echo(f"\n[{t}] Results: {dict(dist)}")


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


@main.command(name="run-hierarchical-ablation")
@click.option("--input", "input_path", default="data/manual_label_dataset_v1.xlsx", help="Path to the manual-label XLSX file.")
@click.option("--output-dir", required=True, type=click.Path(), help="Directory for ablation outputs.")
@click.option("--base-model", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", help="Base NLI checkpoint used for all variants.")
@click.option("--sheet-name", default=None, help="Worksheet name to read from the XLSX file.")
@click.option("--text-column", default="text", help="Primary text column.")
@click.option("--fallback-text-column", default="cleaned_text", help="Fallback text column.")
@click.option("--validation-ratio", default=0.2, type=float, help="Validation split ratio.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--zero-shot-batch-size", default=8, type=int, help="Batch size for the no-fine-tuning ablation.")
@click.option("--num-train-epochs", default=6, type=int, help="Default number of training epochs for all fine-tuned variants.")
@click.option("--learning-rate", default=2e-5, type=float, help="Default learning rate for all fine-tuned variants.")
@click.option("--per-device-train-batch-size", default=4, type=int, help="Default training batch size for all fine-tuned variants.")
@click.option("--per-device-eval-batch-size", default=8, type=int, help="Default validation batch size for all fine-tuned variants.")
@click.option("--weight-decay", default=0.01, type=float, help="Default weight decay for all fine-tuned variants.")
@click.option("--warmup-ratio", default=0.1, type=float, help="Default warmup ratio for all fine-tuned variants.")
@click.option("--classifier-dropout", default=None, type=float, help="Optional classifier-head dropout override for all fine-tuned variants.")
@click.option("--early-stopping-patience", default=2, type=int, help="Default early-stopping patience for all fine-tuned variants.")
@click.option("--early-stopping-threshold", default=0.0, type=float, help="Default early-stopping improvement threshold.")
@click.option("--subjectivity-num-train-epochs", default=None, type=int, help="Stage-specific epoch override for subjectivity.")
@click.option("--subjectivity-learning-rate", default=None, type=float, help="Stage-specific learning-rate override for subjectivity.")
@click.option("--polarity-num-train-epochs", default=None, type=int, help="Stage-specific epoch override for polarity.")
@click.option("--polarity-learning-rate", default=None, type=float, help="Stage-specific learning-rate override for polarity.")
@click.option("--flat-num-train-epochs", default=None, type=int, help="Ablation-specific epoch override for the flat classifier.")
@click.option("--flat-learning-rate", default=None, type=float, help="Ablation-specific learning-rate override for the flat classifier.")
def run_hierarchical_ablation(
    input_path,
    output_dir,
    base_model,
    sheet_name,
    text_column,
    fallback_text_column,
    validation_ratio,
    seed,
    zero_shot_batch_size,
    num_train_epochs,
    learning_rate,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    weight_decay,
    warmup_ratio,
    classifier_dropout,
    early_stopping_patience,
    early_stopping_threshold,
    subjectivity_num_train_epochs,
    subjectivity_learning_rate,
    polarity_num_train_epochs,
    polarity_learning_rate,
    flat_num_train_epochs,
    flat_learning_rate,
):
    """Run baseline and ablation experiments on a shared validation split."""
    from opinion_scraper.classification import (
        AblationConfig,
        AblationRunner,
        ManualLabelDatasetBuilder,
        StageTrainingConfig,
    )

    builder = ManualLabelDatasetBuilder(
        text_column=text_column,
        fallback_text_column=fallback_text_column,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    base_stage = StageTrainingConfig(
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        classifier_dropout=classifier_dropout,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
    )
    runner = AblationRunner(dataset_builder=builder)
    artifacts = runner.run(
        output_dir=output_dir,
        data_path=input_path,
        sheet_name=sheet_name,
        config=AblationConfig(
            base_model=base_model,
            validation_ratio=validation_ratio,
            seed=seed,
            zero_shot_batch_size=zero_shot_batch_size,
            subjectivity=StageTrainingConfig(
                num_train_epochs=subjectivity_num_train_epochs or base_stage.num_train_epochs,
                learning_rate=subjectivity_learning_rate or base_stage.learning_rate,
                per_device_train_batch_size=base_stage.per_device_train_batch_size,
                per_device_eval_batch_size=base_stage.per_device_eval_batch_size,
                weight_decay=base_stage.weight_decay,
                warmup_ratio=base_stage.warmup_ratio,
                classifier_dropout=base_stage.classifier_dropout,
                early_stopping_patience=base_stage.early_stopping_patience,
                early_stopping_threshold=base_stage.early_stopping_threshold,
            ),
            polarity=StageTrainingConfig(
                num_train_epochs=polarity_num_train_epochs or base_stage.num_train_epochs,
                learning_rate=polarity_learning_rate or base_stage.learning_rate,
                per_device_train_batch_size=base_stage.per_device_train_batch_size,
                per_device_eval_batch_size=base_stage.per_device_eval_batch_size,
                weight_decay=base_stage.weight_decay,
                warmup_ratio=base_stage.warmup_ratio,
                classifier_dropout=base_stage.classifier_dropout,
                early_stopping_patience=base_stage.early_stopping_patience,
                early_stopping_threshold=base_stage.early_stopping_threshold,
            ),
            flat_final=StageTrainingConfig(
                num_train_epochs=flat_num_train_epochs or base_stage.num_train_epochs,
                learning_rate=flat_learning_rate or base_stage.learning_rate,
                per_device_train_batch_size=base_stage.per_device_train_batch_size,
                per_device_eval_batch_size=base_stage.per_device_eval_batch_size,
                weight_decay=base_stage.weight_decay,
                warmup_ratio=base_stage.warmup_ratio,
                classifier_dropout=base_stage.classifier_dropout,
                early_stopping_patience=base_stage.early_stopping_patience,
                early_stopping_threshold=base_stage.early_stopping_threshold,
            ),
        ),
    )
    click.echo(json.dumps(artifacts.summary, ensure_ascii=False, indent=2))


@main.command(name="classify-hierarchical")
@click.option("--text", "texts", multiple=True, required=True, help="Text to classify. Repeat the option for multiple inputs.")
@click.option("--subjectivity-model", default="artifacts/ablation/baseline_hierarchical_finetuned/subjectivity", show_default=True, type=click.Path(), help="Path to the stage-1 subjectivity model.")
@click.option("--polarity-model", default="artifacts/ablation/baseline_hierarchical_finetuned/polarity", show_default=True, type=click.Path(), help="Path to the stage-2 polarity model.")
@click.option("--device", default=-1, type=int, help="Transformers device ID. Use -1 for CPU.")
@click.option("--batch-size", default=8, type=int, help="Inference batch size.")
@click.option("--local-files-only", is_flag=True, default=False, help="Only load model files from local storage.")
def classify_hierarchical(texts, subjectivity_model, polarity_model, device, batch_size, local_files_only):
    """Run hierarchical subjectivity and polarity classification."""
    from opinion_scraper.classification import HierarchicalClassifier

    classifier = HierarchicalClassifier(
        subjectivity_model_path=subjectivity_model,
        polarity_model_path=polarity_model,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    for prediction in classifier.predict(list(texts)):
        click.echo(json.dumps(prediction.to_dict(), ensure_ascii=False))


@main.command(name="annotate-hierarchical")
@click.option("--csv-path", default="data/all_opinions.csv", type=click.Path(), help="CSV file to update in place.")
@click.option("--subjectivity-model", default="artifacts/ablation/baseline_hierarchical_finetuned/subjectivity", show_default=True, type=click.Path(), help="Path to the stage-1 subjectivity model.")
@click.option("--polarity-model", default="artifacts/ablation/baseline_hierarchical_finetuned/polarity", show_default=True, type=click.Path(), help="Path to the stage-2 polarity model.")
@click.option("--device", default=0, type=int, help="Transformers device ID. Use -1 for CPU.")
@click.option("--batch-size", default=32, type=int, help="Annotation batch size.")
@click.option("--local-files-only", is_flag=True, default=False, help="Only load model files from local storage.")
@click.option("--force", is_flag=True, default=False, help="Recompute labels even when the target columns are already populated.")
def annotate_hierarchical(
    csv_path,
    subjectivity_model,
    polarity_model,
    device,
    batch_size,
    local_files_only,
    force,
):
    """Update a CSV file in place with hierarchical subjectivity/polarity labels."""
    from opinion_scraper.classification import (
        format_annotation_score,
        HierarchicalBatchAnnotator,
        HierarchicalClassifier,
        load_csv_records,
        prepare_csv_records_for_annotation,
        update_csv_records_in_place,
    )

    rows, _ = load_csv_records(csv_path)
    records = prepare_csv_records_for_annotation(
        rows=rows,
        target_columns=["subjectivity_label", "subjectivity_score", "polarity_label", "polarity_score"],
        force=force,
    )
    if not records:
        click.echo(
            "No rows require hierarchical updates. Use --force to recompute labels and refresh throughput metrics."
        )
        return
    classifier = HierarchicalClassifier(
        subjectivity_model_path=subjectivity_model,
        polarity_model_path=polarity_model,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    annotator = HierarchicalBatchAnnotator(classifier=classifier)
    artifacts = annotator.annotate_records(records=records, batch_size=batch_size)
    updates_by_id = {
        str(row["post_id"]): {
            "subjectivity_label": row["stage1_label"],
            "subjectivity_score": format_annotation_score(row["stage1_score"]),
            "polarity_label": row["final_label"] if row["stage2_label"] is not None else "",
            "polarity_score": format_annotation_score(row["stage2_score"]),
        }
        for row in artifacts.predictions
    }
    update_csv_records_in_place(
        path=csv_path,
        updates_by_id=updates_by_id,
        new_columns=["subjectivity_label", "subjectivity_score", "polarity_label", "polarity_score"],
    )
    annotator.save_stats(artifacts.stats, f"{csv_path}.hierarchical.metrics.json")
    click.echo(f"Annotated {artifacts.stats.total_records} rows in {csv_path}")
    click.echo(json.dumps(asdict(artifacts.stats), ensure_ascii=False))


@main.command(name="evaluate-sarcasm-classifier")
@click.option("--input", "input_path", default="data/manual_label_dataset_v1.xlsx", help="Path to the manual-label XLSX file.")
@click.option("--model", default=DEFAULT_SARCASM_MODEL, show_default=True, help="Zero-shot NLI model checkpoint.")
@click.option("--sheet-name", default=None, help="Worksheet name to read from the XLSX file.")
@click.option("--text-column", default="text", help="Primary text column.")
@click.option("--fallback-text-column", default="cleaned_text", help="Fallback text column.")
@click.option("--batch-size", default=16, type=int, help="Inference batch size.")
@click.option("--device", default=0, type=int, help="Transformers device ID. Use -1 for CPU.")
@click.option("--local-files-only", is_flag=True, default=False, help="Only load model files from local storage.")
@click.option("--threshold", default=0.5, type=float, help="Decision threshold for the sarcastic class.")
@click.option("--threshold-metric", default="f1", type=click.Choice(["accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1"]), help="Metric used when recommending a tuned threshold.")
@click.option("--threshold-search-steps", default=101, type=int, help="Number of threshold candidates to test.")
@click.option("--hypothesis-template", default="{}", show_default=True, help="Zero-shot hypothesis template.")
@click.option("--metrics-out", default=None, type=click.Path(), help="Optional JSON path for evaluation metrics.")
def evaluate_sarcasm_classifier(
    input_path,
    model,
    sheet_name,
    text_column,
    fallback_text_column,
    batch_size,
    device,
    local_files_only,
    threshold,
    threshold_metric,
    threshold_search_steps,
    hypothesis_template,
    metrics_out,
):
    """Evaluate zero-shot sarcasm agreement on the manually labeled dataset."""
    from opinion_scraper.classification import evaluate_sarcasm_on_manual_labels

    artifacts = evaluate_sarcasm_on_manual_labels(
        data_path=input_path,
        model_name=model,
        sheet_name=sheet_name,
        text_column=text_column,
        fallback_text_column=fallback_text_column,
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
        threshold=threshold,
        threshold_metric=threshold_metric,
        threshold_search_steps=threshold_search_steps,
        hypothesis_template=hypothesis_template,
        metrics_out=metrics_out,
    )
    click.echo(json.dumps(artifacts.metrics, ensure_ascii=False, indent=2))


@main.command(name="classify-sarcasm")
@click.option("--text", "texts", multiple=True, required=True, help="Text to classify. Repeat the option for multiple inputs.")
@click.option("--model", default=DEFAULT_SARCASM_MODEL, show_default=True, help="Zero-shot NLI model checkpoint.")
@click.option("--device", default=-1, type=int, help="Transformers device ID. Use -1 for CPU.")
@click.option("--batch-size", default=8, type=int, help="Inference batch size.")
@click.option("--local-files-only", is_flag=True, default=False, help="Only load model files from local storage.")
@click.option("--threshold", default=0.5, type=float, help="Decision threshold for the sarcastic class.")
@click.option("--hypothesis-template", default="{}", show_default=True, help="Zero-shot hypothesis template.")
def classify_sarcasm(texts, model, device, batch_size, local_files_only, threshold, hypothesis_template):
    """Run zero-shot sarcasm classification."""
    from opinion_scraper.classification import SarcasmClassifier

    classifier = SarcasmClassifier(
        model_name=model,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
        hypothesis_template=hypothesis_template,
    )
    for prediction in classifier.predict(list(texts), threshold=threshold):
        click.echo(json.dumps(prediction.to_dict(), ensure_ascii=False))


@main.command(name="annotate-sarcasm")
@click.option("--csv-path", default="data/all_opinions.csv", type=click.Path(), help="CSV file to update in place.")
@click.option("--model", default=DEFAULT_SARCASM_MODEL, show_default=True, help="Zero-shot NLI model checkpoint.")
@click.option("--device", default=0, type=int, help="Transformers device ID. Use -1 for CPU.")
@click.option("--batch-size", default=32, type=int, help="Annotation batch size.")
@click.option("--local-files-only", is_flag=True, default=False, help="Only load model files from local storage.")
@click.option("--threshold", default=0.5, type=float, help="Decision threshold for the sarcastic class.")
@click.option("--hypothesis-template", default="{}", show_default=True, help="Zero-shot hypothesis template.")
@click.option("--force", is_flag=True, default=False, help="Recompute labels even when sarcasm columns are already populated.")
def annotate_sarcasm(
    csv_path,
    model,
    device,
    batch_size,
    local_files_only,
    threshold,
    hypothesis_template,
    force,
):
    """Update a CSV file in place with zero-shot sarcasm labels and scores."""
    from opinion_scraper.classification import (
        format_annotation_score,
        SarcasmBatchAnnotator,
        SarcasmClassifier,
        load_csv_records,
        prepare_csv_records_for_annotation,
        update_csv_records_in_place,
    )

    rows, _ = load_csv_records(csv_path)
    records = prepare_csv_records_for_annotation(
        rows=rows,
        target_columns=["sarcasm_label", "sarcasm_score"],
        force=force,
    )
    if not records:
        click.echo(
            "No rows require sarcasm updates. Use --force to recompute labels and refresh throughput metrics."
        )
        return
    classifier = SarcasmClassifier(
        model_name=model,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
        hypothesis_template=hypothesis_template,
    )
    annotator = SarcasmBatchAnnotator(classifier=classifier)
    artifacts = annotator.annotate_records(
        records=records,
        batch_size=batch_size,
        threshold=threshold,
    )
    updates_by_id = {
        str(row["post_id"]): {
            "sarcasm_label": row["sarcasm_label"],
            "sarcasm_score": format_annotation_score(row["sarcasm_score"]),
        }
        for row in artifacts.predictions
    }
    update_csv_records_in_place(
        path=csv_path,
        updates_by_id=updates_by_id,
        new_columns=["sarcasm_label", "sarcasm_score"],
    )
    annotator.save_stats(artifacts.stats, f"{csv_path}.sarcasm.metrics.json")
    click.echo(f"Annotated {artifacts.stats.total_records} rows in {csv_path}")
    click.echo(json.dumps(asdict(artifacts.stats), ensure_ascii=False))
