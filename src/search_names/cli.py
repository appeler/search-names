"""Command-line interface for the search-names pipeline."""

from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress

from .logging_config import get_logger, setup_logging
from .merge_results import merge_results as merge_results_func
from .pipeline import augment_names, clean_names, preprocess_names, search_names
from .split_text_corpus import DEFAULT_OUTPUT_PATTERN
from .split_text_corpus import split_text_corpus as split_text_corpus_func

# Initialize console for rich output
console = Console()

# Create the main typer application
app = typer.Typer(
    name="search-names",
    help="Prepare personal names and search CSV text corpora",
    add_completion=False,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Show version information."""
    if value:
        from . import __version__

        rprint(f"search-names version [bold green]{__version__}[/bold green]")
        raise typer.Exit


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V", help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only show warning and error logs"
    ),
):
    """Prepare names and search text corpora."""
    # Set up logging based on verbosity
    if quiet:
        log_level = "WARNING"
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    setup_logging(level=log_level)


@app.command("clean")
def clean_names_cmd(
    input_file: Path = typer.Argument(
        ..., help="Input CSV file with names", exists=True
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output CSV file (default: clean_names.csv)"
    ),
    column: str = typer.Option(
        "Name", "--column", "-c", help="Column name containing names"
    ),
    keep_duplicates: bool = typer.Option(
        False, "--keep-duplicates", "-a", help="Keep duplicate names (don't remove)"
    ),
):
    """Clean and standardize names in a CSV file.

    This command parses names from various formats into standardized fields:
    FirstName, MiddleName, LastName, Prefix, Suffix, etc.
    """
    logger = get_logger("cli.clean")

    if not output_file:
        output_file = Path("clean_names.csv")

    try:
        with console.status("[bold green]Cleaning names..."):
            result = clean_names(input_file, output_file, column, keep_duplicates)

        if result:
            rprint(f"✓ Cleaned names saved to [bold green]{output_file}[/bold green]")
            rprint(f"  Processed [bold blue]{len(result)}[/bold blue] names")
        else:
            rprint("[bold red]✗ Failed to clean names[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Error cleaning names: %s", e)
        rprint(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("merge-supp")
def merge_supplementary_data_cmd(
    input_file: Path = typer.Argument(
        ..., help="Input CSV file (output from clean command)", exists=True
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV file (default: augmented_clean_names.csv)",
    ),
    name_column: str = typer.Option(
        "FirstName", "--name-column", "-n", help="Column name for nickname lookup"
    ),
    prefix_column: str = typer.Option(
        "seat", "--prefix-column", "-p", help="Column name for prefix lookup"
    ),
    prefix_file: Path | None = typer.Option(
        None,
        "--prefix-file",
        help="Optional CSV file with prefix mappings",
    ),
    nickname_file: Path | None = typer.Option(
        None,
        "--nickname-file",
        help="Optional text file with nickname mappings",
    ),
):
    """Merge supplementary data (prefixes, nicknames) with cleaned names.

    Adds additional name variations and prefixes based on lookup files.
    """
    logger = get_logger("cli.merge-supp")

    if not output_file:
        output_file = Path("augmented_clean_names.csv")

    try:
        with console.status("[bold green]Merging supplementary data..."):
            augment_names(
                input_file,
                prefix_column,
                name_column,
                output_file,
                prefix_file,
                nickname_file,
            )

        rprint(f"✓ Augmented data saved to [bold green]{output_file}[/bold green]")

    except Exception as e:
        logger.error("Error merging supplementary data: %s", e)
        rprint(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("preprocess")
def preprocess_cmd(
    input_file: Path = typer.Argument(
        ..., help="Input CSV file (output from merge-supp)", exists=True
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV file (default: deduped_augmented_clean_names.csv)",
    ),
    patterns: list[str] | None = typer.Option(
        None, "--patterns", "-p", help="Search patterns (e.g., 'FirstName LastName')"
    ),
    drop_patterns_file: Path | None = typer.Option(
        None,
        "--drop-patterns",
        "-d",
        help="File with patterns to drop (default: drop_patterns.txt)",
    ),
    edit_lengths: list[int] | None = typer.Option(
        None, "--edit-lengths", "-e", help="Edit distance thresholds for fuzzy matching"
    ),
):
    """Preprocess search list: create patterns, deduplicate, filter.

    Converts wide format to long format with search patterns and removes duplicates.
    """
    logger = get_logger("cli.preprocess")

    if not output_file:
        output_file = Path("deduped_augmented_clean_names.csv")

    if not patterns:
        patterns = ["FirstName LastName", "NickName LastName", "Prefix LastName"]

    edit_lengths = edit_lengths or []

    try:
        with console.status("[bold green]Preprocessing search list..."):
            # Load drop patterns first
            from .pipeline.step3_preprocess import load_drop_patterns

            drop_patterns = load_drop_patterns(drop_patterns_file)

            preprocess_names(
                input_file, patterns, output_file, edit_lengths, drop_patterns
            )

        rprint(f"✓ Preprocessed data saved to [bold green]{output_file}[/bold green]")

    except Exception as e:
        logger.error("Error preprocessing: %s", e)
        rprint(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("split")
def split_corpus_cmd(
    input_file: Path = typer.Argument(
        ..., help="Input CSV file with text corpus", exists=True
    ),
    output_pattern: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output pattern (default: {basename}_{chunk_id:04d}.csv)",
    ),
    chunk_size: int = typer.Option(
        1000, "--size", "-s", help="Number of rows per chunk"
    ),
):
    """Split large text corpus into smaller chunks for parallel processing.

    Useful for processing very large datasets across multiple machines.
    """
    logger = get_logger("cli.split")

    try:
        with console.status(
            f"[bold green]Splitting corpus into chunks of {chunk_size}..."
        ):
            chunk_count = split_text_corpus_func(
                str(input_file), output_pattern or DEFAULT_OUTPUT_PATTERN, chunk_size
            )

        rprint(f"✓ Split corpus into [bold blue]{chunk_count}[/bold blue] chunks")

    except Exception as e:
        logger.error("Error splitting corpus: %s", e)
        rprint(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("search")
def search_cmd(
    input_file: Path = typer.Argument(
        ..., help="Input CSV file with text corpus", exists=True
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output CSV file (default: search_results.csv)"
    ),
    names_file: Path | None = typer.Option(
        None, "--names", "-n", help="CSV file with names to search for"
    ),
    max_results: int = typer.Option(
        20, "--max-results", "-m", help="Maximum search results per document"
    ),
    processes: int = typer.Option(
        4, "--processes", "-p", help="Number of parallel processes"
    ),
    text_column: str = typer.Option(
        "text", "--text-column", "-t", help="Column name containing text to search"
    ),
    fuzzy_rules: list[str] | None = typer.Option(
        None,
        "--fuzzy-rule",
        "-f",
        help="Fuzzy rule as MINIMUM_LENGTH:MAX_EDIT_DISTANCE; repeat as needed",
    ),
):
    """Search for exact or fuzzy name matches in a text corpus."""
    logger = get_logger("cli.search")

    if not output_file:
        output_file = Path("search_results.csv")

    if not names_file:
        names_file = Path("deduped_augmented_clean_names.csv")

    parsed_fuzzy_rules = []
    for rule in fuzzy_rules or []:
        try:
            minimum_length, edit_distance = rule.split(":", maxsplit=1)
            parsed_fuzzy_rules.append((int(minimum_length), int(edit_distance)))
        except ValueError as error:
            raise typer.BadParameter(
                f"invalid fuzzy rule {rule!r}; "
                "expected MINIMUM_LENGTH:MAX_EDIT_DISTANCE"
            ) from error

    try:
        with Progress() as progress:
            progress.add_task("[bold green]Searching names...", total=None)

            # Load names first
            from .pipeline.step4_search import load_names_file

            names = load_names_file(str(names_file))

            result = search_names(
                input_file,
                names,
                output_file,
                text_column=text_column,
                max_results=max_results,
                fuzzy_rules=parsed_fuzzy_rules,
                processes=processes,
            )

        rprint(f"✓ Search results saved to [bold green]{output_file}[/bold green]")
        rprint(
            f"  Searched [bold blue]{result['total_rows']}[/bold blue] rows; "
            f"found [bold blue]{result['total_matches']}[/bold blue] matches"
        )

    except Exception as e:
        logger.error("Error during search: %s", e)
        rprint(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("merge-results")
def merge_results_cmd(
    input_files: list[Path] = typer.Argument(..., help="Input CSV files to merge"),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV file (default: merged_search_results.csv)",
    ),
):
    """Merge search results from multiple files into a single file.

    Useful after processing corpus chunks separately.
    """
    logger = get_logger("cli.merge-results")

    if not output_file:
        output_file = Path("merged_search_results.csv")

    # Validate input files exist
    for file_path in input_files:
        if not file_path.exists():
            rprint(f"[bold red]✗ Input file does not exist: {file_path}[/bold red]")
            raise typer.Exit(1)

    try:
        with console.status(f"[bold green]Merging {len(input_files)} result files..."):
            merge_results_func([str(f) for f in input_files], str(output_file))

        rprint(f"✓ Merged results saved to [bold green]{output_file}[/bold green]")

    except Exception as e:
        logger.error("Error merging results: %s", e)
        rprint(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("pipeline")
def pipeline_cmd(
    input_file: Path = typer.Argument(
        ..., help="Input CSV file with raw names", exists=True
    ),
    corpus_file: Path = typer.Argument(..., help="Text corpus CSV file", exists=True),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-d", help="Output directory (default: ./pipeline_output)"
    ),
    skip_clean: bool = typer.Option(
        False, "--skip-clean", help="Skip name cleaning step"
    ),
    skip_merge: bool = typer.Option(
        False, "--skip-merge", help="Skip supplementary data merge step"
    ),
    skip_preprocess: bool = typer.Option(
        False, "--skip-preprocess", help="Skip preprocessing step"
    ),
):
    """Run the complete pipeline: clean → merge-supp → preprocess → search.

    Convenience command that runs all steps in sequence.
    """
    logger = get_logger("cli.pipeline")

    if not output_dir:
        output_dir = Path("./pipeline_output")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Clean names
        if not skip_clean:
            clean_output = output_dir / "01_clean_names.csv"
            rprint("[bold blue]Step 1:[/bold blue] Cleaning names...")
            clean_names(str(input_file), str(clean_output), "Name", False)
            rprint(f"✓ Names cleaned: {clean_output}")
        else:
            clean_output = input_file

        # Step 2: Merge supplementary data
        if not skip_merge:
            merge_output = output_dir / "02_augmented_names.csv"
            rprint("[bold blue]Step 2:[/bold blue] Merging supplementary data...")
            augment_names(str(clean_output), "seat", "FirstName", str(merge_output))
            rprint(f"✓ Supplementary data merged: {merge_output}")
        else:
            merge_output = clean_output

        # Step 3: Preprocess
        if not skip_preprocess:
            preprocess_output = output_dir / "03_preprocessed_names.csv"
            rprint("[bold blue]Step 3:[/bold blue] Preprocessing...")
            # Load drop patterns for preprocessing
            from .pipeline.step3_preprocess import load_drop_patterns

            drop_patterns = load_drop_patterns(None)
            preprocess_names(
                str(merge_output),
                ["FirstName LastName", "NickName LastName", "Prefix LastName"],
                str(preprocess_output),
                [],
                drop_patterns,
            )
            rprint(f"✓ Data preprocessed: {preprocess_output}")
        else:
            preprocess_output = merge_output

        # Step 4: Search
        search_output = output_dir / "04_search_results.csv"
        rprint("[bold blue]Step 4:[/bold blue] Searching...")
        # Load names for search
        from .pipeline.step4_search import load_names_file

        names = load_names_file(str(preprocess_output))
        search_names(
            corpus_file,
            names,
            search_output,
        )
        rprint(f"✓ Search completed: {search_output}")

        rprint("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        rprint(f"Results available in: [bold green]{output_dir}[/bold green]")

    except Exception as e:
        logger.error("Pipeline error: %s", e)
        rprint(f"[bold red]✗ Pipeline failed: {e}[/bold red]")
        raise typer.Exit(1) from e


def main_cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main_cli()
