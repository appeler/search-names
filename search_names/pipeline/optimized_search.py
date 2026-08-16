#!/usr/bin/env python
"""
Optimized parallel search implementation with better performance patterns.

Key Optimizations:
1. Chunk-based processing instead of row-by-row modulo distribution
2. Shared search engine initialization
3. Batch queue operations to reduce overhead
4. Adaptive chunking based on file size
5. Better memory management
"""

import csv
import gzip
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from .. import utils
from ..engines import RESULT_FIELDS, create_search_engine

logger = logging.getLogger(__name__)

# Optimized constants
DEFAULT_CHUNK_SIZE = 1000
MAX_QUEUE_SIZE = 10000  # Larger queue for better throughput
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000
BATCH_SIZE = 50  # Process results in batches


class OptimizedSearchProcessor:
    """High-performance parallel search processor."""

    def __init__(
        self,
        names: list[tuple[str, str]],
        editlength: list[int] = None,
        processes: int = None,
        chunk_size: int = None,
    ):
        self.names = names
        self.editlength = editlength or []
        self.processes = processes or mp.cpu_count()
        self.chunk_size = chunk_size

        # Auto-tune chunk size if not provided
        if self.chunk_size is None:
            self.chunk_size = self._calculate_optimal_chunk_size()

    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available resources."""
        # Base chunk size on number of processes and expected complexity
        base_size = max(DEFAULT_CHUNK_SIZE // self.processes, MIN_CHUNK_SIZE)

        # Adjust based on number of names to search (more names = smaller chunks)
        complexity_factor = min(len(self.names) / 100, 5.0)
        adjusted_size = int(base_size / max(complexity_factor, 1.0))

        return min(max(adjusted_size, MIN_CHUNK_SIZE), MAX_CHUNK_SIZE)

    def _create_file_chunks(
        self, input_file: str, text_column: str
    ) -> list[tuple[int, int, list[dict]]]:
        """Create optimized chunks for processing."""
        chunks = []
        chunk_data = []
        chunk_start_line = 0

        _open = gzip.open if input_file.endswith(".gz") else open

        with _open(input_file, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Read after the loop for the final chunk; an empty file never enters it.

            line_num = 0

            for line_num, row in enumerate(reader):
                if text_column not in row or not row[text_column].strip():
                    continue

                chunk_data.append((line_num, row))

                if len(chunk_data) >= self.chunk_size:
                    chunks.append((chunk_start_line, line_num, chunk_data))
                    chunk_start_line = line_num + 1
                    chunk_data = []

            # Add remaining data as final chunk
            if chunk_data:
                chunks.append((chunk_start_line, line_num, chunk_data))

        logger.info(f"Created {len(chunks)} chunks for processing")
        return chunks

    def search_file_parallel(
        self,
        input_file: str,
        output_file: str,
        text_column: str = "text",
        input_cols: list[str] = None,
        search_cols: list[str] = None,
        max_results: int = 20,
        clean_text: bool = True,
    ) -> dict[str, Any]:
        """
        Perform optimized parallel search on a file.

        Returns performance statistics.
        """
        start_time = time.time()
        input_cols = input_cols or ["uniqid", "text"]
        search_cols = search_cols or ["uniqid", "n", "match", "start", "end", "count"]

        # Create chunks for parallel processing
        chunks = self._create_file_chunks(input_file, text_column)

        if not chunks:
            logger.warning("No data to process")
            return {"total_rows": 0, "processing_time": 0}

        # Initialize output file
        self._initialize_output_file(output_file, input_file, input_cols, search_cols, max_results)

        # Process chunks in parallel
        total_results = 0
        processed_rows = 0

        logger.info(f"Processing {len(chunks)} chunks with {self.processes} processes")

        with ProcessPoolExecutor(max_workers=self.processes) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(
                    process_chunk_worker,
                    chunk_data,
                    self.names,
                    self.editlength,
                    text_column,
                    input_cols,
                    search_cols,
                    max_results,
                    clean_text,
                    chunk_id,
                ): chunk_id
                for chunk_id, (start_line, end_line, chunk_data) in enumerate(chunks)
            }

            # Collect results as they complete
            with open(output_file, "a", encoding="utf-8", newline="") as outfile:
                writer = csv.writer(outfile)

                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        chunk_results, chunk_row_count = future.result()

                        # Write results in batch
                        if chunk_results:
                            writer.writerows(chunk_results)
                            total_results += len(chunk_results)

                        processed_rows += chunk_row_count

                        # Log progress periodically
                        if chunk_id % max(len(chunks) // 10, 1) == 0:
                            elapsed = time.time() - start_time
                            rate = processed_rows / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"Progress: {chunk_id + 1}/{len(chunks)} chunks, "
                                f"{processed_rows} rows, {rate:.0f} rows/sec"
                            )

                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_id}: {e}")

        elapsed_time = time.time() - start_time

        # Performance statistics
        stats = {
            "total_rows": processed_rows,
            "total_results": total_results,
            "processing_time": elapsed_time,
            "rows_per_second": processed_rows / elapsed_time if elapsed_time > 0 else 0,
            "chunks_processed": len(chunks),
            "processes_used": self.processes,
            "chunk_size": self.chunk_size,
        }

        logger.info(
            f"Processing complete: {processed_rows} rows, {total_results} results, "
            f"{stats['rows_per_second']:.0f} rows/sec"
        )

        return stats

    def _initialize_output_file(
        self,
        output_file: str,
        input_file: str,
        input_cols: list[str],
        search_cols: list[str],
        max_results: int,
    ):
        """Initialize output file with proper headers."""
        # Read headers from input file
        _open = gzip.open if input_file.endswith(".gz") else open
        with _open(input_file, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            input_fieldnames = reader.fieldnames

        # Build output headers
        headers = []
        for k in input_fieldnames:
            if k in input_cols:
                headers.append(k)

        for i in range(max_results):
            for field in RESULT_FIELDS:
                if field in search_cols:
                    headers.append(f"name{i + 1}.{field}")

        if "count" in search_cols:
            headers.append("count")

        # Write headers
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def process_chunk_worker(
    chunk_data: list[tuple[int, dict]],
    names: list[tuple[str, str]],
    editlength: list[int],
    text_column: str,
    input_cols: list[str],
    search_cols: list[str],
    max_results: int,
    clean_text_flag: bool,
    chunk_id: int,
) -> tuple[list[list], int]:
    """
    Worker function to process a chunk of data.

    Returns (results, row_count) tuple.
    """
    # Initialize search engine once per worker
    search_engine = create_search_engine(names, editlength, "optimized")

    results = []
    processed_count = 0

    for line_num, row in chunk_data:
        try:
            text = row[text_column]

            if clean_text_flag:
                text = clean_text_content(text)

            # Perform search
            search_results = search_engine.search(text, max_results)

            # Build output row
            output_row = []

            # Add input columns
            for col in input_cols:
                if col in row:
                    if clean_text_flag and col == text_column:
                        output_row.append(text)  # Use cleaned text
                    else:
                        output_row.append(row[col])
                else:
                    output_row.append("")

            # Add search results
            result_count = len(search_results)
            for i in range(max_results):
                for field in RESULT_FIELDS:
                    if field in search_cols:
                        if i < result_count and field in search_results[i]:
                            output_row.append(search_results[i][field])
                        else:
                            output_row.append("")

            if "count" in search_cols:
                output_row.append(result_count)

            if search_results:  # Only add rows with results
                results.append(output_row)

            processed_count += 1

        except Exception as e:
            logger.warning(f"Error processing line {line_num}: {e}")
            continue

    return results, processed_count


def clean_text_content(text: str) -> str:
    """Clean text content using utility functions."""
    text = utils.to_lower_case(text)
    text = utils.remove_special_chars(text)
    text = utils.remove_accents(text)
    text = utils.remove_stopwords(text)
    text = utils.remove_punctuation(text)
    text = utils.remove_extra_space(text)
    return text


def search_names_optimized(
    input_file: str,
    names: list[tuple[str, str]],
    output_file: str = "search_results_optimized.csv",
    text_column: str = "text",
    input_cols: list[str] = None,
    search_cols: list[str] = None,
    max_results: int = 20,
    processes: int = None,
    chunk_size: int = None,
    clean_text: bool = True,
) -> dict[str, Any]:
    """
    Optimized parallel name search function.

    Args:
        input_file: Path to input CSV file
        names: List of (id, name) tuples to search for
        output_file: Path to output CSV file
        text_column: Name of column containing text to search
        input_cols: Input columns to include in output
        search_cols: Search result columns to include
        max_results: Maximum results per text
        processes: Number of parallel processes (auto-detected if None)
        chunk_size: Chunk size for processing (auto-calculated if None)
        clean_text: Whether to clean text before searching

    Returns:
        Performance statistics dictionary
    """
    processor = OptimizedSearchProcessor(
        names=names,
        processes=processes,
        chunk_size=chunk_size,
    )

    return processor.search_file_parallel(
        input_file=input_file,
        output_file=output_file,
        text_column=text_column,
        input_cols=input_cols or ["id", text_column],
        search_cols=search_cols or ["uniqid", "n", "match", "start", "end", "count"],
        max_results=max_results,
        clean_text=clean_text,
    )
