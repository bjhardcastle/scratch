# Use python for csv parsing.
import csv
import polars as pl
# Used to register a new generator on every instantiation.
from polars.io.plugins import register_io_source
from typing import Iterator
import io

def parse_schema(csv_str: str) -> pl.Schema:
    first_line = csv_str.split("\n")[0]

    return pl.Schema({k: pl.String for k in first_line.split(",")})


def my_scan_csv(csv_str: str) -> pl.LazyFrame:
    schema = parse_schema(csv_str)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source.
        This function will be registered as IO source.
        """
        if batch_size is None:
            batch_size = 100

        # Initialize the reader.
        reader = csv.reader(io.StringIO(csv_str), delimiter=",")
        # Skip the header.
        _ = next(reader)

        # Ensure we don't read more rows than requested from the engine
        print(f"n_rows: {n_rows}")
        while n_rows is None or n_rows > 0:
            if n_rows is not None:
                batch_size = min(batch_size, n_rows)

            rows = []

            for _ in range(batch_size):
                try:
                    row = next(reader)
                except StopIteration:
                    n_rows = 0
                    break
                rows.append(row)

            df = pl.from_records(rows, schema=schema, orient="row")
            n_rows -= df.height

            # If we would make a performant reader, we would not read these
            # columns at all.
            if with_columns is not None:
                df = df.select(with_columns)

            # If the source supports predicate pushdown, the expression can be parsed
            # to skip rows/groups.
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return register_io_source(io_source=source_generator, schema=schema)


csv_str2 = """a,b
1,2
9,10
1,2
1,122"""

print(
    my_scan_csv(csv_str2)
    .filter(pl.col('a') != '1')
    .slice(0, 2)
    .collect(engine='streaming')
)