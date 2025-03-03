import pathlib
import polars as pl
import tqdm

df = (
    pl.read_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/session_metadata/tables/sessions.parquet")
    .filter(
        pl.col('is_production'),
        ~pl.col('is_opto_perturbation'),
        ~pl.col('is_injection_perturbation'),
        ~pl.col('is_templeton'),
        pl.col('issues').list.len() == 0,
    )
)
print(len(df))

src = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/qc")
dest = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/qc_prod_20250110")

for session_id in tqdm.tqdm(df['session_id']):
    for file in src.rglob(f"{session_id}*"):
        dest_file = dest / file.relative_to(src)
        if dest_file.exists():
            pass
            continue
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(file.read_bytes())
        # print(f"Copied {file.relative_to(src)} to {dest_file.relative_to(dest)}")