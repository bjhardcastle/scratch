import functools
import pathlib
from npc_io import V
import polars as pl
import polars.selectors as cs
import tqdm

SRC = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/qc")
DEST = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/qc_prod_20250110")
EXISTING = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/qc_prod_20250110/existing.csv")

def get_prod_sessions_df() -> pl.DataFrame:
    return (
        get_all_sessions_df()
        .filter(
            pl.col('is_production'),
            ~pl.col('is_opto_perturbation'),
            ~pl.col('is_injection_perturbation'),
            ~pl.col('is_templeton'),
            # pl.col('issues').list.len() == 0,
        )
    )

def get_all_sessions_df() -> pl.DataFrame:
    return pl.read_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/session_metadata/tables/sessions.parquet")
    
def copy_prod_qc_items(src: pathlib.Path, dest: pathlib.Path) -> pl.DataFrame:
    existing = pl.read_csv(EXISTING)
    records = []
    for session_id in tqdm.tqdm(get_prod_sessions_df()['session_id']):
        record = {
            'session_id': session_id,
            'is_production': True,
            'project': 'DynamicRouting',
            'has_issues': None,
            'checked': None,
            'TODO': None,
        }
        for file in src.rglob(f"{session_id}*"):
            evaluation_name = f"{file.parent.parent.name}_{file.parent.name}"
            if "sorting_view" in evaluation_name:
                continue
            dest_file = dest / file.relative_to(src)
            if file.suffix == '.error':
                msg = file.read_text().splitlines()[-1]
            else:
                msg = 'exists'
            record[evaluation_name] = msg
            if dest_file.exists():
                pass
                # continue
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_bytes(file.read_bytes())
        for k, v in existing.filter(pl.col(' session_id') == session_id).to_dicts()[0].items():
            if v and isinstance(v, (bool, str)):
                record[k] = v
        records.append(record)
    df = (
        pl.from_records(records)
        .with_columns(pl.all().fill_null('[missing]'))
        .with_columns(pl.all().str.replace('exists', ""))
    )
    return df

def main():
    qc_evaluations_df = copy_prod_qc_items(SRC, DEST)
    qc_evaluations_df.write_csv(DEST / "qc_evaluations.csv")
    missing = (
        get_prod_sessions_df()
        .join(qc_evaluations_df, on='session_id', how='anti')
        .select(cs.string(), cs.numeric(), cs.boolean())
    )
    missing.write_csv(DEST / "missing_sessions.csv")
    print(f"Done: {len(missing)} missing sessions found")

if __name__ == "__main__":
    main()

