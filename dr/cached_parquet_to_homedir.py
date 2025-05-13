import upath

version = 'v0.0.268'

CACHE = upath.UPath(f's3://aind-scratch-data/dynamic-routing/cache/nwb_components/{version}/consolidated')
HOMEDIR = upath.UPath('//allen/ai/homedirs/ben.hardcastle/dr-dashboard/data')

(HOMEDIR / version).mkdir(parents=True, exist_ok=True)
for file in CACHE.glob('*.parquet'):
    (HOMEDIR / version / file.name).write_bytes(file.read_bytes())
    print(f"Copied {file} to {HOMEDIR / version / file.name}")