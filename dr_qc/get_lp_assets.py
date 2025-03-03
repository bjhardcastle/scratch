import concurrent.futures as cf
from enum import IntEnum
import logging
import time

import aind_session
import polars as pl
import tqdm
import upath
import upath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sessions_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/session_metadata/tables/sessions.parquet"
).filter(
    pl.col("is_production"),
    pl.col("is_video"),
)

DB_PATH = "//allen/programs/mindscope/workgroups/dynamicrouting/ben/gamma_encoded_vid_paths.parquet"

def get_lp_vid_paths(session_id: str) -> tuple[upath.UPath, ...]:
    sessions = aind_session.get_sessions(
        subject_id=session_id.split("_")[0],
        date=session_id.split("_")[1],
        platform="ecephys",
    )
    if not sessions:
        return ()
    if len(sessions) > 1:
        logger.info(f"Multiple sessions found, using earliest: {sessions}")
        # sessions are sorted chronologically
    session = sessions[0]
    lp_assets = [asset for asset in session.data_assets if "LPFaceParts" in asset.name]
    if not lp_assets:
        logger.warning(f"No LPFaceParts assets found for {session.id}")
    for lp_asset in aind_session.sort_by_created(lp_assets)[::-1]:
        asset_root = aind_session.get_data_asset_source_dir(lp_asset.id)
        gamma_vid_dir = next((p for p in asset_root.glob("*GammaEncoding*")), None)
        if gamma_vid_dir is None:
            continue
        else:
            return tuple(gamma_vid_dir.rglob("*.mp4"))
    else:
        logger.warning(
            f"No LPFaceParts assets contain GammaEncoding folder for {session.id}"
        )
        return ()


def get_session_id_to_vid_paths() -> dict[str, tuple[upath.UPath, ...]]:
    t0 = time.time()
    with cf.ThreadPoolExecutor() as executor:
        future_to_session_id = {}
        session_id_to_paths = {}
        for session_id in sessions_df["session_id"]:
            future = executor.submit(get_lp_vid_paths, session_id)
            future_to_session_id[future] = session_id
        for future in cf.as_completed(future_to_session_id):
            session_id_to_paths[future_to_session_id[future]] = future.result()
    logger.info(f"Got all vid paths in: {time.time() - t0:.2f}s")
    return session_id_to_paths

def create_db(db_path: str | None = DB_PATH) -> pl.DataFrame:
    records = []
    for session_id, paths in get_session_id_to_vid_paths().items():
        if not paths:
            records.append(
                {
                    "session_id": session_id,
                    "path": None,
                    "video_appearance_rating": None,
                    "lp_tracking_rating": None,
                }
            )
        else:
            for path in paths:
                record = {
                    "session_id": session_id,
                    "path": path.as_posix(),
                    "video_appearance_rating": None,
                    "lp_tracking_rating": None,
                }
                records.append(record)
    df = pl.from_records(records)
    if db_path:
        df.write_parquet(db_path)
        logger.info(f"Created db at {db_path}")
    return df

def update_db(db_path: str = DB_PATH) -> None:
    raise NotImplementedError("Not sure how to update yet")
    # copy original
    original_df = pl.read_parquet(db_path)
    original_df.write_parquet(db_path.replace('.parquet', f".parquet.{time.time()}.backup"))
    # update
    new_df = create_db(db_path=None) # create in memory without saving
    (
        original_df
        .drop_nulls('path')
    )
    

class Lock:
    def __init__(self, path=DB_PATH + ".lock"):
        self.path = upath.UPath(path)

    def acquire(self):
        while self.path.exists():
            time.sleep(0.01)
        self.path.touch()

    def release(self):
        upath.UPath(self.path).unlink()

    def __enter__(self):
        self.acquire()
        logger.info(f"Acquired lock at {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        logger.info(f"Released lock at {self.path}")

class Rating(IntEnum):
    BAD = 0
    UNSURE = 1
    GOOD = 2
    

def get_df(db_path: str = DB_PATH) -> pl.DataFrame:
    return pl.read_parquet(db_path)

def set_rating_in_db(
    path: str, 
    video_appearance_rating: int, 
    lp_tracking_rating: int, 
    db_path: str = DB_PATH,
) -> None:
    timestamp = int(time.time())
    original_df = get_df()
    logger.info(f"Updating row for {path} with {video_appearance_rating=}, {lp_tracking_rating=}")
    df = original_df.with_columns(
        video_appearance_rating=pl.when(pl.col("path") == path)
        .then(pl.lit(video_appearance_rating))
        .otherwise(pl.col("video_appearance_rating")), # keep existing rating
        lp_tracking_rating=pl.when(pl.col("path") == path)
        .then(pl.lit(lp_tracking_rating))
        .otherwise(pl.col("lp_tracking_rating")), # keep existing rating
        checked_timestamp=pl.when(pl.col("path") == path)
        .then(pl.lit(timestamp))
        .otherwise(pl.col("checked_timestamp")),
    )
    assert len(df) == len(
        original_df
    ), f"Row count changed: {len(original_df)} -> {len(df)}"
    with Lock():
        df.write_parquet(db_path)
    logger.info(f"Overwrote {db_path}")
    
