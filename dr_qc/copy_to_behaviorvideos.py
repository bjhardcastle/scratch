import tqdm
import pathlib
import concurrent.futures as cf

def get_behavior_without_ephys_vid_paths(session_dir: str) -> tuple[pathlib.Path, pathlib.Path]:
    session_dir = pathlib.Path(session_dir)
    if not session_dir.is_dir():
        return ()
    vid_file = next(session_dir.glob('Behavior*.mp4'), None)
    if not vid_file:
        return ()
    json_file = vid_file.with_suffix('.json')
    if not json_file.exists():
        return ()
    settings_xml_files = tuple(session_dir.rglob('settings*.xml'))
    if settings_xml_files:
        return ()
    return (vid_file, json_file)

def copy_files(vid_file: pathlib.Path, json_file: pathlib.Path):
    mid = vid_file.parent.name.split('_')[1]
    dest = pathlib.Path(f'//allen/programs/mindscope/workgroups/dynamicrouting/behaviorvideos/{mid}')
    dest.parent.mkdir(exist_ok=True, parents=True)
    for file in (vid_file, json_file):
        (dest / file.name).write_bytes(file.read_bytes())
        
def helper(session_dir: str) -> None:
    
    files = get_behavior_without_ephys_vid_paths(session_dir)
    
    if files:
        copy_files(*files)
    

with cf.ThreadPoolExecutor() as pool:
    futures = []
    for p in pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot').iterdir():
        futures.append(pool.submit(helper, p))
    for p in tqdm.tqdm(cf.as_completed(futures), total=len(futures)):
        pass
    