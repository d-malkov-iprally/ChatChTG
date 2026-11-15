"""Build truth cache for all plans: var/room-truth/<stem>.json"""
import sys, json, glob, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from chatchtg.evals.room_truth import read_truth
OUT=Path("var/room-truth"); OUT.mkdir(parents=True,exist_ok=True)
def one(stem):
    o=OUT/f"{stem}.json"
    if o.exists(): return 0
    try:
        img=np.array(Image.open(f"data/floorplans/data/swiss_dwellings/{stem}.png").convert("RGB"))
        t=read_truth(Path(f"data/full/data/swiss_dwellings/{stem}.dxf"),img)
        o.write_text(json.dumps(t.to_json()))
        return 1
    except Exception as e:
        print(stem,"ERR",repr(e),flush=True); return 0
if __name__=="__main__":
    stems=sorted(Path(p).stem for p in glob.glob("data/floorplans/data/swiss_dwellings/*.png"))
    with ProcessPoolExecutor(8) as ex: n=sum(ex.map(one,stems,chunksize=4))
    print("built",n,"of",len(stems))
