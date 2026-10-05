"""Download the MobileSAM checkpoint (about 40 MB) to weights/mobile_sam.pt."""

import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
target = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
target.parent.mkdir(exist_ok=True)
if target.is_file() and target.stat().st_size > 1_000_000:
    sys.exit(f"{target} already exists")
urllib.request.urlretrieve(URL, target)
print(f"saved {target} ({target.stat().st_size / 1e6:.0f} MB)")
