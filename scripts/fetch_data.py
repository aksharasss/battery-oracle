"""Download the validation datasets (Wheeler 2025 cells, Dubarry synthetic LFP).

The Recherche Data Gouv S3 store serves ~40 KB/s per connection, so files are
fetched as many parallel byte ranges and stitched. Re-running resumes.

  python scripts/fetch_data.py wheeler-checkups --cells 6a 1b 7a --every 300   # checkups only
  python scripts/fetch_data.py wheeler --cells 6a 1b 7a 4b 5a 2b 3b             # whole zips
  python scripts/fetch_data.py dubarry
"""
import argparse
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
WHEELER_API = "https://entrepot.recherche.data.gouv.fr/api"
WHEELER_DOI = "doi:10.57745/OLBXKT"
DUBARRY_FILES = "https://data.mendeley.com/public-api/datasets/bs2j56pn7y/files?folder_id=root&version=3"
CHUNK = 2 * 2**20

# Mendeley refuses Python's default user agent
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "Mozilla/5.0 (battery-oracle data fetch)")]
urllib.request.install_opener(_opener)


def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _resolve(url):
    """Follow the repository's redirect to the signed S3 URL once."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})   # signed URL is GET-only
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl()


def _fetch_range(url, start, end, part: Path, tries=6):
    if part.exists() and part.stat().st_size == end - start + 1:
        return
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                part.write_bytes(r.read())
            if part.stat().st_size == end - start + 1:
                return
        except Exception:
            time.sleep(2 ** k)
    raise IOError(f"range {start}-{end} failed")


def download(url, size, out: Path, workers=32):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == size:
        print(f"have {out.name}")
        return
    s3 = _resolve(url)
    parts_dir = out.with_suffix(out.suffix + ".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = [(i, min(i + CHUNK, size) - 1) for i in range(0, size, CHUNK)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: _fetch_range(s3, r[0], r[1], parts_dir / f"{r[0]:012d}"), ranges))
    with open(out, "wb") as f:
        for s, _ in ranges:
            f.write((parts_dir / f"{s:012d}").read_bytes())
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    assert out.stat().st_size == size, f"{out.name}: size mismatch"
    print(f"got {out.name} ({size / 1e6:.0f} MB, {size / 1e6 / (time.time() - t0):.2f} MB/s)", flush=True)


class _RemoteZip(io.RawIOBase):
    """Seekable read-only view of a remote zip, served from prefetched byte
    ranges (falling back to small range requests), so single members can be
    extracted without downloading the whole archive."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.cache = url, size, 0, {}
        self.prefetch(max(0, size - 400_000), size - 1)   # central directory

    def prefetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for k in range(6):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    self.cache[start] = r.read()
                return
            except Exception:
                time.sleep(2 ** k)
        raise IOError(f"range {start}-{end} failed")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        for st, d in self.cache.items():
            if st <= self.pos and self.pos + n <= st + len(d):
                b[:n] = d[self.pos - st:self.pos - st + n]
                self.pos += n
                return n
        self.prefetch(self.pos, min(self.pos + max(n, 65536), self.size) - 1)
        return self.readinto(b)


def _fetch_to_file(url, start, end, out: Path, workers: int, piece=2**20):
    """Byte range start..end of url into out, in resumable 1 MB pieces."""
    if out.exists():
        return
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(a, min(a + piece, end + 1) - 1) for a in range(start, end + 1, piece)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: _fetch_range(url, r[0], r[1], parts / f"{r[0]:012d}"), ranges))
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for a, _ in ranges:
            f.write((parts / f"{a:012d}").read_bytes())
    tmp.replace(out)
    for q in parts.iterdir():
        q.unlink()
    parts.rmdir()


def wheeler_checkups(url, size, out_dir: Path, every: int, workers: int):
    """Extract the characterisation files (C/3 capacity + C/25 pseudo-OCV) of
    one cell: every `every` cycles at 50 degC, plus the 25 degC initial and
    final characterisations. Ageing files (~80% of each zip) are skipped:
    each wanted member is fetched as its own byte range and inflated here."""
    import re
    import struct
    import zipfile
    import zlib

    s3 = _resolve(url)
    rz = _RemoteZip(s3, size)
    infos = zipfile.ZipFile(io.BufferedReader(rz, buffer_size=1 << 16)).infolist()
    charac = [i for i in infos if re.search(r"_charac(\d+)\.csv$", i.filename)]
    last = max(int(re.search(r"(\d+)\.csv$", i.filename).group(1)) for i in charac)
    n = 0
    for i in infos:
        m = re.search(r"_charac(\d+)\.csv$", i.filename)
        keep = (m and (int(m.group(1)) % every == 0 or int(m.group(1)) == last)) or             i.filename.endswith(("_initialcharac.csv", "_finalcharac.csv"))
        dest = out_dir / i.filename
        if not keep or dest.exists():
            continue
        raw = dest.with_suffix(".zraw")
        raw.parent.mkdir(parents=True, exist_ok=True)
        end = min(i.header_offset + 30 + len(i.filename) + 2048 + i.compress_size, size) - 1
        _fetch_to_file(s3, i.header_offset, end, raw, workers)
        b = raw.read_bytes()
        fn_len, extra_len = struct.unpack("<HH", b[26:30])
        data = b[30 + fn_len + extra_len:30 + fn_len + extra_len + i.compress_size]
        out = zlib.decompressobj(-15).decompress(data) if i.compress_type == 8 else data
        assert len(out) == i.file_size, f"{i.filename}: bad size"
        dest.write_bytes(out)
        raw.unlink()
        n += 1
        print(f"  {dest.name}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["wheeler", "wheeler-checkups", "dubarry"])
    ap.add_argument("--every", type=int, default=300, help="wheeler-checkups: cycle spacing")
    ap.add_argument("--cells", nargs="*", default=["6a", "1b", "7a", "4b", "5a", "2b", "3b"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if args.dataset == "wheeler-checkups":
        out_dir = RAW / "wheeler2025"
        files = _get_json(f"{WHEELER_API}/datasets/:persistentId/?persistentId={WHEELER_DOI}")
        files = {f["dataFile"]["filename"]: f["dataFile"] for f in files["data"]["latestVersion"]["files"]}
        for c in args.cells:
            df = files[f"Cell{c}.zip"]
            t0 = time.time()
            n = wheeler_checkups(f"{WHEELER_API}/access/datafile/{df['id']}", df["filesize"],
                                 out_dir, args.every, args.workers)
            print(f"Cell{c}: {n} checkup files in {time.time() - t0:.0f} s", flush=True)
    elif args.dataset == "wheeler":
        out_dir = RAW / "wheeler2025"
        files = _get_json(f"{WHEELER_API}/datasets/:persistentId/?persistentId={WHEELER_DOI}")
        files = {f["dataFile"]["filename"]: f["dataFile"] for f in files["data"]["latestVersion"]["files"]}
        wanted = [f"Cell{c}.zip" for c in args.cells]
        for name in wanted:
            df = files[name]
            download(f"{WHEELER_API}/access/datafile/{df['id']}", df["filesize"], out_dir / name, args.workers)
    else:
        out_dir = RAW / "dubarry_lfp_synth"
        for f in _get_json(DUBARRY_FILES):
            d = f["content_details"]
            download(d["download_url"], d["size"], out_dir / f["filename"], 8)


if __name__ == "__main__":
    sys.exit(main())
