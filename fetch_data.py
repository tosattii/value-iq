"""Step 1: download the raw CSVs with requests."""
import requests

from config import DATA_URL, RAW_DIR, RAW_FILES


def download(name: str) -> None:
    target = RAW_DIR / name
    url = f"{DATA_URL}/{name}"
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  saved {target.relative_to(RAW_DIR.parent.parent)} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        download(name)


if __name__ == "__main__":
    main()
