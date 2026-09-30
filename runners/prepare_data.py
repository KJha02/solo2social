"""Prepare the paper's fixed IFBench split; makes no model calls."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request
import certifi
import nltk


def main():
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    root = Path(__file__).resolve().parents[1]
    # Upstream ships a different copy inside its Python package. Use the
    # repository's data/ file, which supplied the paper's recorded split.
    source = root / 'data/IFBench_test.jsonl'
    expected = 'd2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a'
    url = ('https://raw.githubusercontent.com/allenai/IFBench/'
           '1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl')
    data = source.read_bytes() if source.exists() else urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('IFBench source checksum differs from the paper')
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    for resource in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"):
        if not nltk.download(resource, download_dir=str(root / "data/nltk_data"), quiet=True):
            raise RuntimeError(f"Unable to download NLTK resource: {resource}")
    from rung4_instruction import prepare_dataset
    manifest = prepare_dataset(root / "data/ifbench_train100_test200.json",
                               source, "ifbench", 100, 200, split_seed=0)
    print(json.dumps({k: v for k, v in manifest.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
