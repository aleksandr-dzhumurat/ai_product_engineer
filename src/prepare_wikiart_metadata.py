"""Export local WikiArt JSON records as a Hugging Face JSONL dataset.

python src/prepare_wikiart_metadata.py
Upload the resulting folder only after reviewing README.md and quality_report.json.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile


def prepare(source, output):
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.wikiart-hf-', dir=output.parent))
    counts = Counter()
    styles = Counter()
    errors = []
    try:
        with (staging / 'train.jsonl').open('w', encoding='utf-8') as stream:
            for path in sorted(source.glob('*.json')):
                try:
                    record = json.loads(path.read_text(encoding='utf-8'))
                    if not isinstance(record, dict):
                        raise ValueError('Expected JSON object')
                    def text(key):
                        value = record.get(key)
                        return value if isinstance(value, str) else ''
                    location = record.get('location')
                    if not isinstance(location, dict):
                        location = {}
                    artist_link = text('artist_link')
                    row = {
                        'id': path.stem,
                        'title': text('title'), 'artist': text('artist'),
                        'artist_url': ('https://www.wikiart.org' + artist_link
                                       if artist_link.startswith('/') else artist_link),
                        'date': text('date'), 'style': text('style'), 'genre': text('genre'),
                        'image_url': text('image_url') or text('image'),
                        'venue': str(location.get('venue') or ''),
                        'city': str(location.get('city') or ''),
                        'country': str(location.get('country') or ''),
                        'source_file': path.name,
                    }
                    # The local filename is retained as an ID, not asserted to be a verified page URL.
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    counts['records'] += 1
                    for field in ['title', 'artist', 'style', 'genre', 'image_url']:
                        if not row[field]:
                            counts[f'missing_{field}'] += 1
                    styles[row['style']] += 1
                except (ValueError, OSError) as exc:
                    errors.append({'file': path.name, 'error': str(exc)})
        if not counts['records']:
            raise ValueError('No valid records found')
        report = {'counts': dict(counts), 'invalid_files': errors,
                  'styles': dict(styles.most_common())}
        (staging / 'quality_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (staging / 'README.md').write_text('''---
language:
- en
task_categories:
- feature-extraction
tags:
- art
- wikiart
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# WikiArt artwork metadata

Local export of scraped WikiArt artwork metadata. Images are **not included**;
`image_url` points to an external resource and does not download image bytes.

''' + f"Records: {counts['records']:,}. Invalid input files: {len(errors)}.\n\n" + '''## Schema

`id`, `title`, `artist`, `artist_url`, `date`, `style`, `genre`, `image_url`,
`venue`, `city`, `country`, `source_file`: string columns. Missing values are empty
strings. IDs come from local source filenames; dates remain unparsed source text.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="data/wikiart_hf_metadata/train.jsonl", split="train")
```

After uploading, use `load_dataset("OWNER/REPOSITORY")`.

## Provenance and limitations

Source: https://www.wikiart.org/ via the repository's local scraped JSON files.
The export preserves source labels; they have not been manually verified.
No image license is inferred, and this export does not grant rights to reproduce
artworks. Source-specific permissions and attribution require separate review.
Collection dates and scraping coverage are not established by this export.

The `train` name is a distribution convention: no held-out evaluation split is
provided. Split by artist for experiments evaluating generalization to new artists.
Different records can describe the same artwork; semantic deduplication has not
been performed. See `quality_report.json` for missing fields and invalid files.

## Upload

Install `huggingface_hub` and authenticate with `hf auth login` or `HF_TOKEN`.
Replace `OWNER/REPOSITORY` below. This command creates a private dataset repository.

```bash
hf repo create OWNER/REPOSITORY --repo-type dataset --private
hf upload OWNER/REPOSITORY data/wikiart_hf_metadata . --repo-type dataset
```

The folder contains only the exported JSONL, this card and the quality report;
raw HTML, credentials and unrelated local files are not included.
''', encoding='utf-8')
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/artworks/artworks_json'))
    parser.add_argument('--output', type=Path, default=Path('data/wikiart_hf_metadata'))
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps({'output': str(args.output), 'counts': report['counts'],
                      'invalid_files': len(report['invalid_files'])}, indent=2))


if __name__ == '__main__':
    main()
