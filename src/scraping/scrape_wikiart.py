"""Collect the WikiArt dataset, then build descriptions and a quality report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from stat import S_ISREG
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

if TYPE_CHECKING:
    import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

if __package__:
    from . import utils
else:
    import utils

logger = utils.logger


def load_config() -> dict:
    import yaml

    default_config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    config_path = os.getenv("CONFIG_PATH", default_config_path)
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['root_data_dir'] = os.getenv("ROOT_DATA_DIR", config['root_data_dir'])
    os.makedirs(config['root_data_dir'], exist_ok=True)
    config['data_version_dir'] = os.path.join(config['root_data_dir'], config['data_version'])
    os.makedirs(config['data_version_dir'], exist_ok=True)
    return config


def artifact_path(artifact_name: str, config: dict):
    return os.path.join(config['data_version_dir'], artifact_name)

def generate_files():
    basepath = '/srv/src'
    result = []
    for fname in os.listdir(basepath):
        path = os.path.join(basepath, fname)
        if os.path.isdir(path):
            # skip directories
            continue
        result.append(os.path.join(basepath, fname))
    return result

def files_version() -> bytes:
    hash = hashlib.md5()
    for fn in generate_files():
        try:
            hash.update(Path(fn).read_bytes())
        except IsADirectoryError:
            pass
    return str(hash.hexdigest())[:8]


def request_retries(url, num_retries: int = 3):
    for attempt in range(num_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == num_retries - 1:
                raise
            time.sleep(0.25)
    raise ValueError('num_retries must be positive')

BASE_URL = "https://www.wikiart.org"
DEFAULT_ARTWORK_DOWNLOAD_WORKERS = 4


def get_html(url, html_dir_name='', root_data_dir='data'):
    html_path = Path(_build_html_path(url, html_dir_name, root_data_dir))
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    html_content = request_retries(url).text
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_content, encoding='utf-8')
    return html_content


def n_gram_split(input_str: str) -> list[str]:
    tokens = input_str.split(' ')
    return [
        ' '.join(tokens[start:start + size])
        for size in range(1, 4)
        for start in range(len(tokens) - size + 1)
    ]


@dataclass
class Artist:
    artist_name: str
    artist_link: str
    artist_url: str
    is_popular: bool = False


def prepare_pages_list() -> list[str]:
    import string

    alphabet_url = f"{BASE_URL}/en/Alphabet"
    postfix = 'text-list'

    alphabet = list(string.ascii_lowercase)
    pages_list = [os.path.join(alphabet_url, page_liter, postfix) for page_liter in alphabet]
    return pages_list

def get_artists_pages(result_csv_path: str, popular_artists: list[str] | None = None):
    import pandas as pd

    if popular_artists is None:
        popular_artists = []
    popular_set = set(popular_artists)
    page_list: list[str] = prepare_pages_list()
    if not os.path.exists(result_csv_path):
        all_rows_list = []
        for current_alphabet_page in page_list:
            artists_scraper = BeautifulSoup(
                markup=request_retries(current_alphabet_page).content, features="html.parser"
            )
            res = artists_scraper.find(name="div", class_='masonry-text-view masonry-text-view-all')
            if res is None:
                raise ValueError(f'Could not find the artist listing at {current_alphabet_page}')
            artists = res.find_all(name='li')
            logger.info('Current page: %s, num artists: %d', current_alphabet_page, len(artists))
            rows_list = []
            for artist in artists:
                artist_info = artist.find('a', href=True)
                if artist_info is None:
                    continue
                artist_url = f"{BASE_URL}{artist_info['href']}"
                rows_list.append(Artist(
                    artist_name=artist_info.get_text(),
                    artist_link=artist_info['href'],
                    artist_url=artist_url,
                    is_popular=artist_url in popular_set,
                ))
            all_rows_list.extend(rows_list)
        res_df = pd.DataFrame([asdict(a) for a in all_rows_list], columns=list(Artist.__annotations__))
        res_df = res_df.drop_duplicates(subset='artist_link')
        res_df.to_csv(result_csv_path, index=False)
    else:
        res_df = pd.read_csv(result_csv_path)
    logger.info("Artists pages saved to %s: %d rows", result_csv_path, res_df.shape[0])

def extract_artist_wiki(artist_info_scraper: BeautifulSoup) -> str:
    wiki_text = 'Empty wiki'
    try:
        wiki_text = artist_info_scraper.find(id='info-tab-wikipediaArticle', class_='wiki-layout-artist-info-tab').get_text()
        if wiki_text is None:
            raise AttributeError
    except AttributeError:
        pass
    return wiki_text.replace('\n', ' ')

def extract_artists_info(artist_page_scraper: BeautifulSoup) -> dict[str, str]:
    res_dict = {}
    artist_info = artist_page_scraper.find(name='div', class_='wiki-layout-artist-info')
    if artist_info is not None:
        artist_properties = artist_info.find_all(name='li')
        if artist_properties is not None:
            for prop in artist_properties:
                prop_html = prop.find(name='s')
                if prop_html is not None and prop_html.get_text() is not None:
                    property_name = prop_html.get_text().strip().lower().replace(':', '')
                    if property_name != 'share':
                        links = prop.find_all(name='a')
                        if links:
                            prop_values = ', '.join(
                                i.get_text().strip().replace('\n', ' ') for i in links
                            )
                        else:
                            prop_values = prop.get_text().strip().replace('\n', ' ')
                            # Remove the label prefix (e.g. "Art Movement:")
                            label = prop_html.get_text().strip()
                            if prop_values.startswith(label):
                                prop_values = prop_values[len(label):].strip()
                        res_dict.update({property_name: prop_values})
            image = artist_page_scraper.select_one('.wiki-layout-artist-image-wrapper img[src]')
            res_dict['artist_pic'] = image['src'] if image else ''
    res_dict['artist_field'] = process_field(res_dict.get('field', ''))
    res_dict['artist_movement'] = process_art_movement(res_dict.get('art movement', ''))
    return res_dict

def process_field(raw_str):
    if not isinstance(raw_str, str):
        return ''
    try:
        res = ' '.join(sorted(set([
            i.strip().replace(',', '')
            for i in raw_str.split(' ')
            if len(i.strip().replace(',', '')) > 0]))
        )
    except Exception as e:
        logger.error('Error occurred while processing raw string: %s', raw_str)
        raise e
    return res

def get_artists_info(input_csv_path: str, output_csv_path: str, output_wikitext_csv_path: str, root_data_dir='data'):
    import pandas as pd

    if not (os.path.exists(output_csv_path) and os.path.exists(output_wikitext_csv_path)):
        wiki_descriptions = []
        artists_info = []
        input_df = pd.read_csv(input_csv_path)
        logger.info('Artists information (wiki, etc) scraping started: %d rows', input_df.shape[0])
        for ind, row in input_df.iterrows():
            logger.info('[%d/%d] Fetching artist: %s', ind + 1, input_df.shape[0], row['artist_name'])
            try:
                artist_info_page = get_html(row['artist_url'], 'artists_raw_html', root_data_dir)
            except requests.RequestException as exc:
                logger.warning('Could not fetch %s: %s', row['artist_url'], exc)
                artist_info_page = None
            artist_dict = {
                'artist_name': row['artist_name'], 'artist_link': row['artist_link'],
                'artist_url': row['artist_url'], 'request_result_success': False
            }
            wiki_text = 'Empty wiki'
            if artist_info_page is not None:
                artist_info_scraper = BeautifulSoup(markup=artist_info_page, features="html.parser")
                wiki_text = extract_artist_wiki(artist_info_scraper)
                artist_info = extract_artists_info(artist_info_scraper)
                artist_dict.update(artist_info)
                artist_dict.update({'request_result_success': True})
                logger.info('[%d/%d] OK: %s', ind + 1, input_df.shape[0], row['artist_name'])
            else:
                logger.warning('[%d/%d] FAILED (no page): %s', ind + 1, input_df.shape[0], row['artist_name'])
            wiki_descriptions.append((row['artist_link'], wiki_text))
            artists_info.append(artist_dict)
            if ind % 100 == 0:
                logger.info('Progress: %d of %d artists processed', ind, input_df.shape[0])
        artists_info_df = pd.json_normalize(artists_info) if artists_info else input_df
        artists_info_df.to_csv(output_csv_path, index=False)
        logger.info('authors info saved to %s', output_csv_path)
        (
            pd.DataFrame(wiki_descriptions, columns=['artist_link', 'wiki_text'])
            .to_csv(output_wikitext_csv_path, index=False)
        )
        logger.info('wiki texts saved to %s', output_wikitext_csv_path)
    else:
        logger.info('Artists info already exists files: %s; %s', output_csv_path, output_wikitext_csv_path)

def artwork_image_sources(soup: BeautifulSoup, *, artwork_info: Tag | None = None) -> tuple[str | None, str | None]:
    """Return raw (metadata image, wrapper image) sources, without URL normalization.

    An explicit info block preserves the description parser's first-match
    behavior; the URL collector instead looks for images with a src attribute.
    """
    metadata_selector = 'img[itemprop="image"]'
    wrapper_selector = '.wiki-layout-artist-image-wrapper'
    if artwork_info is None:
        metadata = soup.select_one(f'.wiki-layout-artwork-info {metadata_selector}[src]')
        wrapper_image = soup.select_one(f'{wrapper_selector} img[src]')
    else:
        metadata = artwork_info.select_one(metadata_selector)
        wrapper = soup.select_one(f'div{wrapper_selector}')
        wrapper_image = wrapper.find('img') if wrapper is not None else None
    return (
        metadata.get('src') if metadata is not None else None,
        wrapper_image.get('src') if wrapper_image is not None else None,
    )


def _read_artwork_html(filepath: str | Path) -> str:
    path = Path(filepath)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as file:
        return file.read()


def extract_artwork_description(filepath: str | Path) -> dict:
    artwork_page = _read_artwork_html(filepath)

    artwork_scraper = BeautifulSoup(markup=artwork_page, features="html.parser")
    artwork_info = artwork_scraper.find(name='div', class_='wiki-layout-artist-info wiki-layout-artwork-info')

    if artwork_info is None:
        raise ValueError(f'Could not find artwork info block in {filepath}')

    title = ''
    artist = ''
    artist_link = ''

    title_node = artwork_info.find(name='h1')
    if title_node is not None:
        title = title_node.get_text(strip=True)

    artist_node = artwork_info.find(name='h2')
    if artist_node is not None:
        artist_a = artist_node.find(name='a')
        artist = artist_a.get_text(strip=True) if artist_a else artist_node.get_text(strip=True)
        if artist_a and artist_a.get('href'):
            artist_link = artist_a['href']

    details = dict.fromkeys(('date', 'style', 'genre', 'location'), '')
    article = artwork_info.find('article')
    if article is not None:
        remaining = set(details)
        for li in article.find_all('li'):
            label_node = li.find(name='s')
            if label_node is None:
                continue
            label = label_node.get_text(strip=True).lower().rstrip(':')
            if label not in remaining:
                continue
            li_copy = BeautifulSoup(str(li), features="html.parser")
            for tag in li_copy.find_all(name='s'):
                tag.extract()
            details[label] = li_copy.get_text(' ', strip=True)
            remaining.remove(label)
            if not remaining:
                break

    location_parts = [part.strip() for part in details['location'].split(',') if part.strip()]
    venue, city = (location_parts + ['', ''])[:2]

    image, image_url = artwork_image_sources(artwork_scraper, artwork_info=artwork_info)

    images = []
    variants_container = artwork_info.find(name='ul', class_='image-variants-container')
    if variants_container is not None:
        for a_tag in variants_container.find_all(name='a', attrs={'data-image-url': True}):
            url = a_tag['data-image-url'].strip()
            if url:
                images.append(url)

    return {
        'title': title,
        'artist': artist,
        'artist_link': artist_link,
        'date': details['date'].replace('c. ', 'c.'),
        'style': details['style'],
        'genre': details['genre'],
        'location': {
            'venue': venue,
            'city': city,
            'country': ', '.join(location_parts[2:]),
        },
        'image': image if image is not None else '',
        'image_url': image_url if image_url is not None else '',
        'images': images,
    }

def _process_artwork_file(args: tuple[str, str]) -> tuple[str, str]:
    """Worker function for multiprocessing. Returns (status, filename).
    status is one of: 'processed', 'skipped', 'failed'
    """
    html_filepath, output_json_dir = args
    html_filename = os.path.basename(html_filepath)
    json_filename = f"{html_filename.removesuffix('.gz').removesuffix('.html')}.json"
    json_filepath = os.path.join(output_json_dir, json_filename)

    if os.path.exists(json_filepath):
        return ('skipped', html_filename)

    try:
        artwork_description = extract_artwork_description(html_filepath)
        with open(json_filepath, 'w', encoding='utf-8') as file:
            json.dump(artwork_description, file, ensure_ascii=False, indent=4)
        return ('processed', html_filename)
    except Exception as e:
        logger.warning('Failed to extract artwork description from %s: %s', html_filename, e)
        return ('failed', html_filename)

def extract_artworks_descriptions(
    raw_html_dir: str,
    output_json_dir: str,
    num_workers: int,
):

    os.makedirs(output_json_dir, exist_ok=True)

    if not os.path.exists(raw_html_dir):
        logger.warning('Raw artworks html directory does not exist: %s', raw_html_dir)
        return

    # Either cache format must produce the same JSON, even if both files exist.
    by_name = {}
    for path in Path(raw_html_dir).iterdir():
        if path.name.endswith(('.html', '.html.gz')) and _nonempty_cache_file(path):
            name = path.name.removesuffix('.gz')
            if name not in by_name or path.suffix == '.gz':
                by_name[name] = str(path)
    html_files = sorted(by_name.values())

    logger.info('Artwork description extraction started. Num html files: %d, workers: %d', len(html_files), num_workers)

    counts = Counter(processed=0, skipped=0, failed=0)
    worker_args = [(filepath, output_json_dir) for filepath in html_files]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(_process_artwork_file, worker_args)

        if tqdm is not None:
            results = tqdm(results, total=len(html_files), desc='Processing artworks_raw_html', unit='file')

        for status, _ in results:
            counts[status] += 1
            if tqdm is not None:
                results.set_postfix(**counts)

    logger.info(
        'Artwork description extraction finished. processed=%d skipped=%d failed=%d output_dir=%s',
        counts['processed'], counts['skipped'], counts['failed'],
        output_json_dir,
    )

def build_artworks_json_descriptive_stats(
    json_dir: str,
    stats_output_path: str,
):

    if not os.path.exists(json_dir):
        logger.warning('Artwork json directory does not exist: %s', json_dir)
        return

    json_files = sorted(p for p in Path(json_dir).iterdir() if p.suffix == '.json')
    total_files = len(json_files)
    description_fields = ('title', 'artist', 'date', 'style', 'genre')
    location_fields = ('venue', 'city', 'country')
    empty_field_counts = Counter(dict.fromkeys(description_fields, 0))
    empty_field_counts.update(dict.fromkeys((f'location_{key}' for key in location_fields), 0))
    any_empty_description_cnt = 0
    empty_location_cnt = 0
    invalid_json_files = 0
    genre_counts, style_counts = Counter(), Counter()

    for filepath in json_files:
        try:
            with filepath.open(encoding='utf-8') as file:
                item = json.load(file)
        except Exception as e:
            invalid_json_files += 1
            logger.warning('Invalid json file %s: %s', filepath, e)
            continue

        location = item.get('location') if isinstance(item.get('location'), dict) else {}
        description = {key: (item.get(key) or '').strip() for key in description_fields}
        empty_location = [f'location_{key}' for key in location_fields if not (location.get(key) or '').strip()]
        empty_fields = [key for key, value in description.items() if not value] + empty_location
        empty_field_counts.update(empty_fields)
        any_empty_description_cnt += bool(empty_fields)
        empty_location_cnt += len(empty_location) == len(location_fields)

        for key, counts in (('genre', genre_counts), ('style', style_counts)):
            if description[key]:
                counts[description[key]] += 1

    denominator = max(1, total_files)
    stats = {
        'total_json_files': total_files,
        'json_with_any_empty_description': any_empty_description_cnt,
        'share_json_with_any_empty_description': round(any_empty_description_cnt / denominator, 6),
        'json_with_empty_location': empty_location_cnt,
        'share_json_with_empty_location': round(empty_location_cnt / denominator, 6),
        'empty_field_counts': empty_field_counts,
        'invalid_json_files': invalid_json_files,
        'genre': dict(genre_counts.most_common(20)),
        'style': dict(style_counts.most_common(20)),
    }

    with open(stats_output_path, 'w', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

    logger.info('Artwork json descriptive stats saved to %s', stats_output_path)

def _build_html_path(url, html_dir_name, root_data_dir='data'):
    html_dir = os.path.join(root_data_dir, html_dir_name)
    path_parts = [part for part in urlparse(url).path.split('/') if part]
    artist_slug = path_parts[1] if len(path_parts) > 1 and path_parts[0] == 'en' else ''
    page_slug = path_parts[-1].split('.')[0]
    html_file_name = f"{artist_slug}_{page_slug}" if artist_slug else page_slug
    return os.path.join(html_dir, f"{html_file_name}.html")


def prepare_artworks_index(artists_index: list[dict], limit: int | None = None) -> list[dict]:
    index = []
    for artist in artists_index:
        f_path = Path(artist['artist_artworks_source'])
        if not f_path.exists():
            continue
        page_content = f_path.read_text(encoding='utf-8')
        artwork_scraper = BeautifulSoup(markup=page_content, features="html.parser")
        link_set = artwork_scraper.find(name='ul', class_='painting-list-text')
        if link_set is not None:
            link_iter = link_set.find_all(name='li', class_='painting-list-text-row')
            for artwork_cnt, li in enumerate(link_iter):
                if limit is not None and artwork_cnt >= limit:
                    break
                a_tag = li.find('a')
                if a_tag and a_tag.get('href'):
                    url = urljoin(BASE_URL, a_tag['href'])
                    parsed = urlparse(url)
                    if parsed.scheme not in {'http', 'https'} or parsed.hostname not in {'wikiart.org', 'www.wikiart.org'}:
                        continue
                    index.append({
                        'source': f_path.name,
                        'artist_url': artist['artist_url'],
                        'artwork_url': url,
                    })
    logger.info('Prepared artworks index: %d items from %d artists', len(index), len(artists_index))
    return index


_DownloadItem = TypeVar('_DownloadItem')


def _download_failures(
    items: list[_DownloadItem],
    download: Callable[[_DownloadItem], object],
    *,
    num_workers: int,
    label: str,
) -> Iterator[tuple[_DownloadItem, Exception]]:
    """Run downloads and yield only failures, leaving recovery to the caller."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download, item): item for item in items}
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as exc:
                yield futures[future], exc
            if completed % 50 == 0 or completed == total:
                logger.info('Progress: %d/%d %s attempted', completed, total, label)


def _nonempty_cache_file(path: Path) -> bool:
    try:
        info = path.stat()
    except FileNotFoundError:
        return False
    return S_ISREG(info.st_mode) and info.st_size > 0


def _download_artwork_html(url: str, path: Path) -> None:
    html = request_retries(url).text
    if not html.strip():
        raise ValueError('Empty artwork page')
    # Publish only complete gzip files. An interrupted write is not a cache hit.
    contents = gzip.compress(html.encode('utf-8'), compresslevel=3)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            file.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_artworks_raw_html(
    artworks_index: list[dict], target_html_dir='artworks_raw_html',
    num_workers=DEFAULT_ARTWORK_DOWNLOAD_WORKERS, root_data_dir='data',
) -> list[dict]:
    if not artworks_index:
        logger.warning('Empty artworks index, nothing to download')
        return []
    logger.info('URL example: %s', artworks_index[0]['artwork_url'])
    failed_urls = set()
    urls = list(dict.fromkeys(item['artwork_url'] for item in artworks_index))
    (Path(root_data_dir) / target_html_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    missing = []
    for url in urls:
        plain = Path(_build_html_path(url, target_html_dir, root_data_dir))
        compressed = plain.with_suffix('.html.gz')
        if _nonempty_cache_file(compressed):
            paths[url] = compressed
        elif _nonempty_cache_file(plain):
            paths[url] = plain
        else:
            paths[url] = compressed
            missing.append(url)
    logger.info('Reusing %d cached artwork pages; downloading %d with %d threads (gzip)',
                len(urls) - len(missing), len(missing), num_workers)
    for url, exc in _download_failures(
        missing, lambda url: _download_artwork_html(url, paths[url]),
        num_workers=num_workers, label='artworks',
    ):
        logger.error('Failed to download %s: %s', url, exc)
        failed_urls.add(url)
    artworks_index = [item for item in artworks_index if item['artwork_url'] not in failed_urls]
    for item in artworks_index:
        item['local_html_path'] = str(paths[item['artwork_url']])
    return artworks_index

def prepare_artists_index(input_csv_path, lists_html_dir='art_links_lists', num_workers=4, root_data_dir='data') -> list[dict]:
    import pandas as pd

    rows = pd.read_csv(input_csv_path).to_dict(orient='records')
    logger.info('Downloading artworks list pages for %d artists with %d threads', len(rows), num_workers)
    listings = [(row, f"{row['artist_url']}/all-works/text-list") for row in rows]
    for (row, _), exc in _download_failures(
        listings, lambda listing: get_html(listing[1], lists_html_dir, root_data_dir),
        num_workers=num_workers, label='artist listings',
    ):
        logger.error('Failed to download artworks list for %s: %s', row['artist_name'], exc)
    artists_index = []
    for row, list_url in listings:
        artist_artworks_source = _build_html_path(list_url, lists_html_dir, root_data_dir)
        artists_index.append({
            'artist_url': row['artist_url'],
            'artist_artworks_source': artist_artworks_source,
        })
    logger.info('Prepared artists index: %d items', len(artists_index))
    return artists_index


def process_art_movement(raw_str):
    if not isinstance(raw_str, str):
        return ''
    tags = []
    # Split on commas not inside parentheses
    items = re.split(r',\s*(?![^()]*\))', raw_str)
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Extract main name and parenthesized aliases as separate tags
        parts = re.split(r'\(([^)]+)\)', item)
        for part in parts:
            for sub in part.split(','):
                sub = sub.strip()
                if sub:
                    tags.append(sub)
    # Deduplicate while preserving order
    tags = list(dict.fromkeys(tags))
    return '|'.join(tags)

def greedy_tag_split(raw_tag_description: str, tags_df: pd.DataFrame) -> str:
    """greedy_tag_split('post-impressionism socialist realism')"""
    import pandas as pd

    input_str = raw_tag_description
    res = (
        pd.DataFrame(n_gram_split(input_str), columns=['tag'])
        .merge(tags_df[~tags_df['tag'].isin([input_str])], how='inner', on='tag')
        .sort_values(by='size', ascending=False)  # sort values for greedy matching algorithms
        ['tag']
        .values
    )
    res_tags = []
    for i in res:
        if i in input_str and len(i) < 20: # 27
            res_tags.append(i)
            input_str = input_str.replace(i, '')
        if ' '.join(res_tags) == input_str:
            break
    if len(res_tags) == 0:
        res_tags.append(input_str)
    return ','.join([' '.join(set(j.split(' '))) for j in res_tags])

def compute_tags(artists_df) -> pd.DataFrame:
    from itertools import chain

    import pandas as pd

    def art_movement_helper(raw_str: str):
        return [i.strip() for i in raw_str.split(',') if len(i.strip().lower()) > 0]

    artists_df['art movement'] = artists_df['art movement'].fillna('').apply(lambda x: x.lower())
    artists_df['art_movement_raw_tags'] = artists_df['art movement'].apply(art_movement_helper)
    flatten_genres = list(chain.from_iterable(artists_df['art_movement_raw_tags'].values))
    tags_df = (
        pd.DataFrame(flatten_genres,columns=['tag'])['tag']
        .value_counts()
        .reset_index(name='cnt')
    )
    # tags_df.rename(columns={'index': 'tag'}, inplace=True)
    print(tags_df.head())
    # some useful features
    tags_df['is_complex'] = tags_df['tag'].apply(lambda x: len(x.split(' ')) > 1)
    tags_df['size'] = tags_df['tag'].apply(lambda x: len(x))

    tags_df['splitted_tags'] =  tags_df['tag'].apply(lambda x: greedy_tag_split(x, tags_df))
    processed_tag_mapping = {i: j for i, j in tags_df[['tag', 'splitted_tags']].values}
    artists_df['art_tags'] = artists_df['art_movement_raw_tags'].apply(lambda x: ','.join([processed_tag_mapping[i] for i in x]))

    tags_df = (
        pd.DataFrame(list(chain.from_iterable(artists_df['art_tags'].apply(lambda x: x.split(',')).values)), columns=['tag'])
        .value_counts()
        .reset_index(name='cnt')
    )
    tags_df = tags_df[tags_df['tag'].apply(len) > 1]

    return tags_df

def merge_wikiart_data(
    input_jsonl_path: str,
    input_artists_info_csv_path: str,
    output_csv_path: str,
):
    import pandas as pd

    artists_info_df = pd.read_csv(input_artists_info_csv_path)
    cols_list = ['artist_name', 'artist_link', 'art movement', 'nationality', 'field', 'artist_pic', 'wikipedia', 'artist_url']
    images = pd.DataFrame(utils.read_jsonl(input_jsonl_path), columns=['artist_url', 'image_url'])
    images = images[images['image_url'].astype(bool)].drop_duplicates()
    images = images.groupby('artist_url')['image_url'].agg(list).rename('image').reset_index()
    content_df = artists_info_df.reindex(columns=cols_list).merge(images, on='artist_url')
    logger.info('Num rows %d', content_df.shape[0])
    content_df.to_csv(output_csv_path, index=False, compression="gzip")
    print('Saved to %s, num rows: %d' % (output_csv_path, content_df.shape[0]))

def compute_tags_and_save(input_csv_path: str, output_tags_csv_path: str):
    import pandas as pd

    content_df = pd.read_csv(input_csv_path, compression="gzip")
    tags_df = compute_tags(content_df).query('cnt > 1')
    logger.info('Num tags %d', tags_df.shape[0])
    tags_df.to_csv(output_tags_csv_path, index=False, compression="gzip")
    print('Saved to %s, num rows: %d' % (output_tags_csv_path, tags_df.shape[0]))


def collect_artwork_urls(
    artists_csv: Path, output_jsonl: Path, data_dir: Path, limit: int | None, *,
    artwork_download_workers: int = DEFAULT_ARTWORK_DOWNLOAD_WORKERS,
) -> None:
    artists = prepare_artists_index(artists_csv, root_data_dir=data_dir)
    artworks = prepare_artworks_index(artists, limit=limit)
    artworks = get_artworks_raw_html(artworks, num_workers=artwork_download_workers, root_data_dir=data_dir)

    for artwork in artworks:
        html = _read_artwork_html(artwork['local_html_path'])
        soup = BeautifulSoup(html, 'html.parser')
        metadata, wrapper = artwork_image_sources(soup)
        src = (metadata if metadata is not None else wrapper or '').strip()
        artwork['image_url'] = urljoin(artwork['artwork_url'], src) if src else ''

    utils.write_jsonl(artworks, output_jsonl)
    logger.info('Saved %d artwork records (%d with image URLs) to %s',
                len(artworks), sum(bool(row['image_url']) for row in artworks), output_jsonl)


def pipeline(
    data_dir: str | Path | None = None, *, artworks_per_artist: int | None = None,
    artwork_download_workers: int = DEFAULT_ARTWORK_DOWNLOAD_WORKERS,
) -> Path:
    """Run the six WikiArt stages in order and return the quality report path."""
    if artworks_per_artist is not None and artworks_per_artist < 1:
        raise ValueError('artworks_per_artist must be positive')
    if artwork_download_workers < 1:
        raise ValueError('artwork_download_workers must be positive')
    data_dir = Path(data_dir if data_dir is not None else os.environ.get('DATA_DIR', 'data'))
    data_dir.mkdir(parents=True, exist_ok=True)
    artist_pages = data_dir / '06_artists_pages.csv'
    artist_info = data_dir / '06_artists_info.csv'
    artist_bios = data_dir / '06_artists_wiki_texts.csv'
    artwork_urls = data_dir / '06_artists_artworks.jsonl'
    content_db = data_dir / '06_content_db.csv.gz'
    tags_db = data_dir / '06_tags_db.csv.gz'
    raw_html = data_dir / 'artworks_raw_html'
    descriptions = data_dir / 'artworks_json'
    stats_path = data_dir / 'artworks_json_descriptive_stats.json'

    logger.info('1/6 Collect artist pages')
    get_artists_pages(artist_pages)
    logger.info('2/6 Collect artist info')
    get_artists_info(artist_pages, artist_info, artist_bios, root_data_dir=data_dir)
    logger.info('3/6 Collect artwork URLs')
    collect_artwork_urls(
        artist_pages, artwork_urls, data_dir, artworks_per_artist,
        artwork_download_workers=artwork_download_workers,
    )
    logger.info('4/6 Merge dataset')
    merge_wikiart_data(artwork_urls, artist_info, content_db)
    compute_tags_and_save(content_db, tags_db)
    logger.info('5/6 Extract artwork descriptions')
    extract_artworks_descriptions(raw_html, descriptions, num_workers=2)
    logger.info('6/6 Build descriptive stats')
    build_artworks_json_descriptive_stats(descriptions, stats_path)
    return stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Output and cache directory (default: $DATA_DIR or data).')
    parser.add_argument('--artworks-per-artist', type=int, help='Optional positive limit per artist.')
    parser.add_argument(
        '--artwork-download-workers', type=int, default=DEFAULT_ARTWORK_DOWNLOAD_WORKERS,
        help='Number of threads for downloading artwork pages (default: %(default)s).',
    )
    args = parser.parse_args()
    if args.artworks_per_artist is not None and args.artworks_per_artist < 1:
        parser.error('--artworks-per-artist must be positive')
    if args.artwork_download_workers < 1:
        parser.error('--artwork-download-workers must be positive')
    pipeline(
        args.data_dir, artworks_per_artist=args.artworks_per_artist,
        artwork_download_workers=args.artwork_download_workers,
    )


if __name__ == '__main__':
    main()
