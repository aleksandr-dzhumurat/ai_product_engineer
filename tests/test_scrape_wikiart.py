import ast
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.scraping import scrape_wikiart, utils


class WikiArtPipelineTest(unittest.TestCase):
    def test_description_keeps_first_detail_and_normalizes_date_and_location(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / 'painting.html'
            html.write_text('''
                <div class="wiki-layout-artist-info wiki-layout-artwork-info">
                    <article>
                        <li><s>Genre:</s> Landscape</li>
                        <li><s>Genre:</s> Portrait</li>
                        <li><s>Date:</s> c. 1890</li>
                        <li><s>Location:</s> Gallery, Paris, France, Europe</li>
                    </article>
                </div>
            ''', encoding='utf-8')
            description = scrape_wikiart.extract_artwork_description(html)

        self.assertEqual(description['genre'], 'Landscape')
        self.assertEqual(description['date'], 'c.1890')
        self.assertEqual(description['style'], '')
        self.assertEqual(description['location'], {'venue': 'Gallery', 'city': 'Paris', 'country': 'France, Europe'})

    def test_stats_schema_for_empty_directory_and_counts_for_mixed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptions = root / 'descriptions'
            descriptions.mkdir()
            output = root / 'stats.json'
            build_stats = scrape_wikiart.build_artworks_json_descriptive_stats

            build_stats(descriptions, output)
            empty_report = json.loads(output.read_text())
            self.assertEqual(empty_report['total_json_files'], 0)
            self.assertEqual(empty_report['share_json_with_any_empty_description'], 0.0)
            self.assertEqual(empty_report['empty_field_counts'], dict.fromkeys(
                ('title', 'artist', 'date', 'style', 'genre', 'location_venue', 'location_city', 'location_country'), 0,
            ))

            (descriptions / 'a.json').write_text(json.dumps({'title': 'A', 'genre': ' landscape '}))
            (descriptions / 'b.json').write_text(json.dumps({
                'title': 'B', 'artist': 'Artist', 'date': '1890', 'style': 'Style', 'genre': 'portrait',
                'location': {'venue': 'Gallery', 'city': 'Paris', 'country': 'France'},
            }))
            (descriptions / 'c.json').write_text('{')
            with self.assertLogs(scrape_wikiart.logger, level='WARNING'):
                build_stats(descriptions, output)
            report = json.loads(output.read_text())

        self.assertEqual(report.keys(), empty_report.keys())
        self.assertEqual(report['total_json_files'], 3)
        self.assertEqual(report['invalid_json_files'], 1)
        self.assertEqual(report['json_with_any_empty_description'], 1)
        self.assertEqual(report['json_with_empty_location'], 1)
        self.assertEqual(report['share_json_with_empty_location'], 0.333333)
        self.assertEqual(report['empty_field_counts']['location_venue'], 1)
        self.assertEqual(report['empty_field_counts']['title'], 0)
        self.assertEqual(list(report['genre'].items()), [('landscape', 1), ('portrait', 1)])
        self.assertEqual(report['style'], {'Style': 1})

    def test_collect_artwork_urls_keeps_successes_when_one_download_fails(self):
        artist_url = f'{scrape_wikiart.BASE_URL}/en/example'
        listing_url = f'{artist_url}/all-works/text-list'
        pages = {
            listing_url: '''
                <ul class="painting-list-text">
                    <li class="painting-list-text-row"><a href="/en/example/fallback">Fallback</a></li>
                    <li class="painting-list-text-row"><a href="/en/example/no-image">No image</a></li>
                    <li class="painting-list-text-row"><a href="/en/example/failed">Failed</a></li>
                </ul>
            ''',
            f'{artist_url}/fallback': (
                '<div class="wiki-layout-artist-info wiki-layout-artwork-info"><h1>Fallback</h1></div>'
                '<div class="wiki-layout-artist-image-wrapper"><img src="/media/fallback.jpg"></div>'
            ),
            f'{artist_url}/no-image': '<h1>No image</h1>',
        }

        def fetch(url):
            if url not in pages:
                raise scrape_wikiart.requests.ConnectionError('test download failure')
            return type('Response', (), {'text': pages[url]})()

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            artists = data / 'artists.csv'
            artists.write_text(f'artist_name,artist_url\nExample,{artist_url}\n', encoding='utf-8')
            output = data / 'artworks.jsonl'

            with patch.object(scrape_wikiart, 'request_retries', side_effect=fetch) as request:
                with self.assertLogs(scrape_wikiart.logger, level='ERROR') as logs:
                    scrape_wikiart.collect_artwork_urls(artists, output, data, limit=None)

            self.assertEqual(sum(call.args == (listing_url,) for call in request.call_args_list), 1)
            self.assertIn('test download failure', '\n'.join(logs.output))
            self.assertTrue((data / 'art_links_lists' / 'example_text-list.html').is_file())
            fallback, missing = utils.read_jsonl(output)
            self.assertEqual(fallback['image_url'], f'{scrape_wikiart.BASE_URL}/media/fallback.jpg')
            self.assertEqual(missing['image_url'], '')
            self.assertTrue(fallback['local_html_path'].endswith('.html.gz'))
            description = scrape_wikiart.extract_artwork_description(fallback['local_html_path'])
            self.assertEqual(description['image'], '')
            self.assertEqual(description['image_url'], '/media/fallback.jpg')
            self.assertEqual(description['images'], [])

    def test_pipeline_merges_images_and_resumes_descriptions(self):
        base = scrape_wikiart.BASE_URL
        artist_url = f'{base}/en/example'
        image_url = 'https://uploads.wikiart.org/painting.jpg'
        alphabet_url = f'{base}/en/Alphabet/e/text-list'
        artwork_page = f'''
            <div class="wiki-layout-artist-info wiki-layout-artwork-info">
                <h1>Painting</h1><h2><a href="/en/example">Example</a></h2>
                <img itemprop="image" src="{image_url}">
                <article><li><s>Genre:</s> portrait</li></article>
            </div>
        '''
        pages = {
            alphabet_url: '<div class="masonry-text-view masonry-text-view-all"><li><a href="/en/example">Example</a></li></div>',
            artist_url: '''
                <div class="wiki-layout-artist-info">
                    <li><s>Nationality:</s> French</li><li><s>Art Movement:</s> Impressionism</li>
                </div>
                <div id="info-tab-wikipediaArticle" class="wiki-layout-artist-info-tab">An artist biography.</div>
            ''',
            f'{artist_url}/all-works/text-list': '''
                <ul class="painting-list-text">
                    <li class="painting-list-text-row"><a href="/en/example/painting">Painting</a></li>
                    <li class="painting-list-text-row"><a href="/en/example/study">Study</a></li>
                </ul>
            ''',
            f'{artist_url}/painting': artwork_page,
            f'{artist_url}/study': artwork_page.replace(image_url, '//uploads.wikiart.org/study.jpg'),
        }

        def fetch(url, *, timeout):
            response = scrape_wikiart.requests.Response()
            response.status_code = 200
            response._content = pages[url].encode('utf-8')
            return response

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw_html = data / 'artworks_raw_html'
            raw_html.mkdir()
            legacy = raw_html / 'example_study.html'
            legacy.write_text(pages[f'{artist_url}/study'], encoding='utf-8')
            descriptions = data / 'artworks_json'
            descriptions.mkdir()
            existing = descriptions / 'example_study.json'
            existing.write_text('{"title": "Existing description"}', encoding='utf-8')

            real_extract = scrape_wikiart.extract_artworks_descriptions

            def extract(raw_html, output_json, *, num_workers):
                self.assertEqual(num_workers, 2)
                self.assertTrue((data / '06_content_db.csv.gz').is_file())
                real_extract(raw_html, output_json, num_workers=num_workers)

            with (
                patch.object(scrape_wikiart, 'prepare_pages_list', return_value=[alphabet_url]),
                patch.object(scrape_wikiart.requests, 'get', side_effect=fetch) as request,
                patch.object(scrape_wikiart, 'extract_artworks_descriptions', side_effect=extract),
                patch.object(scrape_wikiart, 'get_artworks_raw_html', wraps=scrape_wikiart.get_artworks_raw_html) as download,
            ):
                report = scrape_wikiart.pipeline(data, artwork_download_workers=8)

            self.assertEqual(download.call_args.kwargs['num_workers'], 8)
            self.assertNotIn(f'{artist_url}/study', [call.args[0] for call in request.call_args_list])
            with patch.object(scrape_wikiart, 'request_retries') as request:
                resumed = scrape_wikiart.get_artworks_raw_html([
                    {'artwork_url': f'{artist_url}/painting'}, {'artwork_url': f'{artist_url}/study'},
                ], root_data_dir=data)
                request.assert_not_called()
            self.assertEqual([Path(item['local_html_path']).name for item in resumed],
                             ['example_painting.html.gz', 'example_study.html'])
            self.assertFalse((raw_html / 'example_painting.html').exists())

            with gzip.open(data / '06_content_db.csv.gz', 'rt', encoding='utf-8') as handle:
                row, = csv.DictReader(handle)
            self.assertEqual(row['nationality'], 'French')
            self.assertEqual(ast.literal_eval(row['image']), [image_url, 'https://uploads.wikiart.org/study.jpg'])
            self.assertIn('An artist biography.', (data / '06_artists_wiki_texts.csv').read_text())
            self.assertEqual(json.loads(existing.read_text())['title'], 'Existing description')
            painting = json.loads((descriptions / 'example_painting.json').read_text())
            self.assertEqual(painting['genre'], 'portrait')
            self.assertEqual(painting['image'], image_url)
            self.assertEqual(report, data / 'artworks_json_descriptive_stats.json')
            self.assertEqual(json.loads(report.read_text())['total_json_files'], 2)
