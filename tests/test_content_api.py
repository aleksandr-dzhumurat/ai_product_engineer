import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.recsys.content_api import app, prepare_service_data
from src.scraping import scrape_wikiart, utils


class ContentAPITest(unittest.TestCase):
    def test_startup_loads_scraper_artifacts_and_preserves_artist_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artist_urls = ['https://www.wikiart.org/en/z', 'https://www.wikiart.org/en/a']
            image = 'https://uploads.wikiart.org/painting.jpg'
            artists = root / '06_artists_info.csv'
            content = root / '06_content_db.csv.gz'
            tags = root / '06_tags_db.csv.gz'
            urls = root / '06_artists_artworks.jsonl'

            pd.DataFrame({
                'artist_name': ['Z', 'A'], 'artist_url': artist_urls,
                'art movement': ['Realism', 'Realism'], 'artist_movement': ['Realism', 'Realism'],
                'artist_field': ['painting', 'painting'], 'field': ['painting', 'painting'],
                'nationality': ['French', ''],
            }).to_csv(artists, index=False)
            utils.write_jsonl([{'artist_url': artist_urls[0], 'image_url': image}], urls)
            scrape_wikiart.merge_wikiart_data(urls, artists, content)
            scrape_wikiart.compute_tags_and_save(content, tags)

            with patch.dict(os.environ, {'WIKIART_DATA_DIR': directory}), TestClient(app) as client:
                health = client.get('/health')
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json(), {
                    'status': 'ok', 'service': 'artworks', 'artists': 2,
                })
                self.assertEqual(client.get('/items/0').json()['item'], {
                    'artist_name': 'Z', 'artist_url': artist_urls[0], 'artist_pic': '',
                    'nationality': 'French', 'wikipedia': '',
                    'artist_movement': 'Realism', 'field': 'painting', 'images': [image],
                })
                self.assertEqual(client.get('/items/1').json()['item']['images'], [])
                recs = client.get('/recommend/0').json()['recs']
                self.assertEqual([row['artist_name'] for row in recs], ['A'])
                response = client.post('/sample', json={'user_name': 'new user'})
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.json()['item_id'], (0, 1))

    def test_missing_artifacts_explain_how_to_proceed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'complete pipeline stage 4: .*06_content_db'):
                prepare_service_data(directory)
