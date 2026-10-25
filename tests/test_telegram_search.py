"""Run: python -m unittest discover -s tests -v (repository dependencies required)."""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import requests
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
# Match the /srv/ml_tools bind mount used by Docker Compose.
if 'ml_tools' not in sys.modules:
    package = types.ModuleType('ml_tools')
    package.__path__ = [str(ROOT / 'src')]
    sys.modules['ml_tools'] = package


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = load_module('search_api', 'dockerfiles/api/src/main.py')
bot = load_module('search_bot', 'dockerfiles/tg_backend/src/app.py')


class SearchApiTests(unittest.TestCase):
    def setUp(self):
        self.context = TestClient(api.app)
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_defaults_normalization_and_result_contract(self):
        upstream = {'hits': {'hits': [{'_source': {'content': 'x' * 1500, 'asin': 'A'}}]}}
        with patch.object(api, 'search', return_value=upstream) as search:
            response = self.client.post('/search', json={'text': ' cough '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{'content': 'x' * 1000, 'asin': 'A'}])
        self.assertEqual(search.call_args.args[1], 'cough')
        self.assertEqual(search.call_args.kwargs['limit'], 5)
        self.assertTrue(search.call_args.kwargs['raise_on_error'])

    def test_validation(self):
        for payload in ({'text': '   '}, {'text': 'x', 'num': 0},
                        {'text': 'x', 'num': 21}, {}, {'text': 'x' * 2001}):
            with self.subTest(payload=repr(payload)[:60]):
                self.assertEqual(self.client.post('/search', json=payload).status_code, 422)

    def test_empty_and_broken_upstream(self):
        with patch.object(api, 'search', return_value={'hits': {'hits': []}}):
            self.assertEqual(self.client.post('/search', json={'text': 'x'}).json(), [])
        for result in (None, {}, {'hits': {'hits': 'invalid'}},
                       {'hits': {'hits': [{'_source': {'content': 123, 'asin': 'A'}}]}}):
            with patch.object(api, 'search', return_value=result):
                self.assertEqual(self.client.post('/search', json={'text': 'x'}).status_code, 502)

    def test_upstream_failures(self):
        errors = [(requests.ConnectionError(), 503), (requests.Timeout(), 503),
                  (ValueError('bad json'), 502)]
        for upstream_status, expected in ((401, 502), (404, 502), (500, 503)):
            response = requests.Response()
            response.status_code = upstream_status
            errors.append((requests.HTTPError(response=response), expected))
        for error, expected in errors:
            with patch.object(api, 'search', side_effect=error):
                self.assertEqual(self.client.post('/search', json={'text': 'x'}).status_code, expected)

    def test_other_practicals_still_available(self):
        self.assertEqual(self.client.get('/ping').json(), {'message': 'pong'})
        model = SimpleNamespace(predict=lambda x: [2], predict_proba=lambda x: [[0, 0, 1]])
        api.app.state.classifier_model = model
        response = self.client.get('/classifier/', params={'x1': 1, 'x2': -2.2, 'x3': 1.05})
        self.assertEqual(response.json()['prediction'], 2)
        self.assertEqual(self.client.post('/predict', json={
            'request_context_device_type': 'phone', 'dsp': 'a', 'ssp': 'b',
            'hour': 12, 'price': 0.5,
        }).status_code, 501)


class BotTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_to_api_to_search(self):
        async with api.lifespan(api.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://api:8000') as client:
                message = SimpleNamespace(text='cough', reply_text=AsyncMock())
                update = SimpleNamespace(effective_message=message)
                context = SimpleNamespace(application=SimpleNamespace(bot_data={'search_client': client}))
                hits = {'hits': {'hits': [{'_source': {'content': '<b>review</b>', 'asin': 'A'}}]}}
                with patch.object(api, 'search', return_value=hits):
                    await bot.bot_dialog(update, context)
                self.assertIn('<b>review</b>', message.reply_text.call_args.args[0])
                self.assertIsNone(message.reply_text.call_args.kwargs['parse_mode'])
                message.reply_text.reset_mock()
                with patch.object(api, 'search', return_value={'hits': {'hits': []}}):
                    await bot.bot_dialog(update, context)
                self.assertIn('Ничего не найдено', message.reply_text.call_args.args[0])
                message.reply_text.reset_mock()
                with patch.object(api, 'search', side_effect=requests.Timeout()):
                    await bot.bot_dialog(update, context)
                self.assertIn('временно недоступен', message.reply_text.call_args.args[0])

    async def test_invalid_json_contract(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'wrong': []}))) as client:
            with self.assertRaises(ValueError):
                await bot.dialog_router('cough', client)

    async def test_utf16_splitting(self):
        text = '😀<b>&' * 2000
        chunks = bot.split_messages(text)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(chunk.encode('utf-16-le')) // 2 <= 3500 for chunk in chunks))


if __name__ == '__main__':
    unittest.main()
