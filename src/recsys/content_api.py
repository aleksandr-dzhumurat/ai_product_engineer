import ast
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __package__ == 'src.recsys':
    from ..scraping.utils import logger
else:  # Also support PYTHONPATH=src uvicorn recsys.content_api:app.
    from scraping.utils import logger


def _artist_tags(row: dict) -> str:
    movements = [tag.strip().lower() for tag in row['artist_movement'].split('|') if tag.strip()]
    fields = [tag.strip().lower() for tag in row['field'].split(',') if tag.strip()]
    return '|'.join(movements * 3 + fields)


def user_tags_ranking(user_actions: dict, all_tags_df: pd.DataFrame):
    LIKE = 'like'
    DISLIKE = 'dislike'
    tags_df = all_tags_df.copy()
    if len(user_actions) > 0:
        user_negative_tags = pd.json_normalize([i for i in user_actions if i['action']==DISLIKE])
        if user_negative_tags.shape[0] > 0:
            user_negative_tags = user_negative_tags['content_tag'].value_counts().to_frame(name='cnt').reset_index()
            user_negative_tags.columns = ['content_tag', 'cnt']
        user_positive_tags = pd.json_normalize([i for i in user_actions if i['action']==LIKE])
        if user_positive_tags.shape[0] > 0:
            user_positive_tags = user_positive_tags['content_tag'].value_counts().to_frame(name='cnt').reset_index()
            user_positive_tags.columns = ['content_tag', 'cnt']
        if user_negative_tags.shape[0] > 0:  # drop disliked tags
            tags_df = (
                tags_df
                .merge(user_negative_tags, how='left', left_on='tag', right_on='content_tag',suffixes=('','_neg'))
                .query('cnt_neg.isnull()')
                [['tag', 'cnt']]
            )
        if user_positive_tags.shape[0] > 0:
            tags_df = (
                tags_df
                .merge(user_positive_tags, how='left', left_on='tag', right_on='content_tag',suffixes=('','_pos'))
                .sort_values('cnt_pos', ascending=False)
                [['tag', 'cnt', 'cnt_pos']]
            ).head(3)  # add
    return tags_df

class ArtistsDB:
    def __init__(self):
        self.df = pd.DataFrame()
        self.tags_df = pd.DataFrame()
        self.images_by_artist: dict[str, list[str]] = {}
        self.embedder = None
        self.corpus_numpy = None

    def init_db(self, path_to_csv: str | Path):
        self.df = pd.read_csv(path_to_csv, dtype=str, keep_default_na=False)
        required = {'artist_name', 'artist_url'}
        if self.df.empty or not required.issubset(self.df.columns):
            raise ValueError(f'{path_to_csv}: expected a non-empty artist table with {sorted(required)}')
        for column in ('artist_movement', 'artist_field', 'field'):
            if column not in self.df:
                self.df[column] = ''

        artists_tags = [_artist_tags(row) for row in self.df.to_dict(orient='records')]
        if any(artists_tags):
            self.embedder = TfidfVectorizer(analyzer=lambda value: value.split('|') if value else [])
            self.corpus_numpy = self.embedder.fit_transform(artists_tags)
        logger.info('Loaded %d artists', len(self.df))
        return self
    
    def get_content(self, content_id: int) -> dict:
        content_info = self.df.iloc[content_id].to_dict()
        return {
            'artist_name': content_info['artist_name'],
            'artist_url': content_info['artist_url'],
            'artist_pic': content_info.get('artist_pic', ''),
            'nationality': content_info.get('nationality', ''),
            'wikipedia': content_info.get('wikipedia', ''),
            'artist_movement': content_info['artist_movement'],
            'field': content_info['artist_field'],
            'images': self.images_by_artist.get(content_info['artist_url'], []),
        }
    
    def recommend(self, content_id: int, num_recs: int = 10) -> list[dict]:
        if self.embedder is None:
            return []
        query_embed = self.corpus_numpy[content_id]
        similarities = cosine_similarity(query_embed, self.corpus_numpy).flatten()
        rec_ids = np.argsort(similarities)[::-1]
        rec_ids = rec_ids[rec_ids != content_id][:num_recs]
        recs_list = []
        for idx in rec_ids:
            row = self.df.iloc[idx]
            recs_list.append({
                'artist_name': row['artist_name'],
                'artist_url': row['artist_url'],
                'score': float(similarities[idx]),
            })
        return recs_list
    
    def sample_content(self, user_actions: list[dict[str, str]], eps: float = 0.3) -> dict:
        if self.embedder is None:
            return {'item_id': int(self.df.sample(1).index[0]), 'item_tag': ''}
        all_tags = list(self.embedder.vocabulary_.keys())
        if np.random.random() < eps or len(user_actions) == 0:
            random_tag = np.random.choice(all_tags)
        else:
            positive_tags = [a['content_tag'] for a in user_actions if a['action'] == 'like']
            random_tag = np.random.choice(positive_tags) if positive_tags else np.random.choice(all_tags)
        logger.info('random tag: %s', random_tag)
        matches = self.df[
            self.df['artist_movement'].fillna('')
            .apply(lambda x: random_tag in x.lower())
        ]
        if matches.empty:
            res = int(self.df.sample(1).index[0])
        else:
            res = int(np.random.choice(matches.index))
        return {'item_id': res, 'item_tag': random_tag}

def prepare_service_data(data_dir: str | Path | None = None) -> ArtistsDB:
    """Load WikiArt's stage 2/4 artifacts without copying or rewriting them."""
    root = Path(data_dir if data_dir is not None else os.getenv('WIKIART_DATA_DIR', 'data/artworks'))
    artists, content, tags = (root / name for name in (
        '06_artists_info.csv', '06_content_db.csv.gz', '06_tags_db.csv.gz',
    ))
    missing = [str(path) for path in (artists, content, tags) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'WikiArt artifacts are missing; complete pipeline stage 4: {", ".join(missing)}')

    db = ArtistsDB().init_db(artists)
    db.tags_df = pd.read_csv(tags, usecols=['tag', 'cnt'])
    images = pd.read_csv(content, usecols=['artist_url', 'image'], dtype=str, keep_default_na=False)
    for artist_url, value in images[['artist_url', 'image']].itertuples(index=False, name=None):
        try:
            urls = ast.literal_eval(value) if value else []  # pandas writes Python list literals to CSV.
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f'{content}: invalid image list for {artist_url}') from exc
        if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
            raise ValueError(f'{content}: expected a list of image URLs for {artist_url}')
        db.images_by_artist[artist_url] = urls
    return db


class InMemoryStorage:
    def __init__(self):
        self._users = {}  # user_id -> {'name': str, 'actions': list}
        self._next_id = 1

    def find_one(self, name: str) -> dict | None:
        for user_id, user in self._users.items():
            if user['name'] == name:
                return {'_id': user_id, **user}
        return None

    def insert_one(self, name: str) -> str:
        user_id = str(self._next_id)
        self._next_id += 1
        self._users[user_id] = {'name': name, 'actions': []}
        return user_id

    def push_action(self, user_id: str, action: dict):
        self._users[user_id]['actions'].append(action)

    def get_actions(self, user_id: str) -> list[dict]:
        return self._users[user_id]['actions']


class UserDB:
    def __init__(self):
        self.storage = InMemoryStorage()

    def get_user_actions(self, user_name: str) -> tuple[str | None, list[dict] | None]:
        user = self.storage.find_one(user_name)
        if user is None:
            return None, None
        return str(user['_id']), user['actions']

    def create_user(self, user_name: str) -> str:
        user_id, _ = self.get_user_actions(user_name)
        if user_id is None:
            user_id = self.storage.insert_one(user_name)
        else:
            logger.info('User already exists')
        return user_id

    def push_action(self, user_id: str, content_id: str, content_tag: str, action_type: str) -> bool:
        action = {'content_id': content_id, 'content_tag': content_tag, 'action': action_type}
        self.storage.push_action(user_id, action)
        return True


class Action(BaseModel):
    user_token: str
    user_name: str
    item_id: int
    item_tag: str
    action_type: str

class User(BaseModel):
    user_name: str

users_db = UserDB()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artists_db = prepare_service_data()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def get_item(item_id: int, request: Request):
    item = request.app.state.artists_db.get_content(item_id)
    return {"item": item}

@app.post("/action")
async def action(action: Action):
    users_db.push_action(action.user_token, str(action.item_id), action.item_tag, action.action_type)
    return {"result": "ok"}

@app.post("/auth")
async def auth(user: User):
    bearer_token = users_db.create_user(user.user_name)
    return {"Bearer": bearer_token}

@app.get("/recommend/{item_id}")
async def recommend(item_id: int, request: Request):
    recs_list = request.app.state.artists_db.recommend(item_id)
    return {"recs": recs_list}

@app.post("/sample")
async def sample(user: User, request: Request):
    _, user_actions = users_db.get_user_actions(user.user_name)
    result = request.app.state.artists_db.sample_content(user_actions or [])
    return result


@app.get("/health")
async def health(request: Request):
    # FastAPI only serves requests after lifespan has loaded the catalog.
    return {
        "status": "ok",
        "service": "artworks",
        "artists": len(request.app.state.artists_db.df),
    }
