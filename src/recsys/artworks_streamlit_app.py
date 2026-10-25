"""ArtSwipe: a compact gallery for exploring the WikiArt recommender."""

import os
from html import escape
from urllib.parse import urlsplit

import requests
import streamlit as st


API_URL = os.getenv('ARTWORKS_API_URL', 'http://127.0.0.1:8000').rstrip('/')

STYLES = """
<style>
[data-testid="stAppViewContainer"] { background: #f8f7f3; color: #252923; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 1240px; padding: 2rem 2.5rem 3rem;
}
h1, h2, h3 { color: #252923; }
h1, h2 { font-family: Georgia, 'Times New Roman', serif !important; }
h1 { font-weight: 400 !important; letter-spacing: -0.035em; }
h2 { padding-top: 0 !important; padding-bottom: 0.3rem !important; }
hr { margin-top: 0.65rem !important; margin-bottom: 0.65rem !important; }
.art-brand { font-size: 0.95rem; font-weight: 700; letter-spacing: 0.19em; }
.art-kicker { color: #6a7365; font-size: 0.75rem; letter-spacing: 0.12em;
    text-transform: uppercase; margin: 0 0 0.25rem; }
.art-intro { font-size: 1.1rem; color: #66705f; margin-bottom: 0.5rem; }
.art-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.2rem 0 0.7rem; }
.art-tag { padding: 0.22rem 0.65rem; border: 1px solid #d9dfd2;
    border-radius: 20px; color: #42513c; font-size: 0.8rem; background: #eef1e8; }
.art-placeholder { min-height: 240px; display: flex; align-items: center;
    justify-content: center; background: #eeede7; color: #707769; border-radius: 8px; }
.st-key-artworks_gallery [data-testid="stImage"] { width: 100%; }
.st-key-artworks_gallery [data-testid="stImage"] img {
    width: 100%; height: 260px; object-fit: contain; background: #eeede7;
    border-radius: 8px; padding: 8px; box-sizing: border-box;
}
.st-key-artist_portrait [data-testid="stImage"] img {
    width: 80px; height: 80px; object-fit: cover; border-radius: 50%;
}
[data-testid="stButton"] button, [data-testid="stLinkButton"] a,
[data-testid="stFormSubmitButton"] button { border-radius: 8px; }
[data-testid="stButton"] button[kind^="primary"],
[data-testid="stFormSubmitButton"] button[kind^="primary"] {
    background: #304d3e; border-color: #304d3e; color: white;
}
[data-testid="stButton"] button[kind^="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind^="primary"]:hover {
    background: #233d2f; border-color: #233d2f; color: white;
}
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #dfe3d9; }
@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 2rem; }
    .st-key-artworks_gallery [data-testid="stImage"] img { height: 240px; }
}
</style>
"""


def api_request(method, path, **kwargs):
    try:
        response = requests.request(method, f'{API_URL}{path}', timeout=10, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        st.error('The art catalog is temporarily unavailable. Please try again in a moment.')
        st.stop()


def web_url(value):
    """Only expose navigable web URLs from catalog metadata."""
    value = str(value or '').strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ''
    return value if parsed.scheme in ('http', 'https') and parsed.netloc else ''


def next_artist():
    st.session_state.sample = api_request(
        'POST', '/sample', json={'user_name': st.session_state.user_name},
    )


def log_out():
    for key in ('user_token', 'user_name', 'sample', 'likes', 'dislikes'):
        st.session_state.pop(key, None)


def record_action(action):
    state = st.session_state
    api_request('POST', '/action', json={
        'user_token': state.user_token,
        'user_name': state.user_name,
        'item_id': state.sample['item_id'],
        'item_tag': state.sample['item_tag'],
        'action_type': action,
    })
    counter = 'likes' if action == 'like' else 'dislikes'
    state[counter] = state.get(counter, 0) + 1
    # Keep a recorded action from being accidentally submitted again while the
    # next sample loads. A failed sample request can be retried on the next run.
    state.pop('sample', None)
    st.toast('Preference saved. Finding your next artist…')
    st.rerun()


def render_login():
    intro, login = st.columns([1.4, 1], gap='large')
    with intro:
        st.markdown('<p class="art-kicker">Your own art discovery</p>', unsafe_allow_html=True)
        st.title('Find the art that moves you.')
        st.markdown(
            '<p class="art-intro">Explore artists, take a closer look at their work, '
            'and tell us what catches your eye.</p>', unsafe_allow_html=True,
        )
        st.markdown('**Explore** a selection of artworks.  \n'
                    '**React** to the artist, not an individual painting.  \n'
                    '**Discover** more artists with similar styles and fields.')
    with login:
        with st.container(border=True):
            st.subheader('Start exploring')
            st.caption('Choose a name for your demo profile. No password needed.')
            with st.form('login'):
                name = st.text_input('Your name', placeholder='e.g. Alex', max_chars=80).strip()
                submit = st.form_submit_button('Start exploring', type='primary', width='stretch')
            if submit and name:
                state = st.session_state
                state.user_token = api_request('POST', '/auth', json={'user_name': name})['Bearer']
                state.user_name = name
                st.rerun()
            if submit:
                st.warning('Enter a name to start exploring.')
            st.caption('This is a teaching demo. A name identifies a profile; it is not secure sign-in.')


def render_artist(artist):
    portrait = web_url(artist.get('artist_pic'))
    profile, navigation = st.columns([4, 1], vertical_alignment='center')
    with profile:
        if portrait:
            avatar, description = st.columns([1, 8], vertical_alignment='center')
            with avatar:
                with st.container(key='artist_portrait'):
                    st.image(portrait, width=80)
        else:
            description = st.container()
        with description:
            st.markdown('<p class="art-kicker">Meet the artist</p>', unsafe_allow_html=True)
            st.header(artist['artist_name'])
    with navigation:
        if st.button('Next artist →', width='stretch', help='Skip without recording a preference.'):
            next_artist()
            st.rerun()

    metadata = [
        (label, str(artist.get(key) or '').strip().replace('|', ' · '))
        for label, key in [('Movement', 'artist_movement'), ('Field', 'field'), ('Nationality', 'nationality')]
    ]
    tags = ''.join(
        f'<span class="art-tag">{escape(label)}: {escape(value)}</span>'
        for label, value in metadata if value
    )
    if tags:
        st.markdown(f'<div class="art-tags">{tags}</div>', unsafe_allow_html=True)

    images = list(dict.fromkeys(url for url in (web_url(v) for v in artist.get('images', [])) if url))
    st.caption(f'A selection of {min(len(images), 4)} of {len(images)} works' if images else 'Artist preview')
    with st.container(key='artworks_gallery'):
        if images:
            for i, (column, url) in enumerate(zip(st.columns(min(len(images), 4)), images[:4]), 1):
                column.image(url, caption=f'Work {i}', width='stretch')
        else:
            st.markdown('<div class="art-placeholder">No artwork previews available for this artist.</div>',
                        unsafe_allow_html=True)

    like, dislike, links = st.columns([1, 1, 2], vertical_alignment='center')
    if like.button('Like & next', type='primary', width='stretch', help='Save a like for this artist.'):
        record_action('like')
    if dislike.button('Not for me', width='stretch', help='Save a dislike and discover another artist.'):
        record_action('dislike')
    with links:
        wikiart, wikipedia = st.columns(2)
        if url := web_url(artist.get('artist_url')):
            wikiart.link_button('View on WikiArt ↗', url, width='stretch')
        wiki = str(artist.get('wikipedia') or '').strip()
        if wiki and '://' not in wiki:
            wiki = 'https://' + wiki.lstrip('/')
        if url := web_url(wiki):
            wikipedia.link_button('Wikipedia ↗', url, width='stretch')
    st.caption('Your reaction is saved for the artist. “Next artist” skips without voting.')


def render_recommendations(artist_id):
    st.divider()
    st.subheader('Continue the discovery')
    st.caption('Similar artists, based on shared movements and fields. Similarity is not a probability.')
    recs = api_request('GET', f'/recommend/{artist_id}')['recs']
    if not recs:
        st.info('No similar artists yet. Try another artist to keep exploring.')
        return

    def cards(items):
        for start in range(0, len(items), 3):
            for column, rec in zip(st.columns(3), items[start:start + 3]):
                with column, st.container(border=True):
                    st.markdown(f"**{escape(str(rec['artist_name']))}**")
                    st.caption(f"Similarity · {rec['score']:.2f}")
                    if url := web_url(rec.get('artist_url')):
                        st.link_button('Explore on WikiArt ↗', url, width='stretch')

    cards(recs[:6])
    if len(recs) > 6:
        with st.expander(f'See {len(recs) - 6} more artists'):
            cards(recs[6:])


def main():
    st.set_page_config(page_title='ArtSwipe · Discover artists', page_icon='🎨', layout='wide')
    st.markdown(STYLES, unsafe_allow_html=True)
    state = st.session_state
    brand, account = st.columns([3, 2], vertical_alignment='center')
    with brand:
        st.markdown('<div class="art-brand">ARTSWIPE</div>', unsafe_allow_html=True)
        st.caption('A little curiosity. A new favorite artist.')
    if 'user_token' in state:
        with account:
            user, logout = st.columns([3, 1], vertical_alignment='center')
            user.caption(f"{state.user_name} · Likes this session: {state.get('likes', 0)} · Passes: {state.get('dislikes', 0)}")
            logout.button('Log out', on_click=log_out, width='stretch')
    st.divider()
    if 'user_token' not in state:
        render_login()
        return
    if 'sample' not in state:
        with st.spinner('Finding an artist for you…'):
            next_artist()
    artist_id = state.sample['item_id']
    artist = api_request('GET', f'/items/{artist_id}')['item']
    render_artist(artist)
    render_recommendations(artist_id)


if __name__ == '__main__':
    main()
