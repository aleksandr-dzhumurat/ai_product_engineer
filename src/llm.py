import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import backoff
import openai
from dotenv import load_dotenv
from google.genai import Client, types
from openai import OpenAI

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
g_client = None

def get_openai_client(openai_key = None):
    if openai_key is None:
       openai_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(
        api_key=openai_key
    )
    return client

def get_gemini_client(env_path: str | Path):
    global g_client

    print(f'ENV loaded from {env_path}: {load_dotenv(env_path)}')
    if g_client is None:
        g_client = Client(api_key=os.environ["GEMINI_API_KEY"])
    return g_client

@dataclass
class GenAIResponse:
    text: str
    total_tokens: str
    model: str
    generation_id: str = None


def generate_id(seed=None, limit=None):
    if seed is None:
        seed = int(time.time())
    res = str(
        hashlib
        .md5(str(seed).encode('utf-8'))
        .hexdigest()
    )
    if limit is not None:
        res = res[:limit]
    return res

def promt_generation(candidates):
  # TODO: use jinja2
  promt = f"""
      Next rows below is an item_id reviews.
      {candidates}
      Utilize reviews to determine the best item_id.
      Avoid including actual reviews; rephrase them succinctly.
      Take in account number of reviews for same item_id and sentiment of review.
      Keep the recommendation under 50 words. Avoid starting with "Based on reviews"; opt for a more creative approach!
      Recommendation:
  """
  return promt

@backoff.on_exception(backoff.expo, openai.APIError)
@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo,openai.Timeout)
@backoff.on_exception(backoff.expo, RuntimeError)
def generate(client, user_prompt, system_prompt = "You are a helpful assistant for business analytics", model="gpt-3.5-turbo"):
    gpt_params = {
        "model": model,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.5,
        "frequency_penalty": 0.5,
    }
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    gpt_params.update({"messages": messages})

    response = client.chat.completions.create(**gpt_params)
    response = GenAIResponse(
        text=response.choices[0].message.content,
        total_tokens=response.usage.total_tokens,
        model=model,
        generation_id=generate_id(user_prompt)
    )

    return response

@backoff.on_exception(backoff.expo, RuntimeError)
def generate_gemini(api_client, user_prompt, system_prompt, model='gemini-2.5-flash-lite'):
    """model: gemini-1.5-flash, gemini-2.0-flash-001"""
    user_input = [
        types.Part(text=user_prompt),
    ]
    dialog_contents = [
        types.Content(
            role="user",
            parts=user_input
        )
    ]
    response = api_client.models.generate_content(
        model=model,
        contents=dialog_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    response = GenAIResponse(text=response.text, total_tokens=response.usage_metadata.total_token_count, model=model)
    return response

def recs_generation(candidates):
    candidates = '\n'.join(["item_id: %s; review: %s" % (i['asin'], i['content']) for i in candidates])
    prompt = promt_generation(candidates)
    generated_result = generate(user_prompt=prompt)
    return generated_result['recs']
