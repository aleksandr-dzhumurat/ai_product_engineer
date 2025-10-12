import hashlib
import os
import time
from dataclasses import dataclass

import backoff
import openai
from openai import OpenAI


def get_openai_client(openai_key = None):
    if openai_key is None:
       openai_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(
        api_key=openai_key
    )
    return client

@dataclass
class GenAIResponse:
    text: str
    total_tokens: str
    model: str
    generation_id: str


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

def recs_generation(candidates):
    candidates = '\n'.join(["item_id: %s; review: %s" % (i['asin'], i['content']) for i in candidates])
    prompt = promt_generation(candidates)
    generated_result = generate(user_prompt=prompt)
    return generated_result['recs']
