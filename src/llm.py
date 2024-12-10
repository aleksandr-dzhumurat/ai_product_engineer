import os
import time
import datetime
import hashlib

import backoff
import openai
from openai import OpenAI


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

def generate_id(seed=None):
    if seed is None:
        seed = int(time.time())
    res = str(
        hashlib
        .md5(str(seed).encode('utf-8'))
        .hexdigest()
    )[:8]
    return res

@backoff.on_exception(backoff.expo, openai.APIError)
@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo,openai.Timeout)
@backoff.on_exception(backoff.expo, RuntimeError)
def gpt_query(gpt_params, verbose: bool = False, avoid_fuckup: bool = False) -> dict:
    print('connecting OpenAI...')
    if verbose:
        print(gpt_params["messages"][1]["content"])
    response = client.chat.completions.create(
        **gpt_params
    )
    gpt_response = response.choices[0].message.content
    if avoid_fuckup:
        if '[' in gpt_response or '?' in gpt_response or '{' in gpt_response:
            raise RuntimeError
    res = {'recs': gpt_response}
    res.update({'completion_tokens': response.usage.completion_tokens, 'prompt_tokens': response.usage.prompt_tokens, 'total_tokens': response.usage.total_tokens})
    seed_phrase = f'{str(datetime.datetime.now().timestamp())}{gpt_response}'
    generation_id = generate_id(seed_phrase)
    res.update({'id': generation_id})
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

def generate(gpt_prompt, verbose=False):
    gpt_params = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 500,
        'temperature': 0.7,
        'top_p': 0.5,
        'frequency_penalty': 0.5,
    }
    if verbose:
        print(gpt_prompt)
    messages = [
        {
          "role": "system",
          "content": "You are a helpful assistant for business analytics",
        },
        {
          "role": "user",
          "content": gpt_prompt,
        },
    ]
    gpt_params.update({'messages': messages})
    res = gpt_query(gpt_params, verbose=False)
    return res

def recs_generation(candidates):
    candidates = '\n'.join(["item_id: %s; review: %s" % (i['asin'], i['content']) for i in candidates])
    prompt = promt_generation(candidates)
    generated_result = generate(prompt)
    return generated_result['recs']
