import openai
from openai import OpenAI
import numpy as np
import json
import time
import sys
import os
from openai import APIError, APIConnectionError, RateLimitError
# import google.generativeai as genai
# from anthropic import Anthropic


def get_openai_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])

def set_anthropic_key():
    pass

# def set_gemini_key():
#
#     # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
#     genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def set_openai_key():
    openai.api_key = os.environ['OPENAI_API_KEY']
    if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'].strip():
        openai.api_base = os.environ['OPENAI_API_BASE']


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            output = output.replace('json', '') # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


# def run_claude(query, max_new_tokens, model_name):
#
#     if model_name == 'claude-sonnet':
#         model_name = "claude-3-sonnet-20240229"
#     elif model_name == 'claude-haiku':
#         model_name = "claude-3-haiku-20240307"
#
#     client = Anthropic(
#     # This is the default and can be omitted
#     api_key=os.environ.get("ANTHROPIC_API_KEY"),
#     )
#     # print(query)
#     message = client.messages.create(
#         max_tokens=max_new_tokens,
#         messages=[
#             {
#                 "role": "user",
#                 "content": query,
#             }
#         ],
#         model=model_name,
#     )
#     print(message.content)
#     return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0):

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return None


def run_chatgpt(
    query,
    num_gen=1,
    num_tokens_request=1000,
    model="gpt-4o-mini",
    temperature=1.0
):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE")
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ],
        max_tokens=num_tokens_request,
        temperature=temperature,
        n=num_gen
    )

    return response.choices[0].message.content
    

def run_chatgpt_with_examples(
    query,
    examples,
    input,
    num_gen=1,
    num_tokens_request=1000,
    use_16k=False,
    wait_time=1,
    temperature=1.0,
    model=None,
):
    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "system", "content": out})
    messages.append({"role": "user", "content": input})

    if model is None:
        model = "gpt-3.5-turbo-16k" if use_16k else "gpt-3.5-turbo"

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE"),
    )

    completion = None
    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages,
            )
        except (APIError, APIConnectionError) as e:
            print(f"OpenAI API error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
        except RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)

    return completion.choices[0].message.content
