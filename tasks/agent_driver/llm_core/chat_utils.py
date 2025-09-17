# import openai
from openai import OpenAI


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    client = OpenAI()
    # print("completion_with_backoff kwargs:", kwargs)
    # input()
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)