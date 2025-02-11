import logging
import time
import os
from functools import lru_cache

import openai
from diskcache import Cache

from commaqa.inference.prompt_reader import fit_prompt_into_given_limit

# use tiktoken instead of gpt2 tokenizer
import tiktoken

logger = logging.getLogger(__name__)


cache = Cache(os.path.expanduser("~/.cache/gpt3calls"))

MODEL_NAME = "davinci-002"
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="http://localhost:8000/v1")


@cache.memoize()
def cached_openai_call(  # kwargs doesn't work with caching.
    prompt,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    best_of,
    logprobs,
):
    return client.completions.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        best_of=best_of,
        logprobs=logprobs,
    )


def openai_call(
    prompt,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    best_of,
    logprobs,
):
    function = cached_openai_call if temperature == 0 else client.completions.create
    return function(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        best_of=best_of,
        logprobs=logprobs,
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    # must fetch tokenizer from transformers to use with current openai client
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)
    # return tiktoken.encoding_for_model(MODEL_NAME)


class GPT3Generator:
    def __init__(
        self,
        engine=MODEL_NAME,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"], # NOTE: changed from \n to . since gpt3.5-turbo-instruct always starts with \n
        retry_after_n_seconds=None,
        n=1,
        best_of=1,
        logprobs=0,
        remove_method="first",
    ):
        print("-" * 100, "ENGINE", engine)
        self.model = MODEL_NAME
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens # idk if i can hardcodde
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        self.retry_after_n_seconds = retry_after_n_seconds
        self.remove_method = remove_method

        # if "code-davinci" not in model:
        #     raise Exception("Not allowed to prevent accidental $$ wastage.")

        # if "code-davinci" not in model and self.retry_after_n_seconds is not None:
        #     raise Exception(
        #         "Retry is only supported for code-davinci as it's free. "
        #         "Using it for other paid models is risky and so is disabled."
        #     )

        self.model_tokens_limit = 20_000

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        # GPT3 can't handle trailing white-space
        prompt = prompt.rstrip()

        # probably don't need this with modern day context limits
        
        # prompt = fit_prompt_into_given_limit(
        #     original_prompt=prompt,
        #     model_length_limit=self.model_tokens_limit,
        #     estimated_generation_length=self.max_tokens,
        #     demonstration_delimiter="\n\n\n",
        #     shuffle=False,
        #     remove_method=self.remove_method,
        #     tokenizer_model_name="gpt2",  # did this before tiktoken was released.
        #     last_is_test_example=True,
        # )
        
        arguments = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "best_of": self.best_of,
            "logprobs": self.logprobs,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }

        print("-" * 100, "STOP", self.stop)
        if self.best_of is not None:
            arguments["best_of"] = self.best_of

        success = False
        for index in range(500):
            try:
                response = openai_call(**arguments)
                success = True
                break
            except Exception as exception:

                success = False

                tokenizer = get_tokenizer()
                # prompt_num_tokens = len(tokenizer(prompt)['input_ids'])
                prompt_num_tokens = len(tokenizer.encode(prompt))
                if prompt_num_tokens + arguments["max_tokens"] > self.model_tokens_limit > prompt_num_tokens:
                    last_used_max_tokens = arguments["max_tokens"]
                    updated_max_tokens = self.model_tokens_limit - prompt_num_tokens
                    arguments["max_tokens"] = updated_max_tokens
                    if last_used_max_tokens == updated_max_tokens:
                        break
                    print(
                        f"WARNING: (Round {index}) Decreasing max_tokens from "
                        f"{last_used_max_tokens} to {updated_max_tokens} and retrying."
                    )
                    continue

                if self.retry_after_n_seconds is None:
                    import traceback

                    print(traceback.format_exc())
                    exit()

                print(f"Encountered exception of class: {exception.__class__}")
                if hasattr(exception, "user_message"):
                    print(exception.user_message)
                print(f"Potentially reached OpenAI rate limit. Will try again in {self.retry_after_n_seconds}s.")
                time.sleep(self.retry_after_n_seconds)
                pass

        if not success:
            raise Exception("Could not complete OpenAI call")

        output_seq_score = []
        for index, choice in enumerate(response.choices):
            if "logprobs" in choice and "token_logprobs" in choice["logprobs"]:
                probs = []
                for prob, tok in zip(choice["logprobs"]["token_logprobs"], choice["logprobs"]["tokens"]):
                    if tok not in self.stop and tok != "<|endoftext|>":
                        probs.append(prob)
                    else:
                        probs.append(prob)
                        break

                score = -sum(probs) / len(probs) if len(probs) else 100.0
                output_seq_score.append((choice.text, score))
            else:
                output_seq_score.append((choice.text, index))

        return sorted(output_seq_score, key=lambda x: x[1])
