from typing import Any, Dict
from collections import defaultdict

import json, re, os, time, logging, math
from concurrent.futures import ThreadPoolExecutor

try:
    import litellm, httpx, sacrebleu
    from openai import OpenAI
    import numpy as np
    import pandas as pd
    from tabulate import tabulate
    from tqdm import tqdm
except ImportError:
    raise ImportError("Some libraries are not installed. Please install them by running: pip install numpy pandas tabulate openai litellm httpx sacrebleu tqdm")


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.propagate = False

litellm._logging._disable_debugging()
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

litellm.client_session = httpx.Client(verify=False)
litellm.aclient_session = httpx.AsyncClient(verify=False)

n = 10


def parse_comments(response):
    comment_pattern = re.compile(r'Комментарий \d+:\s*(.*?)(?=Комментарий \d+:|$)', re.DOTALL)

    comments = comment_pattern.findall(response)

    comments = comments[:n]

    comments = [comment.strip() for comment in comments if comment.strip()]

    return comments


JUDGE_PROMPT_FIRST = """Вы - судья.
Вам будет предоставлен фрагмент изменений в коде и два комментария из кода, описывающие проблему.
Ваша задача — определить, описывают ли оба комментария одну и ту же проблему.
Если два комментария описывают одну и ту же проблему и предлагаемое решение, ответьте: correct.
Если комментарии описывают разные проблемы и решения, ответьте: wrong.
Напишите ваш ответ, correct или wrong, без кавычек и других лишних символов.\n\n"""

JUDGE_PROMPT_SECOND = """Input data:
code block difference: {diff_block}
comment 1: {comment1}
comment 2: {comment2}
answer:"""


class llmAsAJudge():
    def __init__(self) -> None:
        self.few_shot_path = os.getenv('JUDGE_FEW_SHOT_PATH', "../data/few_shot.json")
        self.max_retries = int(os.getenv('JUDGE_MAX_RETRIES', 10))
        self.retry_delay = int(os.getenv('JUDGE_RETRY_DELAY', 5))
        self.model_name = os.getenv('JUDGE_MODEL_NAME', "qwen-coder-32b")
        self.temperature = float(os.getenv('JUDGE_TEMPERATURE', 0.))
        self.max_tokens = int(os.getenv('JUDGE_MAX_TOKENS', 100))
        self.provider = os.getenv('JUDGE_CUSTOM_LLM_PROVIDER', "openai")
        self.api_key = os.getenv('JUDGE_API_KEY', 'None')
        self.url = os.getenv('JUDGE_URL', ""),

        if not os.path.exists(self.few_shot_path):
            err = f"File '{self.few_shot_path}' not found."
            logger.error(err)
            raise FileNotFoundError(err)

        with open(self.few_shot_path, 'r') as file:
            few_shots = json.load(file)

        self.prompt_first = JUDGE_PROMPT_FIRST

        for idx, sample in enumerate(few_shots):
            self.prompt_first += f"Example {idx + 1}:\ncode block difference: {sample['diff_block']}\ncomment 1: {sample['comment1']}\ncomment 2: {sample['comment2']}\nanswer: {sample['answer']}\n\n"

        self.prompt_second = JUDGE_PROMPT_SECOND

    def post_query(self, message):
        messages = [{"role": "user", "content": message}]
        completion_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.url,
        }

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    **completion_params,
                    custom_llm_provider=self.provider,
                    api_key=self.api_key,
                )
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    err = f"All {self.max_retries} attempts failed. Error: {str(e)}"
                    logger.error(err)
                    raise RuntimeError(err)

    def calculate(self, doc: Dict[str, Any], result: str) -> list[float]:
        data = {'diff_block': doc['inputs']['diff_block'], 'comment1': doc['outputs']}

        labels = [0. for _ in range(n)]
        for comment_id, comment in enumerate(parse_comments(result)):
            data['comment2'] = comment

            prompt = self.prompt_first + self.prompt_second.format(**data)

            res = self.post_query(prompt)

            res = res.lower().strip()

            if res.startswith('correct') or res.endswith('correct'):
                labels[comment_id] = 1.
            if res.startswith('wrong') or res.endswith('wrong'):
                labels[comment_id] = 0.

        return labels


class llmASaJudgeScoring():
    def __init__(self) -> None:
        self.judge = llmAsAJudge()
        self.model = os.getenv('MODEL_PATH', "qwen/qwen3-8b")

        self.client = OpenAI(
            base_url=os.getenv('MODEL_BASE_URL', "https://openrouter.ai/api/v1"),
            api_key=os.getenv('MODEL_API_KEY', ""),
        )
        self.extra_headers={
            "HTTP-Referer": os.getenv('HTTP-REFERER', "https://openrouter.ai"),
            "X-Title": os.getenv('X-TITLE', "<YOUR_SITE_NAME>"),
        }

        self.max_retries = int(os.getenv('MODEL_MAX_RETRIES', 10))
        self.retry_delay = int(os.getenv('MODEL_RETRY_DELAY', 60))

        self.predict_dict = defaultdict(dict)

        self.test = os.getenv('MODEL_TEST_MODE', "False")
        self.n = 5 if self.test.lower() == "true" else 10**10

    def post_query(self, message):
        if self.n == 0:
            return ""
        else:
            self.n -= 1

        messages = [{"role": "user", "content": message}]

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                  extra_headers=self.extra_headers,
                  model=self.model,
                  messages=messages,
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    err = f"All {self.max_retries} attempts failed. Error: {str(e)}"
                    logger.error(err)
                    raise RuntimeError(err)

    def process_sample(self, idx: int, doc: Dict[str, Any]) -> Dict[str, float]:
        inp = doc_to_text(doc)
        sample = self.post_query(inp)

        labels = self.judge.calculate(doc, sample)
        sample_metrics = compute_classic_metrics(doc['outputs'], sample)

        for k in [1, 5, n]:
            sample_metrics[f"pass@{k}"] = float(sum(labels[:k]) > 0)

        self.predict_dict[idx]['input'] = inp
        self.predict_dict[idx]['gold'] = doc['outputs']
        self.predict_dict[idx]['pred'] = sample
        self.predict_dict[idx]['metrics'] = sample_metrics
        self.predict_dict[idx]['llm_as_a_judge_labels'] = labels

        return sample_metrics

    def apply(self, docs: list[Dict[str, Any]]) -> list[Dict[str, float]]:
        self.predict_dict = defaultdict(dict)

        num_docs = len(docs)
        with ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 50))) as executor:
            tasks = [(idx, doc) for idx, doc in enumerate(docs)]
            results = []
            with tqdm(total=num_docs, desc="Processing Docs") as pbar:
                for result in executor.map(lambda x: self.process_sample(*x), tasks):
                    results.append(result)
                    pbar.update(1)

        pd.DataFrame(self.predict_dict).T.to_csv(
            os.getenv('RESULT_PATH', "results/" + self.model.split('/')[-1] + ".csv")
        )

        return results


def compute_classic_metrics(answer: str, message: str) -> Dict[str, float]:
    comments = parse_comments(message)

    result = {'bleu': [], 'chrf': []}

    a = [answer]
    for comment in comments:
        b = [[comment]]
        result['bleu'].append(sacrebleu.corpus_bleu(a, b).score)
        result['chrf'].append(sacrebleu.corpus_chrf(a, b).score)

    result['bleu'] = max(result['bleu']) if result['bleu'] else 0.
    result['chrf'] = max(result['chrf']) if result['chrf'] else 0.

    return result


def doc_to_text(doc: Dict[str, Any]) -> str:
    prompt = doc["instruction"].format(**doc["inputs"])

    return prompt


def read_jsonl(file_path):
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)

    return data


def sample_stddev(arr):
    mu = np.mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


def print_result(data, precision=2):
    metric_names = list(data[0].keys())

    metric_values = {metric: [] for metric in metric_names}

    for item in data:
        for metric in metric_names:
            metric_values[metric].append(item[metric])

    table_data = []
    for metric in metric_names:
        values = metric_values[metric]

        table_data.append([metric, f"{np.mean(values):.{precision}f} ± {mean_stderr(values):.{precision}f}"])

    headers = ["Metric", ""]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    scoring = llmASaJudgeScoring()
    docs = read_jsonl(os.getenv('DATASET_PATH', "../data/rucodereview_main.jsonl"))
    metrics = scoring.apply(docs)
    print_result(metrics)


if __name__ == "__main__":
    main()