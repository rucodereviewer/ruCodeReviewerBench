from pydantic import BaseModel, Field
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import litellm
import logging
import httpx
import json
import os
import re


logger = logging.getLogger(__name__)
litellm.client_session = httpx.Client(verify=False)
litellm.aclient_session = httpx.AsyncClient(verify=False)


#  --------------------------  Multimetric  ---------------------------

MULTIPLE_SCORES_PROMPT = """
You are presented with a code instance featuring some issues.
Input information includes the problem code fragment and the review.

Please evaluate the **review** based on the following metrics.
Provide a score from 1-10 for each metric (higher is better).

**Metrics**
1. **Readability**: Is the comment easily understood, written in clear, straightforward language?
2. **Relevance**: Does the comment directly relate to the issues in the code, excluding unrelated information?
3. **Explanation Clarity**: How well does the comment explain the issues, beyond simple problem identification?
4. **Problem Identification**: How accurately and clearly does the comment identify and describe the bugs in the code?
5. **Actionability**: Does the comment provide practical, actionable advice to guide developers in rectifying the code errors?
6. **Completeness**: Does the comment provide a comprehensive overview of all issues within the problematic code?
7. **Specificity**: How precisely does the comment pinpoint the specific issues within the problematic code?
8. **Contextual Adequacy**: Does the comment align with the context of the problematic code, relating directly to its specifics?
9. **Consistency**: How uniform is the comment's quality, relevance, and other aspects comparing to the former sample?
10. **Brevity**: How concise and to-the-point is the comment, conveying necessary information in as few words as possible?

**Input**
- Diff
- Review
"""

class Metrics(BaseModel):
    readability: int
    relevance: int
    explanation_clarity: int
    problem_identification: int
    actionability: int
    completeness: int
    specificity: int
    contextual_adequacy: int
    consistency: int
    brevity: int


#  -------------------------  Best comment  ---------------------------

SYSTEM_PROMPT = """
Ты - ассистент, который помогает выявить лучший комментарий на предложенный код.

# Входные данные
На вход ты получишь следующие данные:
- код с изменениями (Diff)
- комментарии к коду (Comments)

# Задача
Выявить самый лучший комментарий и переписать его.
"""

USER_PROMPT = """
# Diff
{diff}

# Review
{model_review}
"""

class BestComment(BaseModel):
    best_comment: str = Field(..., description="Лучший комментарий, из предложенных")


class LLMJudge:
    def __init__(
        self,
        file_path: str,
        data_path: str,
    ) -> None:
        self.file_path = file_path
        self.file_data = pd.read_json(f'{data_path}/{file_path}')
    
    def prepare_prompt(self, inputs: list[str], preds: list[str]) -> str:
        prepared_prompts = []
        for patch, model_review in tqdm(zip(inputs, preds), total=len(preds)):

            if not isinstance(patch, str):
                patch = ''
            if not isinstance(model_review, str):
                model_review = ''

            message = [
                {
                    "role": "system",
                    "content": MULTIPLE_SCORES_PROMPT.strip()
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        diff=patch,
                        model_review=model_review,
                    ).strip()
                }
            ]

            prepared_prompts.append(message)
        return prepared_prompts

    def count_multiple_scores(
        self,
        eval_human: bool = False
    ) -> pd.DataFrame:
        inputs = self.file_data['extracted_diffs']
        preds = self.file_data['best_comment']

        if eval_human:
            preds = self.file_data['gold']

        prepared_prompts = self.prepare_prompt(inputs, preds)

        logger.info('Sending requests to LLM...')
        responses = litellm.batch_completion(
            model=os.getenv('JUDGE_MODEL_NAME', "qwen/qwen-2.5-coder-32b-instruct"),
            base_url=os.getenv('JUDGE_URL', "https://openrouter.ai/api/v1"),
            api_key=os.getenv('JUDGE_API_KEY', "None"),
            seed=os.getenv('MULTI_SEED', 42),
            messages=prepared_prompts,
            response_format=Metrics,
        )
        logger.info('Formatting responses...')

        responses = [json.loads(i.choices[0].message.content) for i in responses]
        
        answers = pd.DataFrame(responses)

        if eval_human:
            output_name = f'multi_scores/human_multi.json'
            answers.to_json(output_name)
        else:
            output_name = f'multi_scores/{self.file_path.replace('.json', '')}_multi.json'
            answers.to_json(output_name)
        logger.info(f'Saved to {output_name}')
        return answers.mean()
    

def get_best_comment() -> None:
    results_path = Path(__file__).parent / 'results'
    model = os.getenv('MODEL_PATH', "qwen/qwen3-8b")
    model = model.split('/')[-1]

    df = pd.read_csv(f'{results_path}/{model}.csv')
    df['pred'] = df['pred'].fillna('Нет комментариев')
    df['extracted_diffs'] = df['input'].str.extract(
        r'Code changes:(.*?)(?=Answer:)', flags=re.DOTALL
    )

    messages = [
        [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT.strip(),
            },
            {
                'role': 'user',
                'content': USER_PROMPT.format(
                    diff=i['extracted_diffs'],
                    comments=i['pred']
                ).strip(),
            }
        ] for _, i in df.iterrows()
    ]

    answers = litellm.batch_completion(
        model=os.getenv('JUDGE_MODEL_NAME', "qwen/qwen-2.5-coder-32b-instruct"),
        base_url=os.getenv('JUDGE_URL', "https://openrouter.ai/api/v1"),
        temperature=os.getenv('MULTI_TEMPERATURE', 0.7),
        api_key=os.getenv('JUDGE_API_KEY', "None"),     
        top_p=os.getenv('MULTI_TOP_P', 0.8),
        seed=os.getenv('MULTI_SEED', 42),
        response_format=BestComment,
        messages=messages,
        timeout=120,
    )

    parsed_answers = []

    for i in answers:
        try:
            best_com = json.loads(i.choices[0].message.content)['best_comment']
            parsed_answers.append(best_com)
        except Exception:
            parsed_answers.append('')
    
    df['best_comment'] = parsed_answers

    extraction_path = Path(__file__).parent / 'prepared_data'
    df.to_json(f'{extraction_path}/{model}.json')


def get_multimetric() -> None:
    data_path = Path(__file__).parent / 'prepared_data'
    model = os.getenv('MODEL_PATH', "qwen/qwen3-8b")
    model = model.split('/')[-1] + '.json'

    llm_judge = LLMJudge(model, data_path)
    llm_judge.count_multiple_scores()
    llm_judge.count_multiple_scores(eval_human=True)


if __name__ == '__main__':
    get_best_comment()
    get_multimetric()
    results_path = Path(__file__).parent / 'prepared_data'
    logger.info(f"Done. Check results in {results_path} folder")
