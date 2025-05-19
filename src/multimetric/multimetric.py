from prompts import multiple_scores_prompt
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import litellm
import json
import os


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


user_prompt = """
# Diff
{diff}

# Review
{model_review}
"""


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

            user_content = user_prompt.format(
                diff=patch,
                model_review=model_review,
            )

            message = [
                {
                    "role": "system",
                    "content": multiple_scores_prompt.strip()
                },
                {
                    "role": "user",
                    "content": user_content.strip()
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

        print('Sending requests to LLM...')
        responses = litellm.batch_completion(
            model=os.getenv('MODEL'),
            messages=prepared_prompts,
            api_key=os.getenv('API_KEY'),
            base_url=os.getenv('BASE_URL'),
            response_format=Metrics,

            seed=42,
        )
        print('Formatting responses...')

        responses = [json.loads(i.choices[0].message.content) for i in responses]
        
        answers = pd.DataFrame(responses)

        if eval_human:
            output_name = f'multi_scores/human_multi.json'
            answers.to_json(output_name)
        else:
            output_name = f'multi_scores/{self.file_path.replace('.json', '')}_multi.json'
            answers.to_json(output_name)
        print(f'Saved to {output_name}')
        return answers.mean()
 

if __name__ == '__main__':
    data_path = Path(__file__).parent.parent.parent / 'data' / 'prepared_data'
    data_files = os.listdir(data_path)
    for file in data_files:
        llm_judge = LLMJudge(file, data_path)
        llm_judge.count_multiple_scores()
    
    llm_judge.count_multiple_scores(eval_human=True)
