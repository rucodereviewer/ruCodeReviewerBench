from pydantic import BaseModel, Field
from prompts import system_prompt
from pathlib import Path
import pandas as pd
import litellm
import json
import re
import os


class BestComment(BaseModel):
    best_comment: str = Field(..., description="Лучший комментарий, из предложенных")


user_prompt = """
# Diff
{diff}

# Comments
{comments}
"""


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / 'data' / 'error analysis'
    selected_file = 'claude-3.7-sonnet.csv'

    df = pd.read_csv(f'{data_path}/{selected_file}')
    df['pred'] = df['pred'].fillna('Нет комментариев')
    df['extracted_diffs'] = df['input'].str.extract(
        r'Code changes:(.*?)(?=Answer:)', flags=re.DOTALL
    )

    messages = [
        [
            {
                'role': 'system',
                'content': system_prompt.strip(),
            },
            {
                'role': 'user',
                'content': user_prompt.format(
                    diff=i['extracted_diffs'],
                    comments=i['pred']
                ).strip(),
            }
        ] for _, i in df.iterrows()
    ]

    answers = litellm.batch_completion(
        model=os.getenv('MODEL'),
        messages=messages,
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('BASE_URL'),
        response_format=BestComment,
        temperature=0.7,
        timeout=None,
        top_p=0.8,
        seed=42,
    )

    parsed_answers = []

    for i in answers:
        try:
            asd = json.loads(i.choices[0].message.content)['best_comment']
            parsed_answers.append(asd)
        except Exception:
            parsed_answers.append('')
    
    df['best_comment'] = parsed_answers

    extraction_path = Path(__file__).parent.parent.parent / 'data' / 'prepared_data'
    df.to_json(f'{extraction_path}/claude-3.7-sonnet.json')
