from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from pathlib import Path
import pandas as pd
import asyncio
import httpx
import os


async def process_message(client, msg):
    try:
        completion = await client.beta.chat.completions.parse(
            model=os.getenv('MODEL'),
            messages=msg,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return e

async def main(messages):
    # Configure client with SSL verification disabled
    http_client = httpx.AsyncClient(verify=False)
    
    client = AsyncOpenAI(
        base_url=os.getenv('BASE_URL'),
        api_key=os.getenv('API_KEY'),
        http_client=http_client
    )
    
    # Create semaphore to limit concurrent requests to 30
    semaphore = asyncio.Semaphore(50)
    
    async def bounded_process(msg):
        async with semaphore:
            return await process_message(client, msg)
    
    # Process messages concurrently
    tasks = [bounded_process(msg) for msg in messages]
    answers = await tqdm_asyncio.gather(
        *tasks, desc="Processing messages"
    )
    
    # Clean up
    await http_client.aclose()
    await client.close()
    
    return answers


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent.parent / 'data' / 'error analysis'
    file_path = 'gpt-4o.csv'
    data_file = pd.read_csv(f'{data_path}/{file_path}')

    messages = [
        [
            {
                'role': 'user',
                'content': i
            }
        ] for i in data_file['input']
    ]

    answers = asyncio.run(main(messages))
    final_df = pd.DataFrame(answers)
    final_df.to_json('answers.json')
