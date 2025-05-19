# Running the Benchmark

To execute the benchmark, follow these steps:

**1. Setting Environment Variables:**

Before running `llm_as_a_judge.py`, the following environment variables must be configured: 

-   `JUDGE_URL`: The URL to the server hosting the deployed LLM used for evaluating the quality of code reviews.  It is recommended to use the `qwen2.5-coder-32b-instruct` model.
-   `MODEL_PATH`: The fully qualified name of the model used for code review generation.
-   `MODEL_API_KEY`: The API key for accessing the LLM.

**Example Configuration for OpenRouter Models:**

When using models available through [OpenRouter](https://openrouter.ai/), `MODEL_PATH` should correspond to the model name on OpenRouter.  `MODEL_API_KEY` should be your OpenRouter API key.

**2. Executing the LLM as a Judge Script:**

Run the `llm_as_a_judge.py` script using the following command:

```bash
python -m llm_as_a_judge.py
```

**3. Executing the Multimetric Script:**

Run the `multimetric.py` script using the following command:

```bash
python -m multimetric.py
```

**Additional Environment Variables:**

The following environment variables provide further customization options for the scripts behavior:

-   **For LLM-as-a-Judge:**
    *   `JUDGE_FEW_SHOT_PATH`: Path to the file containing few-shot examples for the LLM-as-a-Judge.  Default: `../data/few_shot.json`.
    *   `JUDGE_MAX_RETRIES`: Maximum number of retry attempts when querying the LLM-as-a-Judge.  Default: `10`.
    *   `JUDGE_RETRY_DELAY`: Delay in seconds between retry attempts when querying the LLM-as-a-Judge. Default: `5`.
    *   `JUDGE_MODEL_NAME`: Name of the model used for the LLM-as-a-Judge. Default: `qwen-coder-32b`.
    *   `JUDGE_TEMPERATURE`: Temperature for the LLM-as-a-Judge. Default: `0.0`.
    *   `JUDGE_MAX_TOKENS`: Maximum number of tokens in the LLM-as-a-Judge's response. Default: `100`.
    *   `JUDGE_CUSTOM_LLM_PROVIDER`: LLM provider for LLM-as-a-Judge. Default: `openai`.
    *   `JUDGE_API_KEY`: API key for the LLM provider used by the LLM-as-a-Judge. Default: `None`.

-   **For the Primary LLM:**
    *   `MODEL_BASE_URL`: Base URL for requests to the primary LLM. Default: `"https://openrouter.ai/api/v1"`.
    *   `HTTP-REFERER`: HTTP Referer header for requests to the primary LLM. Default: `"https://openrouter.ai"`.
    *   `X-TITLE`: X-Title header for requests to the primary LLM (required for OpenRouter).
    *   `MODEL_MAX_RETRIES`: Maximum number of retry attempts when querying the primary LLM. Default: `10`.
    *   `MODEL_RETRY_DELAY`: Delay in seconds between retry attempts when querying the primary LLM. Default: `60`.
    *   `MODEL_TEST_MODE`: Enables test mode, processing only 5 examples. Default: `"False"`.
    *   `DATASET_PATH`: Path to the dataset file. Default: `"../data/rucodereview_main.jsonl"`.
    *   `RESULT_PATH`: Path for saving the script's output in CSV format. Default: `"results/" + MODEL_PATH.split('/')[-1] + ".csv"`.
    *   `MAX_WORKERS`: Number of threads for parallel data processing. Default: `50`.

-  **For the Multimetric:**
    *   `MULTI_TEMPERATURE` - Temperature for the Multimetric LLM. Default: `0.7`.
    *   `MULTI_SEED` - Seed for the Multimetric LLM. Default: `42`.
    *   `MULTI_TOP_P` - Top-P for the Multimetric LLM. Default: `0.8`.
    *   `JUDGE_MODEL_NAME`: Name of the model used for the LLM-as-a-Judge. Default: `qwen-coder-32b`.
    *   `JUDGE_API_KEY`: API key for the LLM provider used by the LLM-as-a-Judge. Default: `None`.
