
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic.representation import OpenAI as OpenAI_representation
import datamapplot
import tiktoken
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
import os
from openai import OpenAI
import argparse



prompt = """
You are an expert assistant trained to analyze source code review comments and help categorize them. Your task is to generate a short, precise category label based on a list of keywords and representative comments.

Please follow these instructions:
- The label should be **short and descriptive** (between 2 and 5 words).
- It should **summarize the core idea** of the comments, **not just repeat the keywords**.
- The label will be used to **group similar review comments** into categories, so aim for a meaningful and general name that reflects the issue being discussed.
- Avoid vague terms. Be as specific and useful as possible.
- These categories may include style issues, logic flaws, structure suggestions, performance concerns, testing gaps, and more.

Here is the data:
- Keywords: [KEYWORDS]
- Sample review comments:
[DOCUMENTS]

Generate a clear, useful category label that best describes the type of comments shown above.
return in format:
topic: <category label>
"""


def load_data(path: Path) -> list[str]:
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            documents.append(obj['outputs'])
    return documents


def get_embeddings(documents: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    embeddings = model.encode(documents, show_progress_bar=True)
    return embeddings


def get_representation_model() -> OpenAI_representation:
    
    aspect_model1 = PartOfSpeech("en_core_web_sm")
    aspect_model2 = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]
    aspect_model3 = KeyBERTInspired()
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    client = OpenAI(
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY']
    )
    openai_representation = OpenAI_representation(
            client,
            model=os.environ['MODEL'],
            chat=True,
            doc_length=20000,
            tokenizer=tokenizer,
            nr_docs=10,
            prompt = prompt
        )
    
    representation_model = {
    "Main": openai_representation,
    "Aspect1":  aspect_model1,
    "Aspect2":  aspect_model2,
    "Aspect3":  aspect_model3
    }
    
    return representation_model


def build_model_k(k, embeddings, docs, model, representation_model):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    umap_model = UMAP(n_neighbors=15, n_components=5,
                    min_dist=0.0, metric='cosine', random_state=42)

    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        calculate_probabilities=True,
        verbose=False,
        hdbscan_model=km,
        representation_model=representation_model

    )
    topics, probs = topic_model.fit_transform(
        docs, embeddings=embeddings,
    )
    return topic_model, topics



def add_topics_to_data(topic_model, topics, docs):
    docs_with_topics = []
    for doc, topic in zip(docs, topics):
        docs_with_topics.append({
            "doc": doc,
            "topic": topic
        })
    return docs_with_topics


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    data = load_data(args.data_path)
    
    embeddings = get_embeddings(data, "intfloat/multilingual-e5-large-instruct")
    
    representation_model = get_representation_model()
    
    topic_model, topics = build_model_k(12, embeddings, data, "intfloat/multilingual-e5-large-instruct", representation_model)
    
    docs_with_topics = add_topics_to_data(topic_model, topics, data)
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        for doc in docs_with_topics:
            f.write(json.dumps(doc) + "\n")
    
    