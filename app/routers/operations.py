from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy
from rapidfuzz import fuzz
import requests

url = "http://localhost:11434/api/chat"
headers = {"Content-Type": "application/json"}


def generate_recipe(prompt: str):
    data = {
        "model": "llama3.2",
        "messages": [
            {
                "role": "user",
                "content": f"Generate a food recipe for {prompt}"
            }
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ingredients": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "ingredients", "steps"]
        }
    }

    response = requests.post(url, json=data, headers=headers)
    return response.json()


def fuzzy_search(query, texts, limit=10, threshold=60):
    results = []
    for text in texts:
        score = fuzz.partial_ratio(query.lower(), text.lower())
        if score >= threshold:
            results.append((text, score))  # store both text and score

    # sort by score (highest first)
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # return only the text part
    return [text for text, score in results[:limit]]


class Operations:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def __init__(self):
        self.df = pd.read_pickle('recipes_to_embed.pkl')

        self.names_embeddings = np.load('new/names_embeddings.npy')
        self.average_steps_embeddings = np.load('new/average_steps_embeddings.npy')
        self.ingredient_avg_embeddings = np.load('new/ingredient_avg_embeddings.npy')

    def find_similar_recipe(self, title_embedding, directions_embedding, ingredients_embedding, weights):
        title_scores = cosine_similarity([title_embedding], self.names_embeddings)[0]
        directions_scores = cosine_similarity([directions_embedding], self.average_steps_embeddings)[0]
        ingredients_scores = cosine_similarity([ingredients_embedding], self.ingredient_avg_embeddings)[0]

        combined_scores = (
                weights[0] * title_scores +
                weights[1] * ingredients_scores +
                weights[2] * directions_scores
        )

        most_similar_idx = combined_scores.argmax()
        most_similar = self.df.iloc[most_similar_idx]
        most_similar_score = combined_scores[most_similar_idx]

        return most_similar, most_similar_score

    def lemmatize_ingredients(self, ingredient_list):
        lemmas = []
        for doc in self.nlp.pipe(ingredient_list, batch_size=64):
            lemmatized = " ".join([token.lemma_ for token in doc])
            lemmas.append(lemmatized)
        return lemmas

    def encode_lemmatized_list(self, values):
        if len(values) == 0:
            return np.zeros(384)

        embeddings = self.model.encode(self.lemmatize_ingredients(values), show_progress_bar=False)

        return np.mean(embeddings, axis=0)

    def encode_list(self, values):
        if len(values) == 0:
            return np.zeros(384)

        embeddings = self.model.encode(values, show_progress_bar=False)

        return np.mean(embeddings, axis=0)

    def find_most_similar(self, new_title, new_ingredients, new_directions, weights=(0.3, 0.4, 0.3)):
        title_embedding = self.model.encode(new_title)
        directions_embedding = self.encode_list(new_directions)
        ingredients_embedding = self.encode_lemmatized_list(new_ingredients)

        return self.find_similar_recipe(title_embedding, directions_embedding, ingredients_embedding, weights)

    def search_by_name(self, name):
        elements = fuzzy_search(name, self.df['name'])

        return self.df[self.df['name'].map(lambda el: el in elements)].to_dict(orient='records')

    def encode_weighted_list(self, values):
        if not values:
            return np.zeros(384)

        texts = [item.value for item in values]
        weights = np.array([item.weight for item in values])
        embeddings = self.model.encode(texts, show_progress_bar=False)

        weighted_embeddings = embeddings * weights[:, None]

        weighted_mean = np.mean(weighted_embeddings, axis=0)
        return weighted_mean

    def encode_lemmatized_weighted_list(self, weighted_values):
        if not weighted_values:
            return np.zeros(384)

        texts = [item.value for item in weighted_values]
        weights = np.array([item.weight for item in weighted_values])

        lemmatized_texts = self.lemmatize_ingredients(texts)

        embeddings = self.model.encode(lemmatized_texts, show_progress_bar=False)

        weighted_embeddings = embeddings * weights[:, None]

        weighted_mean = np.mean(weighted_embeddings, axis=0)
        return weighted_mean

    def search_similarity(self, new_titles, new_ingredients, new_directions, weights=(0.3, 0.4, 0.3)):
        title_embedding = self.encode_weighted_list(new_titles)
        directions_embedding = self.encode_weighted_list(new_directions)
        ingredients_embedding = self.encode_lemmatized_weighted_list(new_ingredients)

        return self.find_similar_recipe(title_embedding, directions_embedding, ingredients_embedding, weights)
