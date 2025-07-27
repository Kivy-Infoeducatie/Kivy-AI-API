from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from json import loads

from operations import Operations, generate_recipe

app = FastAPI()

operations = Operations()


@app.get("/recipes/search/by-name")
async def search_recipe_by_name(name: str):
    print(name)
    return operations.search_by_name(name)


class RecipeSearchParams(BaseModel):
    name: str
    ingredients: List[str]
    steps: List[str]
    nameWeight: float
    ingredientsWeight: float
    stepsWeight: float


@app.post("/recipes/search")
async def search_recipes(params: RecipeSearchParams):
    result = operations.find_most_similar(params.name, params.ingredients, params.steps,
                                          weights=(params.nameWeight, params.ingredientsWeight, params.stepsWeight))

    ret = result[0].to_dict()

    ret['score'] = float(result[1])

    return ret


class TextWeight(BaseModel):
    value: str
    weight: float


class SimilaritySearchParams(BaseModel):
    names: List[TextWeight]
    ingredients: List[TextWeight]
    steps: List[TextWeight]
    nameWeight: float
    ingredientsWeight: float
    stepsWeight: float


@app.post("/recipes/similar")
async def similar_recipes(params: SimilaritySearchParams):
    result = operations.search_similarity(params.names, params.ingredients, params.steps,
                                          weights=(params.nameWeight, params.ingredientsWeight, params.stepsWeight))

    ret = result[0].to_dict()

    ret['score'] = float(result[1])

    return ret


@app.post("/recipes/generate")
async def generate(prompt: str):
    return loads(generate_recipe(prompt)['message']['content'])



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
