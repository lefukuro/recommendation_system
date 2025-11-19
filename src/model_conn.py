import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from typing import List
from fastapi import FastAPI

from datetime import datetime
from pydantic import BaseModel
from functools import lru_cache


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

@lru_cache()
def load_models():
    model_path = get_model_path("catboost_model.cbm")
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    return model

@lru_cache()
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

@lru_cache()
def load_features() -> pd.DataFrame:

    features_query = "SELECT * FROM morozova_ekaterina_les_22"
    features = batch_load_sql(features_query)
    
    
    posts_query = "SELECT * FROM public.post_text_df"
    posts = batch_load_sql(posts_query)
    
    features_with_text = features.merge(posts, on='post_id', how='left')

    if 'timestamp' in features_with_text.columns:
        features_with_text['timestamp'] = (
            pd.to_datetime(features_with_text['timestamp']))
    
    return features_with_text

app = FastAPI()
model = load_models()
features = load_features()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int, 
        time: datetime, 
        limit: int = 5) -> List[PostGet]:
    

    user_features = features[features['user_id'] == id]
    
    if user_features.empty:
        return []

    feature_columns = [col for col in user_features.columns 
                      if col not in ['user_id', 'post_id', 'text', 'topic', 'target']]
    
    X = user_features[feature_columns]

    predictions = model.predict_proba(X)[:, 1]
    user_features = user_features.copy()
    user_features['prediction'] = predictions
    
    top_posts = user_features.sort_values('prediction', ascending=False).head(limit)
    
    result = []
    for _, row in top_posts.iterrows():
        post = PostGet(
            id=int(row['post_id']),
            text=str(row['text']),
            topic=str(row['topic'])
        )
        result.append(post)
    
    return result
