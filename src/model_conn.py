import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from typing import List
from fastapi import FastAPI

from datetime import datetime
from pydantic import BaseModel
from functools import lru_cache
from loguru import logger

app = FastAPI()

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

@lru_cache()
def load_models():
    model_path = get_model_path("/Users/lefukuro/recommendation_system/notebooks/catboost_model_2")
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    return model

@lru_cache()
def batch_load_sql(query: str) -> pd.DataFrame:
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f'got chunck: {len(chunk_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)

@lru_cache()
def load_features() -> pd.DataFrame:
    logger.info('loading liked posts')
    liked_posts = batch_load_sql("SELECT DISTINCT post_id, user_id FROM public.feed_data WHERE action='like'")

    logger.info('loading posts features')
    posts_features = batch_load_sql("SELECT * FROM public.posts_features_morozova_ekaterina")
    
    logger.info('loading users features')
    users_features = batch_load_sql(
        "SELECT * FROM public.users_features_morozova_ekaterina"
    )

    return [liked_posts, posts_features, users_features]

logger.info('loading model')
model = load_models()

logger.info('loading features')
features = load_features()
logger.info('service is up and running')

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


def get_recommended_feed(
        id: int, 
        time: datetime, 
        limit: int = 5):
    
    logger.info(f'user id={id}')
    logger.info('reading features')
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    logger.info('dropping columns')
    posts_features = features[1].drop(['text', 'topic'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    logger.info('zipping everything')
    add_users_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info('assigning everything')
    user_posts_features = posts_features.assign(**add_users_features)
    user_posts_features = user_posts_features.set_index('post_id')

    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    logger.info('predicting')
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    logger.info('deleting like posts')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    _filtered = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = _filtered.sort_values('predicts')[-limit:].index

    return [PostGet(**{
        'id': i,
        'text': content[content.post_id == i].text.values[0],
        'topic':content[content.post_id == i].topic.values[0]
    }) for i in recommended_posts]


    
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int, 
        time: datetime, 
        limit: int = 5) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)