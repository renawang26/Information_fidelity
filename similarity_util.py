

# %% use Google TensorFlow universal-sentence-encoder-multilingual_3 model to calculate the similarity

import os
import openai
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
import tensorflow_hub as hub
import tensorflow.compat.v2 as tf
from tensorflow_text import SentencepieceTokenizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# format sliding sentences 
def sliding_window_iter(series, size):
    """series is a column of a dataframe"""
    for start_row in range(len(series) - size + 1):
        yield series[start_row:start_row + size]
        
def get_sliding_sentences(df, size):
    df_sent_windows = sliding_window_iter(df, size)
    
    sent_ids, original_sentences, translation_1_sentences, translation_2_sentences = [], [], [], []

    sent_id = 1
    for i in df_sent_windows:
        sent_ids.append(sent_id)
        org, snet_1, sent_2 = [], [], []
        for idx in range(size): 
            org.append(i.values[idx][0])
            snet_1.append(i.values[idx][1])
            sent_2.append(i.values[idx][2])
            
        original_sentences.append(' '.join(org))
        translation_1_sentences.append(' '.join(snet_1))
        translation_2_sentences.append(' '.join(sent_2))
        sent_id += 1
        print(i.values, end = '\n\n')
        
    return pd.DataFrame(zip(sent_ids, original_sentences, translation_1_sentences, translation_2_sentences), 
                                columns=['sent_id', 'Original', 'Translation_1', 'Translation_2'])

# calculate similarity use universal-sentence-encoder-multilingual_3 model
def calculate_similarity_usem(df):
    sent_ids = df['sent_id'].tolist()
    original_sents = df["Original"].tolist()
    translation_1_sents = df["Translation_1"].tolist()
    translation_2_sents = df["Translation_2"].tolist()

    model_usem = hub.KerasLayer(hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"))
  
    original_embed = model_usem(original_sents)
    translation_1_embed = model_usem(translation_1_sents)
    translation_2_embed = model_usem(translation_2_sents)

    assert len(original_embed) == len(original_sents)
    assert len(translation_1_embed) == len(translation_1_sents)
    assert len(translation_2_embed) == len(translation_2_sents)

    # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
    sim_tf_1 = 1 - np.arccos(sklearn.metrics.pairwise.cosine_similarity(original_embed, translation_1_embed))/np.pi
    sim_tf_2 = 1 - np.arccos(sklearn.metrics.pairwise.cosine_similarity(original_embed, translation_2_embed))/np.pi

    sent_ids_col, original_sentences, translation_1_sentences, translation_2_sentences, sim_tf_1_col, sim_tf_2_col = [], [], [], [], [], []

    for i in range(len(original_embed)):
        sent_ids_col.append(sent_ids[i])
        original_sentences.append(original_sents[i])
        translation_1_sentences.append(translation_1_sents[i])
        translation_2_sentences.append(translation_2_sents[i])
        sim_tf_1_col.append(sim_tf_1[i][i])
        sim_tf_2_col.append(sim_tf_2[i][i])
        
    df_sim_tf = pd.DataFrame(zip(sent_ids_col, original_sentences, translation_1_sentences, translation_2_sentences, sim_tf_1_col, sim_tf_2_col), 
                        columns=['sent_id', 'original', 'translation_1', 'translation_2', 'sim_tf_1', 'sim_tf_2'])
    
    return df_sim_tf
    
# use Transformers to do sentence embed and calculate similarity
def calculate_similarity_transformers(df):
    
    model_transformers = SentenceTransformer('all-MiniLM-L6-v2')
   
    sent_ids = df['sent_id'].tolist()
    original_sents = df["Original"].tolist()
    translation_1_sents = df["Translation_1"].tolist()
    translation_2_sents = df["Translation_2"].tolist()

    original_embed = model_transformers.encode(original_sents)
    translation_1_embed = model_transformers.encode(translation_1_sents)
    translation_2_embed = model_transformers.encode(translation_2_sents)

    assert len(original_embed) == len(original_sents)
    assert len(translation_1_embed) == len(translation_1_sents)
    assert len(translation_2_embed) == len(translation_2_sents)
    
    sim_transform_1 = 1 - np.arccos(sklearn.metrics.pairwise.cosine_similarity(original_embed, translation_1_embed))/np.pi
    sim_transform_2 = 1 - np.arccos(sklearn.metrics.pairwise.cosine_similarity(original_embed, translation_2_embed))/np.pi
    
    sent_ids_col, sim_transform_1_col, sim_transform_2_col = [], [], []
      
    for i in range(len(original_embed)):
        sent_ids_col.append(int(sent_ids[i]))
        sim_transform_1_col.append(sim_transform_1[i][i])
        sim_transform_2_col.append(sim_transform_2[i][i])

    df_sim_transformer = pd.DataFrame(zip(sent_ids_col, sim_transform_1_col, sim_transform_2_col), columns=['sent_id', 'sim_transformers_1', 'sim_transformers_2'])
    
    return df_sim_transformer
    
# use OpenAI to do sentence embed
def calculate_similarity_openai(df):

    openai.api_key = 'replace with you OpenAI key' 
    model_engine = "text-embedding-ada-002"

    sent_openai_ids_col, sim_openai_1_col, sim_openai_2_col = [], [], []

    for index, row in df.iterrows():
        sent_openai_ids_col.append(row['sent_id'])
        
        embeddings_openai_org = openai.Embedding.create(model=model_engine, input=[row['Original']])
        embeddings_openai_1 = openai.Embedding.create(model=model_engine, input=[row['Translation_1']])
        embeddings_openai_2 = openai.Embedding.create(model=model_engine, input=[row['Translation_2']])

        # Extract embedding vectors from the response
        vector_openai_org = np.array(embeddings_openai_org['data'][0]['embedding'])
        vector_openai_1 = np.array(embeddings_openai_1['data'][0]['embedding'])
        vector_openai_2 = np.array(embeddings_openai_2['data'][0]['embedding'])
        
        # Calculate cosine similarity between the embedding vectors
        sim_openai_1_col.append(1 - cosine(vector_openai_org, vector_openai_1))
        sim_openai_2_col.append(1 - cosine(vector_openai_org, vector_openai_2))
        
        print('OpenAI: #{}, sent_1 similarity:{}'.format(row['sent_id'], 1 - cosine(vector_openai_org, vector_openai_1)))
        
    df_sim_openai = pd.DataFrame(zip(sent_openai_ids_col, sim_openai_1_col, sim_openai_2_col), columns=['sent_id', 'sim_openai_1', 'sim_openai_2'])    
    
    return df_sim_openai

def calculate_similarity(df):
    df_sim_usem_result = calculate_similarity_usem(df)

    df_sim_transformer_result = calculate_similarity_transformers(df)

    df_sim_openai_result = calculate_similarity_openai(df)
    # merge the USEM, Transformer and OpenAI similarity result to one DataFrame
    result = pd.merge(df_sim_usem_result, df_sim_transformer_result, on=["sent_id"])
    result = pd.merge(result, df_sim_openai_result, how="left", on=["sent_id"])
    
    return result

print('End of function')

# %%

# read data from Excel
df = pd.read_excel('./Interpreter_Evaluation.xlsx')
df = df.replace(np.nan, '')

slice_size = 2

df_sent_windows = get_sliding_sentences(df.loc[:, ['Original', 'Translation_1', 'Translation_2']], slice_size)

# calculate sentence similarity
result = calculate_similarity(df_sent_windows)

# Save result to Excel file
result.to_excel('./similarity.xlsx', sheet_name='similarity')

print('End of similarity calculation')


# %%
