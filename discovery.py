import json
import pandas as pd
import numpy as np
import pickle
import torch
import os
from pathlib import Path

import random
random.seed(42)
from line_profiler import profile


from sdd.pretrain import load_checkpoint, inference_on_tables
from sdd.retrieval_logger import SimpleIndexLogger

from memory_profiler import memory_usage

from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def clean_table(table, target='Rating'):
    """Clean an input table.
    """
    if target not in table:
        return table

    new_vals = []
    for val in table[target]:
        try:
            if isinstance(val, str):
                val = val.replace(',', '').replace('%', '')
            new_vals.append(float(val))
        except:
            new_vals.append(float('nan'))

    table[target] = new_vals
    return table.dropna(subset=[target])


# lm = SentenceTransformer('paraphrase-MiniLM-L6-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"
lm = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
lm.eval()

def featurize(table, target='Rating'):
    """Featurize a query table.
    """
    all_vectors = []
    for column in table:
        if column == target:
            continue
        if table[column].dtype.kind in 'if':
            all_vectors.append(np.expand_dims(table[column], axis=1))
        else:
            with torch.no_grad():
                vectors = lm.encode(list(table[column].astype('str')))
            all_vectors.append(vectors)

    return np.concatenate(all_vectors, axis=1), np.array(table[target])

def process_query_tables(query_tables):
    """Run ML on a dictionary of query tables.
    """
    for table in query_tables.values():
        N = len(table)
        table['Rating'] = (table['Rating'] - table['Rating'].min()) / (table['Rating'].max() - table['Rating'].min() + 1e-6)
        # table['Rating'] = table['Rating'] / (table['Rating'].max() + 1e-6)
        table = table.sample(frac=1.0, random_state=42)
        train = table[:N//5*4]
        test = table[N//5*4:]

        x, y = featurize(train)
        model = XGBRegressor()
        model.fit(x, y)

        x, y = featurize(test)
        y_pred = model.predict(x)
        # print(len(table), mean_squared_error(y, y_pred))
        print(mean_squared_error(y, y_pred))

@profile
def check_table_pair(table_a, vectors_a, table_b, vectors_b, method='naive', target='target'):
    """Check if two tables are joinable. Return the join result and the similarity score
    """
    best_pair = None
    max_score = -1
    target_sim = 0.0

    for col_a, vec_a in zip(table_a, vectors_a):
        norm_vec_a = np.linalg.norm(vec_a)
        if col_a == target:
            if method == 'cl':
                for col_b, vec_b in zip(table_b, vectors_b):
                    if table_a[col_a].dtype != table_b[col_b].dtype:
                        continue
                    sim = np.dot(vec_a, vec_b) / norm_vec_a / np.linalg.norm(vec_b)
                    # if sim > 0:
                    target_sim += sim
            else:
                continue
        seta = set(table_a[col_a].unique())

        for col_b, vec_b in zip(table_b, vectors_b):
            if table_a[col_a].dtype != table_b[col_b].dtype:
                continue
            setb = set(table_b[col_b].unique())
            if method == 'jaccard':
                score = len(seta.intersection(setb)) / len(seta.union(setb))
            elif method == 'cl':
                overlap = len(seta.intersection(setb)) # / len(seta.union(setb))
                # score = float(overlap >= 10) * np.dot(vec_a, vec_b) / np.linalg.norm(vec_a) / np.linalg.norm(vec_b)
                score = float(overlap) * (1.0 + np.dot(vec_a, vec_b) / norm_vec_a / np.linalg.norm(vec_b))
            elif method == 'overlap':
                score = len(seta.intersection(setb)) / len(seta)
            else:
                score = 0.0

            if score > max_score:
                max_score = score
                best_pair = col_a, col_b

    if target_sim > 0:
        max_score *= target_sim

    return best_pair, max_score


def profile_inference(base_path, tables_data_path, query_paths, query_data_path):
    datalake_tables = {}
    print("Loading checkpoint")
    ckpt_path = f"results/metadata/{base_path.stem}/model_drop_col_head_column_0.pt"
    ckpt = torch.load(ckpt_path)
    table_model, table_dataset = load_checkpoint(ckpt)

    if not Path(tables_data_path).exists():
        tot = sum(1 for _ in base_path.glob("**/*.json"))
        for p in tqdm(base_path.glob("**/*.json"), total=tot, desc="Reading metadata: "):
            with open(p, "r") as fp:
                mdata = json.load(fp)
                cnd_path = mdata["full_path"]
                cnd_hash = mdata["hash"]
                table = pd.read_parquet(cnd_path)
                if len(table) >= 50:
                    datalake_tables[cnd_hash] = table

        datalake_table_vectors = {}
        all_tables = list(datalake_tables.values())
        v_ = inference_on_tables(all_tables, table_model, table_dataset)
        for tid, v in zip(datalake_tables, v_):
            datalake_table_vectors[tid] = v

        pickle.dump((datalake_tables, datalake_table_vectors), open(tables_data_path, 'wb'))
    else:
        datalake_tables, datalake_table_vectors = pickle.load(open(tables_data_path, 'rb'))
    query_tables = {}    
    for query_table_path in query_paths:
        if query_table_path.exists():
            query_table = pd.read_parquet(query_table_path)
            query_tables[query_table_path.stem] = query_table
    query_vectors = {}
    v_ = inference_on_tables(list(query_tables.values()), table_model, table_dataset)
    for tid, v in zip(query_tables, v_):
        query_vectors[tid] = v
    pickle.dump((query_tables, query_vectors), open(query_data_path, 'wb'))

    return datalake_tables, datalake_table_vectors, query_tables, query_vectors

# @profile
def profile_query(base_table,datalake_tables, datalake_vectors, v_base_table, method):
    prepared_candidates = []
    best_similarity = -1.0
    best_pair = None
        
    subsample_keys = random.choices(list(datalake_tables.keys()), k=100)

            
    for candidate_tid in tqdm(datalake_tables, total=len(datalake_tables), desc="Candidate: ", position=1, leave=False):
    # for candidate_tid in tqdm(subsample_keys, total=100, desc="Candidate: ", position=1, leave=False):
        cand_table = datalake_tables[candidate_tid]
        vectors_cand_table = datalake_vectors[candidate_tid]

        res, similarity = check_table_pair(base_table, v_base_table,
                        cand_table, vectors_cand_table, method=method)
        if res is not None and similarity > 0:
            prepared_candidates.append({"cand_table": candidate_tid, "join_columns": res, "similarity": similarity})                    
        
        if res is not None and similarity > best_similarity:
            best_similarity = similarity
        
    return prepared_candidates

if __name__ == '__main__':
    # Set to True to profile memory. This will disable parallelism. 
    profile_memory = False
    
    # step 1: load columns and vectors
    data_lake_version = "wordnet_full"
    print(f"Working on data lake {data_lake_version}")
    case=  f"metadata/{data_lake_version}"
    base_path = Path("data/metadata", data_lake_version)

    logger = SimpleIndexLogger(
        "starmie",
        "query",
        data_lake_version=data_lake_version, 
        index_parameters={},
        log_path="results/query_logging.txt"
    )

    tables_data_path = Path(base_path, 'datalake_tables.pkl')
    query_data_path = Path(base_path, 'query_tables.pkl')
    
    query_paths = [
        "data/source_tables/yadl/movies_vote_large-yadl-depleted.parquet",
        "data/source_tables/yadl/movies_large-yadl-depleted.parquet",
        "data/source_tables/yadl/us_accidents_2021-yadl-depleted.parquet",
        "data/source_tables/yadl/us_accidents_large-yadl-depleted.parquet",
    ]
    # query_paths = [
    #     "data/source_tables/yadl/company_employees-yadl-depleted.parquet",
    #     "data/source_tables/yadl/housing_prices-yadl-depleted.parquet",
    #     "data/source_tables/yadl/movies_vote-yadl-depleted.parquet",
    #     "data/source_tables/yadl/movies-yadl-depleted.parquet",
    #     "data/source_tables/yadl/us_accidents-yadl-depleted.parquet",
    #     "data/source_tables/yadl/us_county_population-yadl-depleted.parquet",
    #     "data/source_tables/yadl/us_elections-yadl-depleted.parquet",
    # ]
    query_paths = list(map(Path, query_paths))
    
    
    # step 2: select data lake tables
    print("Running inference")
    logger.start_time("load")
    
    if profile_memory:
        mem_usage, (datalake_tables, datalake_vectors, query_tables, query_vectors) = memory_usage(
            (
                profile_inference,
                (base_path,tables_data_path, query_paths, query_data_path)
            ), timestamps=True, max_iterations=1, retval=True
        )
        logger.mark_memory(mem_usage, "inference")
    else:
        (
            datalake_tables, 
            datalake_vectors, 
            query_tables, 
            query_vectors
            ) = profile_inference(
            base_path,tables_data_path, query_paths, query_data_path
            )
    logger.end_time("load")
    
    logger.to_logfile()

    # step 4: run each data discovery method
    candidate_joins = {"cl":[], "jaccard": [], "overlap": []}
    for method in ['cl']:
    # for method in ['cl', 'jaccard', 'overlap']:
        result_tables = []
        for query_tid in tqdm(query_tables):
            tqdm.write(f"Querying {query_tid}")
            logger = SimpleIndexLogger(
                "starmie",
                "query",
                data_lake_version=data_lake_version, 
                index_parameters={},
                log_path="results/query_logging.txt"
            )

            base_table = query_tables[query_tid]
            v_base_table = query_vectors[query_tid]
            
            logger.update_query_parameters(query_tid, "")
            logger.start_time("query")
            
            if profile_memory:            
                mem_usage, prepared_candidates = memory_usage(
                    (profile_query, 
                    (base_table, datalake_tables, datalake_vectors, v_base_table, method)), 
                    timestamps=True, max_iterations=1, retval=True
                )
                logger.mark_memory(mem_usage, "query")
            else:
                prepared_candidates = profile_query(base_table, datalake_tables, datalake_vectors, v_base_table, method)
            candidates = pd.DataFrame().from_records(prepared_candidates).sort_values("similarity", ascending=False)
            out_path = Path("results", case, "starmie-%s_%s.parquet"% (method, query_tid))
            candidates.to_parquet(out_path, index=False)
            # candidates.to_csv(out_path, index=False)
            logger.end_time("query")
            logger.to_logfile()
        pickle.dump(result_tables, open('%s_%s_result_tables.pkl' % (query_tid, method), 'wb'))
