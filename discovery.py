import pandas as pd
import numpy as np
import pickle
import torch
import os
from pathlib import Path


from sdd.pretrain import load_checkpoint, inference_on_tables

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


lm = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# lm = SentenceTransformer('paraphrase-MiniLM-L6-v2').to('cuda')
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


def check_table_pair(table_a, vectors_a, table_b, vectors_b, method='naive', target='Rating'):
    """Check if two tables are joinable. Return the join result and the similarity score
    """
    best_pair = None
    max_score = -1
    target_sim = 0.0

    for col_a, vec_a in zip(table_a, vectors_a):
        if col_a == target:
            if method == 'cl':
                for col_b, vec_b in zip(table_b, vectors_b):
                    if table_a[col_a].dtype != table_b[col_b].dtype:
                        continue
                    sim = np.dot(vec_a, vec_b) / np.linalg.norm(vec_a) / np.linalg.norm(vec_b)
                    # if sim > 0:
                    target_sim += sim
            else:
                continue
        seta = set(table_a[col_a])

        for col_b, vec_b in zip(table_b, vectors_b):
            if table_a[col_a].dtype != table_b[col_b].dtype:
                continue
            setb = set(table_b[col_b])
            if method == 'jaccard':
                score = len(seta.intersection(setb)) / len(seta.union(setb))
            elif method == 'cl':
                overlap = len(seta.intersection(setb)) # / len(seta.union(setb))
                # score = float(overlap >= 10) * np.dot(vec_a, vec_b) / np.linalg.norm(vec_a) / np.linalg.norm(vec_b)
                score = float(overlap) * (1.0 + np.dot(vec_a, vec_b) / np.linalg.norm(vec_a) / np.linalg.norm(vec_b))
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


if __name__ == '__main__':
    # step 1: load columns and vectors
    case=  "yadl/binary_update"
    base_path = Path("data/", case)

    # step 2: select data lake tables
    if os.path.exists('datalake_tables.pkl'):
    # if False:
        tables, table_vectors = pickle.load(open('datalake_tables.pkl', 'rb'))
        query_tables, query_vectors = pickle.load(open('query_tables.pkl', 'rb'))
    else:
        tables = {}
        tot = sum(1 for _ in base_path.glob("**/*.parquet"))
        for t_path in tqdm(base_path.glob("**/*.parquet"), total=tot):
            # get table length
            table = pd.read_parquet(t_path)
            if len(table) >= 50:
                tables[t_path.stem] = table

        table_vectors = {}
        ckpt_path = "results/%s/model_drop_col_head_column_0.pt" % case
        ckpt = torch.load(ckpt_path)
        table_model, table_dataset = load_checkpoint(ckpt)
        all_tables = list(tables.values())
        vectors = inference_on_tables(all_tables, table_model, table_dataset)
        for tid, v in zip(tables, vectors):
            table_vectors[tid] = v

        pickle.dump((tables, table_vectors), open('datalake_tables.pkl', 'wb'))
    
        query_table_path = Path("data/source_tables/yadl/company_employees-yadl-depleted.parquet")  
        query_table = pd.read_parquet(query_table_path)
        query_tables = {
            query_table_path.stem: query_table
        }
        query_vectors = {}
        vectors = inference_on_tables(list(query_tables.values()), table_model, table_dataset)
        for tid, v in zip(query_tables, vectors):
            query_vectors[tid] = v
        pickle.dump((query_tables, query_vectors), open('query_tables.pkl', 'wb'))


    # step 3: select query tables
    to_query_tables = {}
    total_rows = 0
    for tid, table in tables.items():
        table = clean_table(table)
        if len(table) >= 200:
            to_query_tables[tid] = table
            total_rows += len(table)


    # step 4: run each data discovery method
    candidate_joins = {"cl":[], "jaccard": [], "overlap": []}
    for method in ['cl']:
    # for method in ['none', 'cl', 'jaccard', 'overlap']:
        result_tables = []
        for tid_a in tqdm(query_tables):
            best_table = query_tables[tid_a]
            if method == 'none':
                candidate_joins[method][tid_a] = best_table
                continue

            best_similarity = -1.0
            best_pair = None
            table_a = query_tables[tid_a]
            vectors_a = query_vectors[tid_a]

            for tid_b in tables:
                table_b = tables[tid_b]
                vectors_b = table_vectors[tid_b]

                res, similarity = check_table_pair(table_a, vectors_a,
                                table_b, vectors_b, method=method)
                if res is not None and similarity > 0:
                    candidate_joins[method].append({"cand_table": tid_b, "join_columns": res, "similarity": similarity})                    
                
                if res is not None and similarity > best_similarity:
                    best_similarity = similarity
                    best_table = table_b
                    best_pair = res
            candidates = pd.DataFrame().from_records(candidate_joins[method]).sort_values("similarity", ascending=False)
            out_path = Path("results", case, "query-results_%s_%s.parquet"% (method, tid_a))
            candidates.to_parquet(out_path, index=False)
            # candidates.to_csv(out_path, index=False)
            if best_similarity >= 0:
                result_tables.append({"cand_table": best_table, "best_pair": best_pair, "similarity": best_similarity})
        pickle.dump(result_tables, open('%s_joined_tables.pkl' % method, 'wb'))
        # process_query_tables(result_tables)
