from typing import List
from tqdm import tqdm
import pandas as pd

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

from .notebook_ranking import BaseNotebookRanker


class DenseNotebookRanker(BaseNotebookRanker): 
    ''' Ranking notebooks using dense retrieval models
    Dependent on pre-built indexes. 
    '''
    def __init__(self, index_dir: str, query_file: str, result_dir: str, ranker: dict, k: int=10): 
        super().__init__(query_file, result_dir, ranker)
        self.k = k
        self.index_dir = index_dir

        self.document_store = None
        self.embedding_model = self.ranker['embedding_model']

    def _load_faiss_index(self) -> FAISSDocumentStore:
        ''' Load pre-built index to document_store
        There must be `faiss.db`, index.faiss` and `config.json`
        '''
        index_path=f"{self.index_dir}/{self.embedding_model[0]}/index.faiss"
        config_path=f"{self.index_dir}/{self.embedding_model[0]}/config.json"
        document_store = FAISSDocumentStore.load(index_path=index_path, config_path=config_path)

        if document_store: 
            print(f"Loaded index from {index_path}")
        # Check if the DocumentStore is loaded correctly
        self.document_store = document_store


    def bulk_rank(self): 
        self._load_faiss_index()
        retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=self.embedding_model[1],
        )
        queries = self._read_queries()
        results = {}
        for query in tqdm(queries):
            qid = query["qid"]
            text = query["text"]
            ranked_docs = self.rank(retriever, text, self.k)
            results[qid] = {"qid": qid, "docs": ranked_docs}
        self._output_rankings(results)
        return True


    @ staticmethod
    def rank(retriever, query, k: int=10) -> List[tuple]: 
        ''' Retrieve top k notebooks and output ranking scores.'''
        retrieved_docs = retriever.retrieve(query=query, top_k=k*2)

        docids = []
        scores = []
        for doc in retrieved_docs: 
            docids.append(doc.meta['name'])
            scores.append(doc.score)

        # Create a sample dataframe
        df = pd.DataFrame({'docid': docids, 'score': scores})

        # Group the scores by ID and apply max pooling
        max_pooled_scores = df.groupby('docid')['score'].max()

        # Sort the max pooled scores in descending order and select the top k records
        top_k_scores = max_pooled_scores.sort_values(ascending=False).head(k)

        # Create a list of dictionaries to show the top k scores with their corresponding document IDs
        output_list = []
        for docid, score in top_k_scores.items():
            output_dict = {'docid': docid, 'score': score}
            output_list.append(output_dict)

        # Output the top k scores with their corresponding document IDs as a list of dictionaries
        return output_list
    


def retrieve_text(index_dir, model_index, query_set):
    
    EMBEDDING_MODELS = [("model1", "sentence-transformers/multi-qa-mpnet-base-dot-v1"),
                    ("model2", "sentence-transformers/all-mpnet-base-v2")]
    
    embedding_model = EMBEDDING_MODELS[model_index]
    nb_ranker = DenseNotebookRanker(
    index_dir = index_dir, 
    query_file = f"./data/evaluation/queries/{query_set}.json",
    result_dir = f"./data/evaluation/results/{query_set}/text", 
    ranker={
        "type": embedding_model[0],
        "embedding_model": embedding_model, 
        "params": {
        }
    }, 
    k = 50)
    
    nb_ranker.bulk_rank()

def retrieve_code(index_dir, model_index, query_set):
    EMBEDDING_MODELS = [("model1", "microsoft/codebert-base"), 
                    ("model2", "flax-sentence-embeddings/st-codesearch-distilroberta-base"), 
                    ("model3", "sentence-transformers/multi-qa-mpnet-base-dot-v1"),
                    ("model4", "sentence-transformers/all-mpnet-base-v2")]
    embedding_model = EMBEDDING_MODELS[model_index]
    nb_ranker = DenseNotebookRanker(
    index_dir = index_dir, 
    query_file = f"./data/evaluation/queries/{query_set}.json",
    result_dir = f"./data/evaluation/results/{query_set}/code", 
    ranker={
        "type": embedding_model[0],
        "embedding_model": embedding_model, 
        "params": {
        }
    }, 
    k = 50)
    
    nb_ranker.bulk_rank()


def retrieve_text_code(index_dir, model_index, query_set):
    EMBEDDING_MODELS = [("model1", "sentence-transformers/multi-qa-mpnet-base-dot-v1"),
                    ("model2", "sentence-transformers/all-mpnet-base-v2")]
    embedding_model = EMBEDDING_MODELS[model_index]
    nb_ranker = DenseNotebookRanker(
    index_dir = index_dir, 
    query_file = f"./data/evaluation/queries/{query_set}.json",
    result_dir = f"./data/evaluation/results/{query_set}/text_code", 
    ranker={
        "type": embedding_model[0],
        "embedding_model": embedding_model, 
        "params": {
        }
    }, 
    k = 50)
    
    nb_ranker.bulk_rank()
    
if __name__ == '__main__': 
    # retrieve_text(0)
    # retrieve_text(index_dir = './faiss_indexes_512', model_index = 1, query_set = 'evaluation_queries')
    retrieve_code(index_dir = './code_faiss_indexes_512', model_index = 3, query_set = 'evaluation_queries')
    # retrieve_text_code(index_dir = './text_code_faiss_indexes_512', model_index = 1, query_set = 'evaluation_queries')

    # retrieve_code(0)
