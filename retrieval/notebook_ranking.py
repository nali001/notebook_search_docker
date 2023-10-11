import os
import json
from typing import List
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


class BaseNotebookRanker: 
    ''' Base class for notebook rankers. 
    Common query file and result forms. 
    Support several rankers. '''
    def __init__(self, query_file: str, result_dir: str, ranker:dict): 
        self.query_file = query_file
        self.result_dir = result_dir
        self.ranker = ranker

    def bulk_rank(self):
        pass
    
    def rank(self): 
        pass

    def _read_queries(self) -> List[dict]:
        queries = []
        with open(self.query_file, "r") as f:
            queries = json.load(f)
        return queries

    def _output_rankings(self, results: dict):
        ranker_name = self.ranker['type']
        output_dir = os.path.join(self.result_dir, ranker_name)
        os.makedirs(output_dir, exist_ok=True)
        for qid, result in results.items():
            output_file = os.path.join(output_dir, f"{qid}.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)


class NotebookRanker(BaseNotebookRanker):
    def __init__(self, notebook_dir: str, query_file: str, result_dir: str, ranker: dict):
        super().__init__(query_file, result_dir, ranker)
        self.notebook_dir = notebook_dir
        self.rank_functions = {
            "bm25": self._bm25_rank,
            "tfidf": self._tfidf_rank,
        }
        file_count = sum(1 for file in os.listdir(notebook_dir) if file.endswith('.json'))
        print(f"[{file_count}] computational notebooks")
        
        # self.ranking_contents = {
        #     'md_text': 'md_text_clean', 
        #     'code': 'code', 
        #     'code_comments': 'code_comments', 
        #     }
        
    def _read_notebooks(self) -> List[dict]:
        notebooks = []
        for filename in os.listdir(self.notebook_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.notebook_dir, filename), "r") as f:
                    nb_json = json.load(f)
                    notebooks.append(nb_json)
        return notebooks
    
    def bulk_rank(self):
        ''' Rank for all the queries
        '''
        notebooks = self._read_notebooks()
        queries = self._read_queries()
        results = {}
        for query in tqdm(queries):
            qid = query["qid"]
            text = query["text"]
            rankings = self.rank(notebooks, text)
            ranked_docs = [{"docid": docid, "score": score} for docid, score in rankings]
            results[qid] = {"qid": qid, "docs": ranked_docs}
        self._output_rankings(results)
        return True
    
    def rank(self, notebooks: List[dict], query: str) -> List[tuple]:
        ''' Rank for one query
        '''
        ranker_type = self.ranker["type"]
        ranking_contents = self.ranker["ranking_contents"]
        ranker_params = self.ranker.get("params", {})
        if ranker_type not in self.rank_functions:
            raise ValueError("Invalid ranker specified.")

        # Extract contents from specified field  
        contents = []
        for nb in notebooks: 
            combined_contents = ''
            for ranking_content in ranking_contents: 
                combined_contents += '\n' + nb[ranking_content]
            contents.append(combined_contents)
        # print(contents)
        scores = self.rank_functions[ranker_type](contents, query, **ranker_params)
        docids = [nb["docid"] for nb in notebooks]
        ranked_results = sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
        return ranked_results

    @staticmethod
    def _bm25_rank(notebooks: List[str], query: str, k1: float = 1.2, b: float = 0.75, eps: float = 0.25) -> List[float]:
        # Compute BM25 scores between query and each notebook
        tokenized_notebooks = [nb.split(" ") for nb in notebooks]
        tokenized_query = query.split(" ")
        bm25 = BM25Okapi(tokenized_notebooks, k1=k1, b=b, epsilon=eps)
        return bm25.get_scores(tokenized_query)
    
    @staticmethod
    def _tfidf_rank(self, notebooks: List[str], query: str, min_df: float = 2, max_df: float = 0.8) -> List[float]:
        # Compute TF-IDF scores between query and each notebook
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
        tfidf_matrix = vectorizer.fit_transform(notebooks)
        query_tfidf = vectorizer.transform([query])
        scores = (tfidf_matrix * query_tfidf.T).toarray()
        return scores.flatten().tolist()
    


if __name__ == '__main__': 
    '''
    python -m retrieval.notebook_ranking
    '''
    nb_ranker = NotebookRanker(
    notebook_dir = "./data/evaluation/notebooks/notebooks_contents",
    # query_file = "./data/evaluation/queries/queries.json",
    # result_dir = "./data/evaluation/results/queries/text_code", 

    query_file = "./data/evaluation/queries/evaluation_queries.json",
    result_dir = "./data/evaluation/results/evaluation_queries/code_comments", 
    ranker={
        "type": "bm25",
        "ranking_contents": ['code_comments'], 
        "params": {
        }
    })
    nb_ranker.bulk_rank()