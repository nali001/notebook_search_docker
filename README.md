# A Dense Retrieval System and Evaluation Dataset for Scientific Computational Notebooks

The discovery and reutilization of scientific codes are crucial in many research activities. Computational notebooks have emerged as a particularly effective medium for sharing and reusing scientific codes. Nevertheless, effectively locating relevant computational notebooks is a significant challenge. First, computational notebooks encompass multi-modal data comprising unstructured text, source code, and other media, posing complexities in representing such data for retrieval purposes. Second, the absence of evaluation datasets for the computational notebook search task hampers fair performance assessments within the research community. Prior studies have either treated computational notebook search as a code-snippet search problem or focused solely on content-based approaches for searching computational notebooks. To address the aforementioned difficulties, we present DeCNR, tackling the information needs of researchers in seeking computational notebooks. Our approach leverages a fused sparse-dense retrieval model to represent computational notebooks effectively. Additionally, we construct an evaluation dataset including actual scientific queries, computational notebooks, and relevance judgments for fair and objective performance assessment. Experimental results demonstrate that the proposed method surpasses baseline approaches in terms of F1@5 and NDCG@5. The proposed system has been implemented as a web service shipped with REST APIs, allowing seamless integration with other applications and web services. 


## Folder structure

`indexing`: Build dense indexes

`retrieval`: Rank computational notebooks using different models

`evaluation`: Compute metrics for different ranking results


## Setup
Run `setup_env.sh`



## Usage
### Prepare data
Download data: 
`https://surfdrive.surf.nl/files/index.php/s/jrqqsqisewUjFxf`

Pipeline: indexing -> retrieval -> evaluation

### Indexing
Text: `indexing/faiss_indexing.ipynb`

Code: `indexing/code_faiss_indexing.ipynb`

### Retrieval
Sparse: 
`pythong -m retrieval.notebook_ranking`

Dense: `pythong -m retrieval.dense_ranking`

### Evaluation
`evaluation/compute_metrics_EQ.ipynb`


### Relevancy labels
+ 3: Perfectly relevant
+ 2: Highly relevant
+ 1: Relevant
+ 0: Irrelevant

---

## Cite our work
Li, Na, Yangjun Zhang, and Zhiming Zhao. "A Dense Retrieval System and Evaluation Dataset for Scientific Computational Notebooks." 2023 IEEE 19th International Conference on e-Science (e-Science). IEEE, 2023. 

```
@inproceedings{li2023dense,
  title={A Dense Retrieval System and Evaluation Dataset for Scientific Computational Notebooks},
  author={Li, Na and Zhang, Yangjun and Zhao, Zhiming},
  booktitle={2023 IEEE 19th International Conference on e-Science (e-Science)},
  pages={1--10},
  year={2023},
  organization={IEEE}
}
```