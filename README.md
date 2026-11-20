# DocCIDR

Implementation of **Zheng et al. (2018), *An effective neural model extracting document-level chemical-induced disease relations from biomedical literature*** ([Journal of Biomedical Informatics](https://doi.org/10.1016/j.jbi.2018.05.001)).

The paper’s PDF is in this repo:  
[`6-An effective neural model extracting document level chemical-induced disease relations from biomedical literature.pdf`](6-An%20effective%20neural%20model%20extracting%20document%20level%20chemical-induced%20disease%20relations%20from%20biomedical%20literature.pdf).

## Goal

Reproduce the article’s **document-level Chemical-Induced Disease (CID)** pipeline on the BioCreative V **CDR** corpus:

1. Represent PubMed abstracts with pretrained **word** and **POS** embeddings (word2vec).
2. Build the paper’s **neural model** over candidate chemical–disease concept pairs in each document.
3. Classify each pair as CID or not, and evaluate with the official CDR metrics.

## Task (from the paper)

- **Input:** a document with chemical and disease mentions (MeSH-normalized when possible).
- **Output:** document-level CID relations between chemical and disease **concepts** (not sentence-bound mention pairs only).
- **Data:** BioCreative V CDR — 500 train / 500 development / 500 test PubMed articles.

## Methods we implement

| Stage | Paper method |
| --- | --- |
| Corpus | CDR gold annotations (PubTator / BioC); optional tmChem + DNorm mentions on the test set |
| Embeddings | word2vec word vectors on PubTator bioconcepts texts + CDR (~22 GB in the paper); POS vectors on a random 1/5 of that text (~4.6 GB, 43 tags) |
| Features | Word + POS embeddings as token inputs to the neural encoder |
| Model | Document-level neural CID extractor as described in the article |
| Evaluation | Precision / recall / F1 on document-level CID relations |

## Repository layout

| Path | Role |
| --- | --- |
| Paper PDF | Source article for the methods |
| [`Proj2.ipynb`](Proj2.ipynb) | Data exploration of CDR (PubTator parsing, mention / CID stats) |
| `CDR_Data/` *(local, gitignored)* | Extracted CDR corpus |
| `CDR_Data.zip` *(local, gitignored)* | CDR archive |
| `corenlp-english-extra/` *(local, gitignored)* | Stanford CoreNLP English-extra models (POS, etc.) |

Large binaries stay off GitHub; keep them locally next to the notebooks.

## Getting started

1. Place `CDR_Data.zip` in the repo root and run the first cell of [`Proj2.ipynb`](Proj2.ipynb) to extract it.
2. (Optional) Download CoreNLP English-extra models into `corenlp-english-extra/` for POS tagging, as in the paper’s preprocessing stack.
3. (Optional, heavy) For paper-scale embeddings, use PubTator bioconcepts text + CDR; for a smaller run, train word2vec on CDR alone first.

## Status

- [x] Repo setup, paper, CDR exploration notebook  
- [ ] Preprocessing + embedding training (word / POS)  
- [ ] Neural CID model  
- [ ] Train / evaluate on CDR test  
