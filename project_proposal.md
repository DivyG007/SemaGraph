# Project Proposal  
**Divyanshu Giri**  
**2024114009**  
**Course: Computational Linguistics 2**

---

## P7: Exploring Graph-based Models for Sentence-Level Semantic Representation

### Research Question
**How effectively can graph-based sentence representations capture semantic similarity compared to statistical vector-based representations such as Latent Semantic Analysis (LSA)?**

---

## Methods

This project explores the use of **graph-based models** in semantic parsing by constructing and analyzing **sentence-level meaning graphs**.

### 1. Parsing and Graph Construction
- Sentences will be parsed using a dependency parser (**spaCy** or **Stanza**).  
- Parsed relations will be converted into **graph structures** using **NetworkX**:  
  - **Nodes** represent lexical items  
  - **Edges** represent syntactic or semantic dependencies  

### 2. Similarity Measures

#### Graph-based similarity
- **Graph Edit Distance**
- **Jaccard Index**

#### LSA-based similarity
- Sentence vectors will be created using an LSA pipeline.
- Similarity will be measured using **cosine similarity** of reduced-dimensional vectors.

### 3. Additional Comparison (Optional)
If time permits, neural embedding methods will also be evaluated:
- **Word2Vec**
- **GloVe**

### 4. Evaluation
Similarity scores from:
- Graph-based models  
- LSA models  
- Embedding-based models  

will be compared with **human similarity judgments** from benchmark datasets.

---

## Datasets

- **STS Benchmark Dataset**  
  Contains sentence pairs annotated with human similarity scores.

- **SICK Dataset** (optional)  
  Additional validation set involving compositional semantics.

---

## Key Research Paper References

1. **Banarescu, L. et al. (2013).**  
   *Abstract Meaning Representation for Natural Language.* Proceedings of the 7th Linguistic Annotation Workshop.

2. **Zhang, S. et al. (2019).**  
   *AMR Parsing as Sequence-to-Graph Transduction.* Proceedings of ACL 2019.

3. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).**  
   *Efficient Estimation of Word Representations in Vector Space.* Proceedings of ICLR 2013.

4. **Landauer, T. K., & Dumais, S. T. (1997).**  
   *A Solution to Plato’s Problem: The Latent Semantic Analysis Theory of the Acquisition, Induction, and Representation of Knowledge.* Psychological Review.

---

## Expected Outcome

A comparative study demonstrating:
- Differences in semantic information captured by **graph-based** vs **LSA-based** models  
- Optional comparison with **neural embedding models**  
- Insights supported by:
  - Quantitative evaluation  
  - Visualizations of sentence meaning structures  

