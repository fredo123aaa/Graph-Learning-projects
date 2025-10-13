# Graph Learning Projects 🎯

Projects developed during Graph Learning courses.  
This repository demonstrates applications of various graph algorithms and learning methods, such as graph embeddings, diffusion processes, graph neural networks (GNNs), clustering on graphs, PageRank, and graph energy concepts.

---

## Table of Contents

- [Motivation](#motivation)  
- [Projects Overview](#projects-overview)  
- [Installation & Requirements](#installation--requirements)  
- [Usage](#usage)  
- [Folder / Notebook Descriptions](#folder--notebook-descriptions)  
- [Results & Visualizations](#results--visualizations)  
- [Future Directions](#future-directions)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Motivation

Graph-structured data appears in many domains — social networks, biological networks, knowledge graphs, transportation networks, etc.  
Through coursework and hands-on projects, this repo aims to:

- Explore core algorithms for graphs (e.g. PageRank, diffusion)  
- Learn representation learning on graphs (embeddings, GNNs)  
- Investigate clustering and community detection in graph settings  
- Analyze graph properties like energy or spectral features  
- Compare methods and visualize results  

---

## Projects Overview

This repository currently contains notebooks for:

- **Pagerank_Fredo_Alejos.ipynb** — PageRank algorithm and variations  
- **Clustering_Fredo_Alejos.ipynb** — Clustering / community detection on graphs  
- **Diffusion_Alejos_Arrieta_Fredo.ipynb** — Diffusion processes on graphs  
- **Embedding_Fredo_Alejos_Arrieta.ipynb** — Graph embedding techniques (e.g. node2vec, spectral embeddings)  
- **GNN_Alejos_Arrieta_Fredo.ipynb** — Graph neural network experiments  
- (And possibly others as the course progresses)  

Each notebook includes theoretical background, code, experiments, and visualizations.

---

## Installation & Requirements

To run the notebooks and experiments, you’ll need:

- Python (3.7+)  
- Standard data / ML libraries, e.g.  
  - `numpy`  
  - `pandas`  
  - `networkx`  
  - `scipy`  
  - `matplotlib`, `seaborn`  
  - `scikit-learn`  
  - `torch` / `tensorflow` / `dgl` / `pyg` (depending on GNN implementation)  
- Jupyter (or JupyterLab) environment  

You can install dependencies with:

```bash
pip install numpy pandas networkx scipy matplotlib seaborn scikit-learn jupyter torch
```

(or adjust according to your GNN framework of choice).

If you maintain a `requirements.txt`, you can use:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/fredo123aaa/Graph-Learning-projects.git
   cd Graph-Learning-projects
   ```

2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open one of the notebooks (e.g. `GNN_Alejos_Arrieta_Fredo.ipynb`) and run cells sequentially.

4. You can modify:

   - Graph datasets used  
   - Model architectures (layers, hidden dims, aggregation functions)  
   - Hyperparameters (learning rate, epochs, regularization)  
   - Experiment settings (noise, graph perturbations, feature sets)  

---

## Folder & Notebook Descriptions

| Notebook | Topic / Focus | Key Techniques |
|---|---|---|
| `Pagerank_Fredo_Alejos.ipynb` | Ranking importance of nodes | PageRank, personalized PageRank, damping factor |
| `Clustering_Fredo_Alejos.ipynb` | Community detection / graph clustering | Spectral clustering, modularity, clustering metrics |
| `Diffusion_Alejos_Arrieta_Fredo.ipynb` | Diffusion on graphs | Heat kernel diffusion, random walk diffusion |
| `Embedding_Fredo_Alejos_Arrieta.ipynb` | Node embeddings | Spectral embeddings, node2vec, t-SNE visualization |
| `GNN_Alejos_Arrieta_Fredo.ipynb` | Graph neural networks | GCN, GraphSAGE, message passing, class prediction |

You may expand this as new notebooks are added.

---

## Results & Visualizations

Inside each notebook you’ll find:

- Graph visualizations (nodes, edges, partitions)  
- Embedding plots (2D / 3D projections)  
- Performance metrics (classification accuracy, clustering scores)  
- Comparison tables across algorithms  

If you like, we can also add sample result images to the README to showcase highlights.

---

## Future Directions

Some ideas to further this work:

- Apply to real-world datasets (e.g. social networks, citation networks)  
- Extend to **heterogeneous graphs** or **knowledge graphs**  
- Implement **graph contrastive learning**  
- Explore **dynamic graphs / temporal graph models**  
- Combine with **explainability** (why a GNN makes a prediction)  
- Benchmark across many graph learning libraries  

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create a feature branch (e.g. `feature/new-notebook`)  
3. Make your additions / modifications  
4. Commit and push  
5. Open a Pull Request, describing the change  

Please ensure your notebooks run end-to-end and include explanations/comments for clarity.

---

## License

Specify the license under which this work is shared (e.g. MIT, Apache 2.0).  
If you already have a `LICENSE` file, you can reference it here:

```
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
```
