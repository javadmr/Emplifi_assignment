
# Topic Hierarchy Clustering and Visualization

This project clusters topics based on their co-occurrence in posts and visualizes the hierarchy using word clouds. The solution employs advanced machine learning techniques, including dimensionality reduction, clustering, and hierarchical data structures, to generate a JSON-compatible output and visualization.

---

## Project Structure

The project is organized as follows:

```
├── solution_small_data.ipynb                 # Jupyter Notebook for processing and analyzing small datasets
├── solution_small_data.py                    # Python script equivalent of the notebook for small datasets
├── solution_big_data.ipynb                   # Jupyter Notebook optimized for large datasets
├── solution_big_data.py                      # Python script for large dataset processing
├── posts_with_topics.json                    # Input JSON file containing post data with topics
├── small_data_topics_hierarchy.json          # Output JSON file with the clustered hierarchy for the small data
├── big_data_topics_hierarchy.json            # Output JSON file with the clustered hierarchy for the big data
├── solution_documentation.docx               # Detailed documentation of the project
├── readme.md                                 # This README file
├── requirements.txt                          # List of dependencies for the project
```

---

## Installation

To run this project, ensure you have Python installed and follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/javadmr/Emplifi_assignment.git
   cd Emplifi_assignment
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### For Small Datasets
- Use the Jupyter notebook `solution_small_data.ipynb` for interactive analysis or the Python script `solution_small_data.py` for batch processing.

```bash
python solution_small_data.py
```

### For Large Datasets
- Use the Jupyter notebook `solution_big_data.ipynb` or the Python script `solution_big_data.py` optimized for handling big data.

```bash
python solution_big_data.py
```

### Input File
The input file `posts_with_topics.json` should be structured as:
```json
[
  {
    "post_id": "1",
    "topics": ["topic1", "topic2", "topic3"]
  },
  ...
]
```

### Output File
The processed output hierarchy will be saved as `small_data_topics_hierarchy.json` and `big_data_topics_hierarchy.json`:
```json
{
  "topics": [...],
  "children": [...]
}
```

---

## Features

- **Dimensionality Reduction**: PCA and UMAP are used to reduce high-dimensional co-occurrence matrices.
- **Hierarchical Clustering**: Creates a tree structure of topics using agglomerative clustering.
- **Visualization**: Word clouds are generated for the clusters to understand topic distribution visually.
- **Scalability**: Separate scripts for small and large datasets ensure efficient handling of varying data sizes.

---

## Project Documentation

Refer to `solution_documentation.docx` for:
- Detailed explanation of algorithms and methodology.
- Examples of input and output.
- Insights into performance and scalability.

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contact

For further questions or contributions, please contact:
- [Javad M.Rad](mailto:javad_mohamadi_rad@yahoo.com)
- [GitHub Profile](https://github.com/javadmr)
