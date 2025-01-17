# %% [markdown]
# ## Solution of the home assignment for Emplifi
# 
# ### Javad M.Rad
# ### javad_mohamadi_rad@yahoo.com
# ### +420 721024504

# %% [markdown]
# # 1. Environment Setup and loading libraries

# %%
import json
import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.neighbors import NearestNeighbors

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %% [markdown]
# # 2. Data Loading and Preprocessing

# %%
# Load data
current_folder = os.getcwd()  # Get the current working directory
assignment_folder = os.path.join(current_folder, "assignment")  # Navigate to the 'assignment' subfolder
input_json_path = os.path.join(assignment_folder, "posts_with_topics.json")  # Full path to the JSON file
with open(input_json_path, 'r', encoding='utf-8') as f:
    posts = json.load(f)

# Collect all unique topics
all_topics = set()
for post in posts:
    for t in post["topics"]:
        all_topics.add(t)

all_topics = list(all_topics)
topic_index = {topic: i for i, topic in enumerate(all_topics)}
num_topics = len(all_topics)

print('Number of unique topics: ', num_topics)

# %% [markdown]
# # 3. Creating Sparse Co-occurrence Matrix, Dimensionality Reduction with PCA and UMAP
# A common way to measure how closely topics are related is to compute Co-occurrence Matrix that is, how often they appear together.

# %%
# Create a sparse co-occurrence matrix
data, row, col = [], [], []
for post in posts:
    ts = post["topics"]
    indices = [topic_index[t] for t in ts]
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            row.append(indices[i])
            col.append(indices[j])
            data.append(1)

co_matrix_sparse = coo_matrix((data, (row, col)), shape=(num_topics, num_topics))
co_matrix_sparse = co_matrix_sparse + co_matrix_sparse.T

# Calculate topic frequency
topic_frequency = np.array(co_matrix_sparse.sum(axis=1)).flatten()


# Dimensionality Reduction with PCA
co_matrix_dense = co_matrix_sparse.toarray()
pca = PCA(random_state=42, n_components=0.99) # Least amount of variance to meet
reduced_matrix = pca.fit_transform(co_matrix_dense)


# Dimensionality Reduction with UMAP
umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=50, random_state=42)
umap_embedding = umap.fit_transform(co_matrix_dense)

# Quick checks
print("Co-occurrence sparse matrix shape:", co_matrix_sparse.shape)
# print("Example row (first topic):", co_matrix_sparse[0, :])

print("\nCo-occurrence matrix shape with PCA dimensionality reduction:", reduced_matrix.shape)
print("Selected components with PCA:", pca.components_.tolist())
# print("Example row (first topic):", reduced_matrix[0, :])

print("\nCo-occurrence matrix shape with UMAP dimensionality reduction:", umap_embedding.shape)
print("Selected components with UMAP:", umap_embedding.tolist())
# print("Example row (first topic):", umap_embedding[0, :])

# %% [markdown]
# # 4. Hierarchical Clustering
# We'll define a recursive function that:
# 1) Clusters the given list of topic indices (topic_indices).
# 2) Splits them into sub-clusters if they exceed certain size / depth.
# 3) Returns a hierarchical tree.

# %%
# Hierarchical Clustering
def hierarchical_clustering(topic_indices, umap_embedding, depth=0, max_depth=3, min_cluster_size=5, n_neighbors=10):
    """
    Recursively clusters a subset of topics using hierarchical clustering with a UMAP embedding.

    Args:
        topic_indices (list[int]): A list of indices representing the topics to be clustered at the current node.
        umap_embedding (np.array): A 2D array of UMAP-embedded points, where each row corresponds to a topic.
        depth (int, optional): The current depth of recursion. Default is 0.
        max_depth (int, optional): The maximum number of recursive levels in the hierarchy. Default is 3.
        min_cluster_size (int, optional): The minimum size of a cluster to be further subdivided. Default is 5.
        n_neighbors (int, optional): The number of nearest neighbors to consider when calculating distances. Default is 10.

    Returns:
        dict: A hierarchical tree structure representing clusters and their subclusters:
            - "topics" (list[int]): Indices of topics in the current cluster.
            - "subclusters" (list[dict]): List of child cluster dictionaries with the same structure.

    What it Does:
        - Checks base conditions for recursion (e.g., maximum depth reached or cluster size too small).
        - Uses the UMAP embedding to calculate distances between points in the current cluster.
        - Applies `AgglomerativeClustering` to cluster the topics based on their distances.
        - Groups topics into subclusters and recursively calls itself for each subcluster.
        - Constructs and returns a hierarchical tree where each node contains:
            - Topics in the current cluster.
            - Recursively constructed subclusters.
    """
    if depth >= max_depth or len(topic_indices) < min_cluster_size:
        return {
            "topics": topic_indices,
            "subclusters": []
        }

    nn = NearestNeighbors(n_neighbors=min(len(topic_indices), n_neighbors))
    nn.fit(umap_embedding[topic_indices])
    distances, _ = nn.kneighbors(umap_embedding[topic_indices])

    cluster_model = AgglomerativeClustering(
        metric='euclidean',
        linkage='average',
        n_clusters=min(5, len(topic_indices))
    )
    labels = cluster_model.fit_predict(distances)

    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_dict[label].append(topic_indices[i])

    subclusters = []
    for _, child_topic_indices in clusters_dict.items():
        child_cluster = hierarchical_clustering(
            topic_indices=child_topic_indices,
            umap_embedding=umap_embedding,
            depth=depth + 1,
            max_depth=max_depth,
            min_cluster_size=min_cluster_size,
            n_neighbors=n_neighbors
        )
        subclusters.append(child_cluster)

    return {
        "topics": topic_indices,
        "subclusters": subclusters
    }

root_cluster = hierarchical_clustering(
    topic_indices=list(range(num_topics)),
    umap_embedding=umap_embedding,
    depth=0,
    max_depth=3,
    min_cluster_size=5,
    n_neighbors=10
)

# %% [markdown]
# # 5. Building a Tree Structure and Saving the Hierarchy as JSON

# %%
# Building a Tree Structure
def build_output_tree(cluster_node, all_topics, topic_frequency):
    """
    Transform the hierarchical clustering node into a JSON-ready structure with:
    - 'topics': a list of { "topic": str, "frequency": int }
    - 'children': a list of subclusters
    
    Args:
        cluster_node (dict): A node from hierarchical_clustering
        all_topics (list[str]): Master list of topic strings
        topic_frequency (np.array): Frequencies of each topic index

    Returns:
        dict: { "topics": [...], "children": [...] }
    """
    topic_data = []
    for idx in cluster_node["topics"]:
        topic_data.append({
            "topic": all_topics[idx],
            "frequency": int(topic_frequency[idx])
        })

    children = []
    for sub_node in cluster_node["subclusters"]:
        children.append(build_output_tree(sub_node, all_topics, topic_frequency))

    return {
        "topics": topic_data,
        "children": children
    }

hierarchy_json = build_output_tree(root_cluster, all_topics, topic_frequency)

# Save as JSON
output_json_path = 'big_data_topics_hierarchy.json'
print(f"Saving hierarchy to {output_json_path}")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(hierarchy_json, f, indent=2, ensure_ascii=False)

print("Hierarchy successfully saved!")

# %% [markdown]
# # 6. Visualization with Word Cloud

# %%
# Visualization - Word Cloud
def generate_word_cloud(topic_data, title=None):
    """
    Generate a word cloud visualization from a list of topics and their frequencies.

    Args:
        topic_data (list[dict]): A list of dictionaries, where each dictionary represents a topic and its frequency.
                                 Example: [{"topic": "Politics", "frequency": 20}, {"topic": "Economy", "frequency": 15}]
        title (str, optional): The title of the word cloud visualization. Default is None.

    What it Does:
        - Converts the input `topic_data` into a dictionary of frequencies.
        - Uses the WordCloud library to generate a word cloud from these frequencies.
        - Displays the word cloud using Matplotlib.
        - If a title is provided, it adds the title above the word cloud.

    Returns:
        None: Displays the word cloud plot.
    """
    frequencies = {topic["topic"]: topic["frequency"] for topic in topic_data}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.show()

def display_cluster_wordcloud(cluster_node, all_topics, topic_frequency, depth=0):
    """
    Recursively generate word clouds for clusters in a hierarchical structure.

    Args:
        cluster_node (dict): A dictionary representing a node in the cluster hierarchy.
                             Contains:
                             - "topics": List of topic indices in the current cluster.
                             - "subclusters": List of child cluster dictionaries (each in the same format).
        all_topics (list[str]): A list of all topic names, where the index corresponds to the topic ID.
                                Example: ["Politics", "Economy", "Science"]
        topic_frequency (np.array): An array of frequencies for each topic, where index matches the topic ID.
                                    Example: [20, 15, 10]
        depth (int, optional): The depth of the current cluster in the hierarchy. Default is 0.

    What it Does:
        - Extracts the topics and their frequencies for the current cluster using the indices in `cluster_node["topics"]`.
        - Generates a word cloud for the current cluster with a title that indicates its depth.
        - Recursively calls itself for each subcluster in `cluster_node["subclusters"]` to generate word clouds for deeper levels.

    Returns:
        None: Displays the word cloud plots for the current cluster and all its subclusters.
    """
    # Get the topics for the current cluster
    topic_data = [
        {"topic": all_topics[idx], "frequency": int(topic_frequency[idx])}
        for idx in cluster_node["topics"]
    ]
    
    # Generate title based on the depth and primary topic (if available)
    if topic_data:
        title = f"Cluster at Depth {depth} - {topic_data[0]['topic']}"
    else:
        title = f"Cluster at Depth {depth}"

    # Generate word cloud for the current cluster
    generate_word_cloud(topic_data, title=title)

    # Recursively generate word clouds for subclusters
    for sub_node in cluster_node["subclusters"]:
        display_cluster_wordcloud(sub_node, all_topics, topic_frequency, depth=depth + 1)

# Initial Word Cloud for All Topics
print("Displaying Word Cloud for All Topics")
generate_word_cloud(
    [{"topic": all_topics[i], "frequency": topic_frequency[i]} for i in range(num_topics)], 
    title="All Topics Word Cloud"
)

# Recursive Word Clouds for Clusters
print("Displaying Word Clouds for Clusters")
display_cluster_wordcloud(root_cluster, all_topics, topic_frequency)


