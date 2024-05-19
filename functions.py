import importlib.util
import requests
import re

def setup():
    def download_file(url, local_path):
        response = requests.get(url)
        with open(local_path, 'wb') as file:
            file.write(response.content)

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # URLs of the files in the GitHub repository
    word_transformer_url = 'https://raw.githubusercontent.com/pierluigic/xl-lexeme/main/WordTransformer/WordTransformer.py'
    input_example_url = 'https://raw.githubusercontent.com/pierluigic/xl-lexeme/main/WordTransformer/InputExample.py'

    # Local paths where the files will be saved
    word_transformer_path = '/tmp/WordTransformer.py'
    input_example_path = '/tmp/InputExample.py'

    # Download the files
    download_file(word_transformer_url, word_transformer_path)
    download_file(input_example_url, input_example_path)

    # Load the modules
    word_transformer_module = load_module("WordTransformer", word_transformer_path)
    input_example_module = load_module("InputExample", input_example_path)

    # Import the desired classes
    WordTransformer = word_transformer_module.WordTransformer
    InputExample = input_example_module.InputExample

    return WordTransformer, InputExample

def edit_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_position_of_similar_word(word, sentence):
    """
    Find the start and end positions of the word in the sentence that has the smallest Levenshtein distance
    to the target word. If multiple words have the same smallest distance, the first one is returned.

    Args:
        word (str): The target word to find a similar match for.
        sentence (str): The sentence to search within.

    Returns:
        tuple: A tuple containing the start and end positions (start_pos, end_pos) of the similar word.
               If no match is found, returns None.
    """
    # Find all whole words in the sentence
    matches = list(re.finditer(r'\b\w+\'?\w*\b', sentence))
    min_distance = float('inf')
    best_position = None

    # Iterate through each word in the sentence
    for match in matches:
        sentence_word = match.group(0)
        distance = edit_distance(word, sentence_word)
        
        # Update the best position if a smaller distance is found
        if distance < min_distance:
            min_distance = distance
            best_position = (match.start(), match.end())
        # If the same minimum distance is found, retain the first occurrence
        elif distance == min_distance:
            pass

    return best_position

def get_positions(sentence, word):
    """
    Finds the start and end character positions of a word in a sentence. If the word is found as a complete word,
    returns its start and end positions. If not found, returns the start and end positions of the word with the
    smallest Levenshtein edit distance from the target word.

    Args:
        sentence (str): The sentence to search within.
        word (str): The word to find the positions of.

    Returns:
        list: A list containing the start and end positions [start_pos, end_pos] of the word or the closest match.
    """
    match = re.search(rf'\b{re.escape(word)}\b', sentence)
    if match:
        start_pos = match.start()
        end_pos = match.end()
    else:
        start_pos, end_pos = find_position_of_similar_word(word, sentence)
    return [start_pos, end_pos]

def format_sentences(sentences, words):
    formatted = [InputExample(texts=sentence, positions=get_positions(sentence, word)) for sentence, word in zip(sentences, words)]
    return formatted


WordTransformer, InputExample = setup()

#################################### Plotting embeddings functions

import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS
from umap.umap_ import UMAP
from sklearn.decomposition import PCA



def project_group_and_scatter_plot_embeddings(embeddings, sentences, words, n_clusters=5, dim_reducer='mds'):
    if n_clusters > len(embeddings):
        n_clusters = len(embeddings)
    # Step 1: Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_

    # Step 2: Reduce dimensionality to 2D for visualization (using MDS in this case)
    if dim_reducer == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    elif dim_reducer == 'mds':
        reducer = MDS(n_components=2, random_state=42)
    elif dim_reducer == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif dim_reducer == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown dimensionality reducer: {dim_reducer}")

    # Step 2: Reduce dimensionality to 2D for visualization 
    embeddings_2d = reducer.fit_transform(embeddings)

    # Extract the 2D x and y coordinates
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]

    # Create scatter plot with Plotly
    fig = go.Figure()

    def split_text(text, max_line_length):
        return '<br>'.join(text[i:i+max_line_length] for i in range(0, len(text), max_line_length))

    def make_word_bold(text, word):
        """Replace all occurrences of 'word' in 'text' with the same word wrapped in HTML bold tags."""
        bold_word = f"<b>{word}</b>"
        return text.replace(word, bold_word)

    # Loop through the clusters to plot each one as a separate scatter trace
    for i in range(n_clusters):
        cluster_idx = labels == i
        # get index of sentences in original sentences list
        # get index of the embeddings in the original embeddings list
        sentences_ids = [idx for idx, val in enumerate(embeddings) if val in embeddings[cluster_idx]]

        hover_text = [f"({sent_id}) - {sentences[sent_id]}" for idx, sent_id in zip(cluster_idx, sentences_ids)]
        # split up each text in hover text to have max 100 characters per line
        hover_text = [split_text(text, 95) for text in hover_text]
        # make the word bold in each sentence
        hover_text = [make_word_bold(text, word) for text, word in zip(hover_text, words)]
        
        fig.add_trace(go.Scatter(x=x[cluster_idx], y=y[cluster_idx], mode='markers', 
                                marker=dict(size=10),
                                text=hover_text,  # Sentences for hover
                                hoverinfo="text",
                                name=f'Cluster {i+1}'))

    # Customize layout
    fig.update_layout(title=f'Embeddings of sentences projected using {dim_reducer.upper()}',
                      xaxis_title='Dimension 1',
                      yaxis_title='Dimension 2',
                      hovermode='closest',# Tooltip shows info of the closest point
                      width=600,
                      height=600,
                      autosize=False
                      )  
    # Show plot
    fig.show()

def scatter_plot_word_sentences(embeddings, sentences, words, n_clusters=2, dim_reducer='mds'):
    # print('Getting sentences')
    # sentences = get_sentences(text, word)
    sentences = [sentence.split('\t')[1] for sentence in sentences if len(sentence.split('\t')) > 1]

    # print('Embedding sentences')
    # embeddings = [embed_word_in_sentence(sentence, word) for sentence, word in zip(sentences, words)]
    # if len(embeddings) == 0:
    #     print(f'No sentences found with the word {word}')
    #     return
    # embeddings = torch.cat(embeddings)

    print('Projecting and plotting')
    project_group_and_scatter_plot_embeddings(embeddings, sentences, words, n_clusters=n_clusters, dim_reducer=dim_reducer)

    return sentences
