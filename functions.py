import re
from IPython.display import display, HTML
import plotly.graph_objects as go
import ipywidgets as widgets
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP

def split_text(text, max_line_length):
    return '<br>'.join(text[i:i + max_line_length] for i in range(0, len(text), max_line_length))

def make_word_bold(text, word):
    return text.replace(word, f"<b>{word}</b>")

def project_group_and_scatter_plot_embeddings_interactive(
    embeddings, sentences, words, n_clusters=5, dim_reducer='mds'
):
    if n_clusters > len(embeddings):
        n_clusters = len(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_

    # Dimensionality reduction
    reducer = {
        'umap': UMAP(n_components=2, random_state=42),
        'mds': MDS(n_components=2, random_state=42),
        'tsne': TSNE(n_components=2, random_state=42),
        'pca': PCA(n_components=2)
    }.get(dim_reducer)
    if reducer is None:
        raise ValueError(f"Unknown dimensionality reducer: {dim_reducer}")

    embeddings_2d = reducer.fit_transform(embeddings)
    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]

    label_colors = ["green", "red", "blue", "orange", "purple"]
    current_label = [0]
    point_labels = [-1] * len(sentences)
    click_groups = [[] for _ in label_colors]

    hover_text = [make_word_bold(split_text(s, 95), w) for s, w in zip(sentences, words)]

    scatter_trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=[10] * len(sentences), color=["grey"] * len(sentences)),
        text=hover_text,
        hoverinfo="text"
    )
    fig = go.FigureWidget([scatter_trace])
    fig.update_layout(
        title=f'Embeddings of sentences projected using {dim_reducer.upper()}',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        hovermode='closest',
        width=700,
        height=700
    )

    output = widgets.Output()
    scatter_trace = fig.data[0]

    def on_click(trace, points, selector):
        c = list(scatter_trace.marker.color)
        for i in points.point_inds:
            point_labels[i] = current_label[0]
            c[i] = label_colors[current_label[0]]
        with fig.batch_update():
            scatter_trace.marker.color = c

    scatter_trace.on_click(on_click)

    def next_label(_):
        current_label[0] = (current_label[0] + 1) % len(label_colors)
        with output:
            print(f"Switched to label {current_label[0]} ({label_colors[current_label[0]]})")

    def finish_labeling(_):
        for label in range(len(label_colors)):
            group = [sentences[i] for i, lbl in enumerate(point_labels) if lbl == label]
            click_groups[label] = group

        with output:
            print("✅ Sentence groups by label:")
            for i, group in enumerate(click_groups):
                color = label_colors[i]
                display(HTML(
                    f"<div style='color:{color}; font-weight: bold; margin-top:10px;'>"
                    f"Label {i} ({color}): {len(group)} sentence(s):</div>"
                ))
                for sent in group:
                    print(sent)

            print("\n📋 Full result (list of lists):")
            for i, group in enumerate(click_groups):
                print(f"[Label {i}] {group}")

    def save_sentences(_):
        saved_sentences = pd.DataFrame(columns=['lemma','sentence','sense','start','end'])
        word = words[0]
        for i, group in enumerate(click_groups):
            for sent in group:
                start, end = get_positions(sent, word)
                saved_sentences.loc[len(saved_sentences)] = [word, sent, i, start, end]


        count = 1
        # if saved_sentences/ doesn't exist, create it
        if not os.path.exists('saved_sentences'):
            os.makedirs('saved_sentences')
        while os.path.exists(f'saved_sentences/{word}_labelled_sentences{count}.csv'):
            count += 1
        filename = f'saved_sentences/{word}_labelled_sentences{count}.csv' if count > 1 else f'saved_sentences/{word}_labelled_sentences.csv'

        saved_sentences.to_csv(filename, index=False)
        with output:
            print(f"💾 Saved to {filename}")

    btn_save = widgets.Button(description="Save Sentences", button_style='warning')
    btn_save.on_click(save_sentences)
    btn_next = widgets.Button(description="Next Label", button_style='info')
    btn_finish = widgets.Button(description="Finish Labelling", button_style='success')
    btn_next.on_click(next_label)
    btn_finish.on_click(finish_labeling)

    display(widgets.HBox([btn_next, btn_finish, btn_save]))
    display(fig, output)

    with output:
        print(f"💡 Click to label. Current label: {current_label[0]} ({label_colors[0]})")

    return fig


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
