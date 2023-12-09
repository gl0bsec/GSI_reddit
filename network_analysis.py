import pandas as pd
import networkx as nx

def prepare_author_user_network(data):

    # Create a graph
    G = nx.Graph()

    # Define colors for each subreddit
    color_map = {
        subreddit: f"C{i}" for i, subreddit in enumerate(data['subreddit'].unique())
    }

    # Define lighter shades for sentiment colors
    light_red = "#FF9999"  # Light red for negative sentiment
    light_blue = "#9999FF"  # Light blue for positive sentiment

    # Add edges and nodes to the graph, ensuring all nodes are of type str
    for _, row in data.iterrows():
        user = str(row['user'])
        author = str(row['author'])
        subreddit = str(row['subreddit'])
        sentiment = row['vader.comment']  # Assuming 'sentiment' column exists

        # Add nodes with subreddit-based color and label
        G.add_node(user, color=color_map.get(subreddit, "C0"), label=subreddit)
        G.add_node(author, color=color_map.get(subreddit, "C0"), label=subreddit)

        # Determine edge color based on sentiment (using lighter shades)
        edge_color = light_blue if sentiment > 0 else light_red

        # Add an edge between the user and author
        G.add_edge(user, author, color=edge_color)

    # Update node sizes based on degree centrality after the graph is built
    centrality = nx.degree_centrality(G)
    for node in G.nodes():
        G.nodes[node]['size'] = centrality[node] * 100  # Adjust the multiplier as needed for visibility

    return G, color_map

# Usage:
# G, color_map = prepare_author_user_network('path_to_your_file.csv')
