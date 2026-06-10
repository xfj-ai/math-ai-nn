# Backprop visualization
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_backprop_graph():
    G = nx.DiGraph()
    pos = {}

    nodes = [
        ("x", 0, 0), ("w", 0, -1), ("b", 0, -2),
        ("u", 1, -0.5), ("y", 2, -0.5), ("t", 2, -1.5), ("L", 3, -0.5),
    ]
    for name, layer, idx in nodes:
        G.add_node(name)
        pos[name] = (layer, idx)

    edges_fwd = [("x","u"),("w","u"),("b","u"),("u","y"),("y","L"),("t","L")]
    edges_bwd = [("L","y"),("y","u"),("u","x"),("u","w"),("u","b")]

    plt.figure(figsize=(12, 6))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=14)
    nx.draw_networkx_edges(G, pos, edgelist=edges_fwd, edge_color="blue",
                          arrows=True, arrowstyle="->", width=2, label="forward")
    nx.draw_networkx_edges(G, pos, edgelist=edges_bwd, edge_color="red",
                          arrows=True, arrowstyle="->", width=2, style="dashed", label="backward")
    plt.legend(); plt.title("Backpropagation Computational Graph")
    plt.axis("off")
    plt.savefig("images/ch05/NN05_backprop_graph.png", dpi=150)

draw_backprop_graph()
print("Backprop graph saved")

