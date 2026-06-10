import matplotlib.pyplot as plt
import networkx as nx

def draw_network(layers=[2,4,1]):
    G = nx.DiGraph()
    pos = {}
    for li, sz in enumerate(layers):
        for ni in range(sz):
            n = f"L{li}N{ni}"
            G.add_node(n)
            pos[n] = (li, -ni)
    for i in range(len(layers)-1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                G.add_edge(f"L{i}N{j}", f"L{i+1}N{k}")
    nx.draw(G, pos, with_labels=False, node_color="lightblue",
            node_size=600, edge_color="gray", arrows=True)
    for i, sz in enumerate(layers):
        plt.text(i, -sz/2+.5, f"Layer{i+1}({sz})", ha="center",
                 bbox=dict(boxstyle="round", facecolor="wheat"))
    plt.title("2-Layer Neural Network")
    plt.savefig("images/ch01/NN01_network_structure.png", dpi=150)

if __name__ == "__main__":
    draw_network()

