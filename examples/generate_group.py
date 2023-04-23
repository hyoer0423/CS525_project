

def try_insert(node):
        
    for key in union:
        is_all_connected = 1
        for group_node in union[key]:
            if (node,group_node) in graph and graph[(node,group_node)] == 0 :
                is_all_connected = 0
                break
        if is_all_connected:
            union[key].add(node)
            return 

    union[node] = set()
    union[node].add(node)
    
        
    
                
def get_graph_snd_nodes():
    graph = {}
    nodes = {}
    for line in open('/root/hloenv/examples/conflict_passes_result.text'):
        node1, node2, total_conflict = line.strip().split()
        graph[(node1,node2)] = int(total_conflict)
        graph[(node2,node1)] = int(total_conflict)
        nodes[node1] = 1
        nodes[node2] = 1
    return graph, nodes
        
    

union = {}
graph,nodes = get_graph_snd_nodes()
print(len(nodes))
for node in nodes:
    try_insert(node) 
    
for key in union:
    print("group:",union[key],"\n\n")

# python3 /root/hloenv/examples/generate_group.py