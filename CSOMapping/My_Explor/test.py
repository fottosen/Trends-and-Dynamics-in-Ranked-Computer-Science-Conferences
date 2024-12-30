import rdflib
import json

# Load the RDF file
graph = rdflib.Graph()
graph.parse("CSO.3.3.owl", format="xml")  # Adjust format if necessary

# Query to find all classes and their labels
def fetch_children(parent_uri):
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        
        SELECT ?child ?label WHERE {{
            <{parent_uri}> <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf> ?child .
            OPTIONAL {{ ?child rdfs:label ?label }}
        }}
    """
    return graph.query(query)

# Recursive function to build the tree with depth limit and collect labels
def build_tree_and_collect_labels(parent_uri, current_depth, max_depth=3, labels=None):
    if current_depth >= max_depth:
        return {}
    
    if labels is None:
        labels = []
    
    children = fetch_children(parent_uri)
    tree = {}
    
    for child, label in children:
        child_uri = str(child)
        label_text = str(label) if label else None
        if label_text:
            labels.append(label_text)
        tree[child_uri] = {
            'label': label_text,
            'children': build_tree_and_collect_labels(child_uri, current_depth + 1, max_depth, labels)
        }
    
    return tree, labels

# Start building the tree from the root with depth control and collect labels
root_uri = "https://cso.kmi.open.ac.uk/topics/computer_science"
tree_structure, all_labels = build_tree_and_collect_labels(root_uri, current_depth=1, max_depth=3)

# Save the labels list to a JSON file
def save_labels(labels, filename):
    with open(filename, 'w') as f:
        json.dump(labels, f, indent=2)

# Save the tree structure to a JSON file
def save_tree_structure(tree, filename):
    with open(filename, 'w') as f:
        json.dump(tree, f, indent=2)

# Save the tree structure and labels to files
save_tree_structure(tree_structure, 'tree_structure.json')
save_labels(all_labels, 'labels_list.json')

# To load the structure or labels later, uncomment the lines below
# tree_structure_loaded = load_tree_structure('tree_structure.json')
# all_labels_loaded = load_labels('labels_list.json')

# Function to print the tree structure nicely
def print_tree(tree, indent=0):
    for key, value in tree.items():
        print(' ' * indent + f"{value['label'] or key}")
        print_tree(value['children'], indent + 2)

# Print the resulting tree structure
print_tree(tree_structure)
