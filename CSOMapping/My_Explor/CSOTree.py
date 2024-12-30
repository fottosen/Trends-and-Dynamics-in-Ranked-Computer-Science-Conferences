import rdflib
import json
from collections import defaultdict

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

# Recursive function to build the tree with depth limit
def build_tree(parent_uri, current_depth, max_depth=2):
    if current_depth >= max_depth:
        return {}
    
    children = fetch_children(parent_uri)
    tree = {}
    
    for child, label in children:
        child_uri = str(child)
        tree[child_uri] = {
            'label': str(label) if label else None,
            'children': build_tree(child_uri, current_depth + 1, max_depth)
        }
    
    return tree

# Start building the tree from the root with depth control
root_uri = "https://cso.kmi.open.ac.uk/topics/computer_science"
tree_structure = {
    root_uri: {
        'label': None,  # Root may not have a label
        'children': build_tree(root_uri, current_depth=1, max_depth=2)
    }
}


# Save the tree structure to a JSON file
def save_tree_structure(tree, filename):
    with open(filename, 'w') as f:
        json.dump(tree, f, indent=2)

# Load the tree structure from a JSON file
def load_tree_structure(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Save the tree structure to a file
save_tree_structure(tree_structure, 'tree_lvl2.json')

# To load the structure later, uncomment the line below
# tree_structure_loaded = load_tree_structure('tree_structure.json')



# Function to print the tree structure nicely
def print_tree(tree, indent=0):
    for key, value in tree.items():
        print(' ' * indent + f"{value['label'] or key}")
        print_tree(value['children'], indent + 2)

# Print the resulting tree structure
print_tree(tree_structure)


"""

------------------ COUNT NUMBER OF CHILDREN ------------------ 

"""


import json

def count_children(tree):
    # Base case: if there are no children, return 0
    if not tree['children']:
        return 0

    # Count the children in the current level
    total_children = len(tree['children'])

    # Recursively count the children in the next levels
    for child_uri, child_info in tree['children'].items():
        total_children += count_children(child_info)
    
    return total_children

# Load the tree structure from a JSON file
def load_tree_structure(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Load the tree structure
tree_structure = load_tree_structure('tree_structure_lvl3.json')

# Since the root has one main key, start from its first child
root_key = next(iter(tree_structure))  # Get the root key (the main URI)
total_children_count = count_children(tree_structure[root_key])

print(f"Total number of children: {total_children_count}")

