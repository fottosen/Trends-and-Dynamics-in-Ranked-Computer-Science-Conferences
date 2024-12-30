import rdflib

# Load the RDF file
graph = rdflib.Graph()
graph.parse("CSO.3.3.owl", format="xml")  # Adjust format if necessary

# Query to find all classes and their labels
query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT ?child ?label WHERE {
        <https://cso.kmi.open.ac.uk/topics/computer_science> <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf> ?child .
        OPTIONAL { ?child rdfs:label ?label }
    }
"""

classes = graph.query(query)

# Print the results
for cls in classes:
    child, label = cls
    print(label)






