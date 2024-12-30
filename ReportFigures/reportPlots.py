import pandas as pd
import plotly.graph_objects as go

# Load the CSV data
"""type = ["Book and Theses","Conference and Workshop Papers","Editorship","Informal Publications","Journal Articles","Parts in Books or Collections","Reference Works"]

Publications = [157181,3586154,62676,726969,2981498,43340,27365]"""
type = ["Book and Theses", "Conference and Workshop Papers", "Other", "Informal Publications", "Journal Articles"]

Publications = [157181, 3586154, 133381, 726969, 2981498]

# Create the pie chart with labels and percentages
fig = go.Figure(data=[go.Pie(
    labels=type,
    values=Publications,
    pull=[0, 0.08, 0, 0, 0],
    textinfo='label+percent',  # Display both label and percentage
    textfont=dict(size=14),     # Increase the text size
)])

fig.update_layout(
    title_text="Distribution of Publications by Type in DBLP",
)

# Show the plot
fig.show()





"""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = ""  # Replace with the path to your CSV file
data = pd.read_csv(file_path, delimiter=';')

# Extract the data
types = data["type"]
publications = data["#Publications"]

# Calculate percentages
total_publications = publications.sum()
percentages = (publications / total_publications) * 100

# Function to fix label overlap
def fix_labels(mylabels, tooclose=0.1, sepfactor=2):
    vecs = np.zeros((len(mylabels), len(mylabels), 2))
    dists = np.zeros((len(mylabels), len(mylabels)))
    for i in range(0, len(mylabels)-1):
        for j in range(i+1, len(mylabels)):
            a = np.array(mylabels[i].get_position())
            b = np.array(mylabels[j].get_position())
            dists[i, j] = np.linalg.norm(a - b)
            vecs[i, j, :] = a - b
            if dists[i, j] < tooclose:
                mylabels[i].set_x(a[0] + sepfactor * vecs[i, j, 0])
                mylabels[i].set_y(a[1] + sepfactor * vecs[i, j, 1])
                mylabels[j].set_x(b[0] - sepfactor * vecs[i, j, 0])
                mylabels[j].set_y(b[1] - sepfactor * vecs[i, j, 1])

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))

# Create the pie chart with a hole (donut chart)
wedges, texts = ax.pie(
    publications,
    wedgeprops=dict(width=0.6),
    startangle=-40,
    labels=None  # Remove default labels for manual annotations
)

# Annotation styling
kw = dict(arrowprops=dict(arrowstyle="-"), va="center", fontsize=12)
annotations = []
for wedge, label, percent in zip(wedges, types, percentages):
    # Calculate angle for annotation
    ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
    y = np.sin(ang)
    x = np.cos(ang)
    horizontalalignment = (
        "center" if abs(x) < abs(y) else "right" if x < 0 else "left"
    )
    # Combine label and percentage
    label_with_percent = f"{label} ({percent:.1f}%)"
    ann = ax.annotate(
        label_with_percent,
        xy=(0.70 * x, 0.70 * y),  # Arrow tip position
        xytext=(1.3 * x, 1.3 * y),  # Text position
        horizontalalignment=horizontalalignment,
        **kw
    )
    annotations.append(ann)

# Fix overlapping labels
fix_labels(annotations, tooclose=0.1, sepfactor=2)

# Finalize layout and show plot
plt.tight_layout()
plt.show()"""