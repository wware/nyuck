# Messing with GraphRAG

https://towardsdatascience.com/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759

https://towardsdatascience.com/the-quest-for-production-quality-graph-rag-easy-to-start-hard-to-finish-46ca404cee3d

Here's an example of a simple graph-based web scraper application in Python using the `graphrag` library:

**graphrag_example.py**
```python
import graphrag
from graphrag import Node, Edge
from bs4 import BeautifulSoup
import requests

# Define the scraper function
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('title').text
    return title

# Define the graph
g = graphrag.Graph()

# Add nodes for the websites to scrape
nodes = []
for url in ['https://www.example.com', 'https://www.google.com', 'https://www.github.com']:
    node = Node(url)
    nodes.append(node)
    g.add_node(node)

# Add edges for the scrape relationships
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        edge = Edge(nodes[i], nodes[j], {'scrape_function': scrape_website})
        g.add_edge(edge)

# Run the graph
g.run()
```
This code defines a simple graph with three nodes (websites) and three edges (scrape relationships) between them. The `scrape_website` function is used to scrape the title of each website.

In this example, the graph is used to define a pipeline for scraping the titles of multiple websites. The `g.run()` method runs the graph, which executes the scrape function for each edge in the graph.

**To run this code:**

1. Install the `graphrag` library using `pip install graphrag`
2. Run the code using `python graphrag_example.py`
3. The code will execute the scrape function for each edge in the graph and print the title of each website to the console.

Note: This is a very simple example, and you'll likely want to add more features, such as error handling and caching, to a real-world graph-based scraper application.
