import networkx as nx
from bs4 import BeautifulSoup
import requests
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class GraphRAG:
    def __init__(self):
        self.graph = nx.Graph()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.content_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
    def scrape_website(self, url: str) -> str:
        """Scrape content from a website."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            # Basic text cleaning
            text = ' '.join(text.split())
            return text
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""
        
    def add_node(self, url: str):
        """Add a node to the graph and scrape its content."""
        if url not in self.graph:
            content = self.scrape_website(url)
            self.content_cache[url] = content
            embedding = self.embedder.encode([content])[0]
            self.embedding_cache[url] = embedding
            self.graph.add_node(url, content=content, embedding=embedding)
            
    def add_edge(self, url1: str, url2: str, weight: float = None):
        """Add an edge between nodes with optional weight."""
        if weight is None and url1 in self.embedding_cache and url2 in self.embedding_cache:
            # Calculate similarity between nodes if weight not provided
            sim = cosine_similarity(
                [self.embedding_cache[url1]], 
                [self.embedding_cache[url2]]
            )[0][0]
            weight = sim
            
        self.graph.add_edge(url1, url2, weight=weight)
        
    def query(self, query: str, top_k: int = 3) -> List[Dict]:
        """Query the graph and return most relevant nodes."""
        query_embedding = self.embedder.encode([query])[0]
        
        similarities = []
        for url in self.graph.nodes():
            node_embedding = self.embedding_cache[url]
            sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
            similarities.append((url, sim))
            
        # Sort by similarity and get top k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for url, sim in similarities[:top_k]:
            # Get neighboring nodes
            neighbors = list(self.graph.neighbors(url))
            
            results.append({
                'url': url,
                'content': self.content_cache[url][:500] + "...",  # Preview only
                'similarity': sim,
                'neighbors': neighbors
            })
            
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the GraphRAG
    rag = GraphRAG()
    
    # Add some example URLs
    urls = [
        'https://www.example.com',
        'https://www.python.org',
        'https://www.github.com'
    ]
    
    # Build the graph
    for url in urls:
        rag.add_node(url)
    
    # Add edges between all nodes
    for i in range(len(urls)):
        for j in range(i+1, len(urls)):
            rag.add_edge(urls[i], urls[j])
    
    # Example query
    query = "What is Python programming?"
    results = rag.query(query)
    
    # Print results
    for result in results:
        print(f"\nURL: {result['url']}")
        print(f"Content Preview: {result['content']}")
        print(f"Similarity Score: {result['similarity']:.3f}")
        print(f"Connected to: {', '.join(result['neighbors'])}")
