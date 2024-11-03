from dotenv import load_dotenv
import os
from typing import List, Dict, Optional
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from groq import Groq
import networkx as nx
from dataclasses import dataclass

@dataclass
class MedicalEntity:
    uri: str
    label: str
    description: Optional[str]
    entity_type: str
    related_entities: List[Dict]
    source_links: List[str]

class DBpediaMedicalGraphRAG:
    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat(JSON)
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.knowledge_graph = nx.DiGraph()

    def query_medical_entity(self, search_term: str) -> MedicalEntity:
        """
        Query DBpedia for medical entities and their relationships
        """
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        
        SELECT DISTINCT ?entity ?label ?description ?type ?related ?relatedLabel WHERE {
            ?entity rdfs:label ?label ;
                    a ?type .
            OPTIONAL { 
                ?entity dbo:abstract ?description .
                FILTER(LANG(?description) = "en")
            }
            OPTIONAL {
                ?entity ?relation ?related .
                ?related rdfs:label ?relatedLabel .
                FILTER(LANG(?relatedLabel) = "en")
                FILTER(?relation IN (dbo:drug, dbo:disease, dbo:protein, dbo:anatomicalStructure))
            }
            FILTER(CONTAINS(LCASE(STR(?label)), LCASE(%s)))
            FILTER(LANG(?label) = "en")
            FILTER(?type IN (dbo:Disease, dbo:Drug, dbo:Protein, dbo:AnatomicalStructure))
        }
        LIMIT 10
        """ % f'"{search_term}"'

        self.sparql.setQuery(query)
        results = self.sparql.query().convert()

        # Process results into MedicalEntity
        entity_data = {}
        for result in results["results"]["bindings"]:
            uri = result["entity"]["value"]
            if uri not in entity_data:
                entity_data[uri] = {
                    "uri": uri,
                    "label": result["label"]["value"],
                    "description": result.get("description", {}).get("value"),
                    "entity_type": result["type"]["value"].split("/")[-1],
                    "related_entities": [],
                    "source_links": [uri]  # DBpedia URI as source
                }
            
            if "related" in result and "relatedLabel" in result:
                entity_data[uri]["related_entities"].append({
                    "uri": result["related"]["value"],
                    "label": result["relatedLabel"]["value"]
                })

        # Convert to MedicalEntity object
        if entity_data:
            first_uri = next(iter(entity_data))
            return MedicalEntity(**entity_data[first_uri])
        return None

    def update_knowledge_graph(self, entity: MedicalEntity):
        """
        Update the knowledge graph with new entity information
        """
        # Add main entity node
        self.knowledge_graph.add_node(
            entity.uri,
            label=entity.label,
            entity_type=entity.entity_type,
            description=entity.description
        )

        # Add related entity nodes and edges
        for related in entity.related_entities:
            self.knowledge_graph.add_node(
                related["uri"],
                label=related["label"]
            )
            self.knowledge_graph.add_edge(entity.uri, related["uri"])

    def generate_rag_context(self, query: str) -> str:
        """
        Generate RAG context from knowledge graph for Groq
        """
        relevant_nodes = []
        for node, data in self.knowledge_graph.nodes(data=True):
            if any(term.lower() in data.get('label', '').lower() for term in query.lower().split()):
                data['uri'] = node  # Add uri to node data
                relevant_nodes.append(data)

        context = "Medical Knowledge Context:\n\n"
        for node_data in relevant_nodes:
            context += f"- {node_data.get('label')}: {node_data.get('description', 'No description available')}\n"
            # Add connected entities
            connected = [self.knowledge_graph.nodes[n]['label'] 
                       for n in self.knowledge_graph.neighbors(node_data['uri'])]
            if connected:
                context += f"  Related: {', '.join(connected)}\n"

        return context

    def query_with_rag(self, user_query: str) -> str:
        """
        Query the medical knowledge using RAG through Groq
        """
        context = self.generate_rag_context(user_query)
        
        prompt = f"""Context: {context}

Query: {user_query}

Based on the medical knowledge provided in the context, please provide a detailed response. 
Include relevant relationships between medical entities and cite sources when possible.

Response:"""

        response = self.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content


# Load environment variables at the start (incl API key)
load_dotenv()


# Example usage
def main():
    rag_system = DBpediaMedicalGraphRAG()
    
    # Example usage
    search_terms = [
        "type 2 diabetes",
        "insulin",
        "pancreas"
    ]
    
    # Build knowledge graph
    for term in search_terms:
        entity = rag_system.query_medical_entity(term)
        if entity:
            rag_system.update_knowledge_graph(entity)
    
    # Example medical query
    query = "What is the relationship between insulin and diabetes, and how does it affect the pancreas?"
    print(query)
    print(40 * "-")
    response = rag_system.query_with_rag(query)
    print(response)

if __name__ == "__main__":
    main()
