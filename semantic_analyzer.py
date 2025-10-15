import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SemanticAnalyzer:
    """Handles semantic analysis and grouping of text units"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sentence_model = SentenceTransformer(model_name)
    
    def calculate_semantic_similarity(self, units: List[str]) -> np.ndarray:
        """Calculate semantic similarity matrix between units"""
        if len(units) < 2:
            return np.array([[1.0]])
        
        embeddings = self.sentence_model.encode(units)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def group_related_units(self, units: List[str], similarity_matrix: np.ndarray, 
                          similarity_threshold: float = 0.3) -> List[List[int]]:
        """Group semantically related units using optimized clustering"""
        if len(units) <= 1:
            return [[0]] if units else []
        
        adjacency_matrix = similarity_matrix > similarity_threshold
        np.fill_diagonal(adjacency_matrix, False)
        
        visited = set()
        groups = []
        
        def dfs(node: int, current_group: List[int]):
            if node in visited:
                return
            visited.add(node)
            current_group.append(node)
            
            for neighbor in range(len(units)):
                if neighbor not in visited and adjacency_matrix[node][neighbor]:
                    dfs(neighbor, current_group)
        
        for i in range(len(units)):
            if i not in visited:
                group = []
                dfs(i, group)
                groups.append(group)
        
        return groups