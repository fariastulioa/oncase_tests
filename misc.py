from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle
import os

class FastRecipeSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the search system with a small but effective model.
        all-MiniLM-L6-v2 is fast and lightweight while maintaining good performance.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.recipes = None
        self.index_path = 'recipe_index.faiss'
        self.recipes_path = 'recipes.pkl'
        
    def save_recipes(self, df, text_column):
        """
        Save recipes to FAISS index for fast similarity search
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing recipes
        text_column (str): Name of the column containing recipe text
        """
        print("Converting recipes to embeddings...")
        # Get recipes as list
        self.recipes = df[text_column].tolist()
        
        # Create embeddings in batches to manage memory
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(self.recipes), batch_size):
            batch = self.recipes[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, 
                                               show_progress_bar=True,
                                               convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            
        # Combine all embeddings
        embeddings = np.vstack(embeddings)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Save index and recipes
        print("Saving index and recipes...")
        faiss.write_index(self.index, self.index_path)
        with open(self.recipes_path, 'wb') as f:
            pickle.dump(self.recipes, f)
            
        print(f"Saved {len(self.recipes)} recipes to index")
        
    def load_index(self):
        """Load saved index and recipes if they exist"""
        if os.path.exists(self.index_path) and os.path.exists(self.recipes_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.recipes_path, 'rb') as f:
                self.recipes = pickle.load(f)
            return True
        return False
    
    def search_by_tags(self, tags, df, n_results=5):
        """
        Search recipes using tags
        
        Parameters:
        tags (list): List of tags to search for
        n_results (int): Number of results to return
        
        Returns:
        list: List of dictionaries containing recipe text and similarity scores
        """
        if not self.index or not self.recipes:
            if not self.load_index():
                raise ValueError("No index found. Please run save_recipes first.")
        
        # Combine tags into search query
        search_query = " ".join(tags)
        
        # Get query embedding
        query_vector = self.model.encode([search_query], 
                                       convert_to_numpy=True, 
                                       show_progress_bar=False)
        
        # Search index
        distances, indices = self.index.search(query_vector.astype('float32'), k=n_results)
        
        # Format results
        results = []
        for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
            # Convert distance to similarity score (closer to 1 is better)
            similarity = 1 / (1 + distance)
            results.append({
                'title': df.loc[index, "title"],
                'ingredients': df.loc[index, "ingredients"],
                'directions': df.loc[index, "directions"],
                'original_index': index,
                'similarity_score': similarity
            })
            
        return results

# Example usage:
"""
# Initialize search system
search_system = FastRecipeSearch()

# First time setup with your DataFrame
df = pd.read_csv('recipes.csv')
search_system.save_recipes(df, 'full_recipe')

# Later searches (fast!)
tags = ['vegetarian', 'quick', 'pasta', 'italian', 'simple']
results = search_system.search_by_tags(tags)

# Print results
for result in results:
    print(f"\nRecipe from index {result['original_index']}")
    print(f"Similarity score: {result['similarity_score']:.2f}")
    print(f"Recipe text: {result['recipe'][:200]}...")
"""