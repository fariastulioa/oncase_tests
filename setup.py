from misc import FastRecipeSearch
import pandas as pd

# Load your recipes
df = pd.read_csv('recipes.csv')

# Initialize and save index
search_system = FastRecipeSearch()
search_system.save_recipes(df, 'full_recipe')