import streamlit as st
from misc import FastRecipeSearch
import pandas as pd



df = pd.read_csv("clean_recipes.csv")


# Initialize search system
@st.cache_resource
def get_search_system():
    search_system = FastRecipeSearch()
    if not search_system.load_index():
        st.error("No recipe index found. Please run the setup script first.")
    return search_system

# Format ingredients list to bullet points
def format_ingredients(ingredients_list):
    if isinstance(ingredients_list, str):
        # If the list is stored as a string, evaluate it
        ingredients_list = eval(ingredients_list)
    return "\n".join([f"• {ingredient.strip()}" for ingredient in ingredients_list])

# Format directions list to numbered steps
def format_directions(directions_list):
    if isinstance(directions_list, str):
        # If the list is stored as a string, evaluate it
        directions_list = eval(directions_list)
    return "\n".join([f"{i+1}. {direction.strip()}" for i, direction in enumerate(directions_list)])

# Set page configuration
st.set_page_config(layout="wide", page_title="Fast Recipe Search")

# Add title
st.title("Recipe Search by Tags")

# Initialize session state for search counter if it doesn't exist
if 'search_counter' not in st.session_state:
    st.session_state.search_counter = 0

# Create two columns
col1, col2 = st.columns([1, 2])

# Initialize search system
search_system = get_search_system()

# Left column - Search inputs
with col1:
    st.subheader("Enter Search Tags")
    
    # Create 5 text input boxes
    tags = []
    for i in range(5):
        tag = st.text_input(f"Tag {i+1}", key=f"tag_{i}")
        tags.append(tag)
    
    # Search button
    search_clicked = st.button("Search", type="primary")

# Right column - Results
with col2:
    st.subheader("Search Results")
    
    if search_clicked:
        # Increment search counter to ensure unique keys
        st.session_state.search_counter += 1
        
        # Filter out empty tags
        filtered_tags = [tag for tag in tags if tag.strip()]
        
        if len(filtered_tags) > 0:
            try:
                # Get search results
                results = search_system.search_by_tags(filtered_tags, df)
                
                # Display results
                for idx, result in enumerate(results, 1):
                    search_id = st.session_state.search_counter
                    with st.expander(f"Recipe: {result['title']} (Similarity: {result['similarity_score']:.2f})", expanded=True):
                        # Format ingredients and directions
                        formatted_ingredients = format_ingredients(result['ingredients'])
                        formatted_directions = format_directions(result['directions'])
                        
                        # Create unique keys for each markdown by combining search_id and recipe index
                        ingredients_key = f"search_{search_id}_recipe_{idx}_ingredients"
                        directions_key = f"search_{search_id}_recipe_{idx}_directions"
                        
                        st.markdown("### Ingredients")
                        st.markdown(formatted_ingredients)
                        
                        st.markdown("### Directions")
                        st.markdown(formatted_directions)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter at least one tag to search.")