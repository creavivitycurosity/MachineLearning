# ML-based approach

from fuzzywuzzy import fuzz, process

# Sample menu items dataset
menu_items = [
    {
        'name': 'Chicken Fried Rice',
        'tags': ['rice', 'chicken', 'fried', 'Asian'],
        'rating': 4.7,
        'user_history': []
    },
    {
        'name': 'Vegan Rice Bowl',
        'tags': ['rice', 'vegan', 'bowl', 'healthy'],
        'rating': 4.3,
        'user_history': ['purchased']
    },
    {
        'name': 'Spicy Tofu Rice',
        'tags': ['rice', 'tofu', 'spicy', 'vegan'],
        'rating': 4.5,
        'user_history': ['searched']
    },
    {
        'name': 'Margherita Pizza',
        'tags': ['pizza', 'Italian', 'cheese', 'vegetarian'],
        'rating': 4.8,
        'user_history': ['purchased']
    }
    # Add more items as needed
]

# Function to rank items by user history first, then by ratings
def rank_items(items):
    # First rank based on user history, prioritize 'purchased', then 'searched'
    purchased = [item for item in items if 'purchased' in item['user_history']]
    searched = [item for item in items if 'searched' in item['user_history']]
    not_interacted = [item for item in items if not item['user_history']]
    
    # Sort based on rating within each history category
    purchased = sorted(purchased, key=lambda x: x['rating'], reverse=True)
    searched = sorted(searched, key=lambda x: x['rating'], reverse=True)
    not_interacted = sorted(not_interacted, key=lambda x: x['rating'], reverse=True)
    
    return purchased + searched + not_interacted

# Traditional Auto-Suggestion (Exact match)
def traditional_auto_suggestion(user_input):
    name_matches = [item for item in menu_items if user_input.lower() in item['name'].lower()]
    tag_matches = [item for item in menu_items if user_input.lower() in item['tags']]
    
    # Rank the matched items
    ranked_name_matches = rank_items(name_matches)
    ranked_tag_matches = rank_items(tag_matches)
    
    return ranked_name_matches, ranked_tag_matches

# ML-Based Auto-Suggestion (Fuzzy match and ranking by history & rating)
def ml_auto_suggestion(user_input):
    # Fuzzy match names
    name_matches = [item for item in menu_items if fuzz.partial_ratio(user_input.lower(), item['name'].lower()) > 70]
    # Fuzzy match tags
    tag_matches = [item for item in menu_items if any(fuzz.partial_ratio(user_input.lower(), tag.lower()) > 70 for tag in item['tags'])]
    
    # Rank the matched items
    ranked_name_matches = rank_items(name_matches)
    ranked_tag_matches = rank_items(tag_matches)
    
    return ranked_name_matches, ranked_tag_matches

# Function to display suggestions
def display_suggestions(user_input):
    # Get traditional suggestions
    traditional_names, traditional_tags = traditional_auto_suggestion(user_input)
    
    # Get ML-based suggestions
    ml_names, ml_tags = ml_auto_suggestion(user_input)
    
    # Display Traditional Suggestions
    print("\n--- Traditional Auto-Suggestion ---")
    print(f"Matches by Name for '{user_input}':")
    for item in traditional_names:
        print(f"{item['name']} (Rating: {item['rating']}) - Tags: {', '.join(item['tags'])}")
    
    print(f"\nMatches by Tag for '{user_input}':")
    for item in traditional_tags:
        print(f"{item['name']} (Rating: {item['rating']}) - Tags: {', '.join(item['tags'])}")
    
    # Display ML-Based Suggestions
    print("\n--- ML-Based Auto-Suggestion ---")
    print(f"Recommended by Name for '{user_input}':")
    for item in ml_names:
        print(f"{item['name']} (Rating: {item['rating']}) - Tags: {', '.join(item['tags'])}")
    
    print(f"\nRecommended by Tag for '{user_input}':")
    for item in ml_tags:
        print(f"{item['name']} (Rating: {item['rating']}) - Tags: {', '.join(item['tags'])}")

# Testing the system with sample user input
user_input = "chid"
display_suggestions(user_input)
