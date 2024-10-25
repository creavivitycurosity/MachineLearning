# from flask import Flask, request, jsonify
# import mysql.connector
# from fuzzywuzzy import fuzz
# from flask_cors import CORS  # Import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Connect to MySQL database
# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",  # Replace with your MySQL username
#         password="root",  # Replace with your MySQL password
#         database="food_app16"  # Replace with your database name
#     )

# # Function to fetch items from MySQL
# def fetch_items_from_db():
#     conn = get_db_connection()
#     cursor = conn.cursor(dictionary=True)
#     cursor.execute("SELECT * FROM items")
#     items = cursor.fetchall()
#     conn.close()
#     return items

# # Rank items by user history and rating
# def rank_items(items):
#     purchased = [item for item in items if 'purchased' in item['user_history']]
#     searched = [item for item in items if 'searched' in item['user_history']]
#     not_interacted = [item for item in items if not item['user_history']]

#     purchased = sorted(purchased, key=lambda x: x['rating'], reverse=True)
#     searched = sorted(searched, key=lambda x: x['rating'], reverse=True)
#     not_interacted = sorted(not_interacted, key=lambda x: x['rating'], reverse=True)

#     return purchased + searched + not_interacted

# # Traditional and ML auto-suggestion logic
# def auto_suggestion(user_input, items):
#     # Traditional: Exact match
#     name_matches = [item for item in items if user_input.lower() in item['name'].lower()]
#     tag_matches = [item for item in items if 'tags' in item and user_input.lower() in item['tags']]  # Check if 'tags' exists

#     # ML-based: Fuzzy matching
#     ml_name_matches = [item for item in items if fuzz.partial_ratio(user_input.lower(), item['name'].lower()) > 70]
#     ml_tag_matches = [item for item in items if 'tags' in item and any(fuzz.partial_ratio(user_input.lower(), tag.lower()) > 70 for tag in item['tags'])]  # Check if 'tags' exists

#     # Rank matches
#     ranked_name_matches = rank_items(name_matches)
#     ranked_tag_matches = rank_items(tag_matches)

#     ml_ranked_name_matches = rank_items(ml_name_matches)
#     ml_ranked_tag_matches = rank_items(ml_tag_matches)

#     return {
#         "traditional": {
#             "name_matches": ranked_name_matches,
#             "tag_matches": ranked_tag_matches
#         },
#         "ml_based": {
#             "name_matches": ml_ranked_name_matches,
#             "tag_matches": ml_ranked_tag_matches
#         }
#     }

# # API endpoint for auto-suggestions
# @app.route('/suggest', methods=['GET'])
# def suggest():
#     user_input = request.args.get('query')  # Get user input from query parameter
#     items = fetch_items_from_db()  # Fetch items from the database

#     suggestions = auto_suggestion(user_input, items)

#     return jsonify(suggestions)

# @app.route('/', methods=['GET'])
# def hello():
#     return jsonify(message="Hello, World!")

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import mysql.connector
from fuzzywuzzy import fuzz
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Connect to MySQL database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="root",  # Replace with your MySQL password
        database="food_app16"  # Replace with your database name
    )

# Fetch user order history based on email (assumed logged-in user)
def fetch_user_history(email):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT oi.item_id
    FROM orders o
    JOIN order_item oi ON o.id = oi.orders_id
    JOIN ourusers u ON o.user_id = u.id
    WHERE u.email = %s
    """
    cursor.execute(query, (email,))
    user_items = cursor.fetchall()
    
    conn.close()
    return [item['item_id'] for item in user_items]

# Fetch all items with their average rating and tags
def fetch_items_with_ratings():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Fetch all items with their average rating from ItemReview and group tags
    query = """
    SELECT 
        i.id, 
        i.name, 
        i.price, 
        GROUP_CONCAT(t.tags) AS tags,  -- Grouping tags together as a comma-separated string
        COALESCE(AVG(r.rating), 0) AS average_rating
    FROM item i
    LEFT JOIN item_review r ON i.id = r.item_id
    LEFT JOIN item_tags t ON i.id = t.item_id  -- Joining the tags table
    GROUP BY i.id
    """
    cursor.execute(query)
    items = cursor.fetchall()
    
    # Post-process to split tags back into a list
    for item in items:
        if item['tags']:
            item['tags'] = item['tags'].split(',')
        else:
            item['tags'] = []  # No tags, return an empty list
    
    conn.close()
    return items

# Rank items based on user history and average rating
def rank_items(items, user_history):
    # First rank based on user history (purchased items), then by rating
    purchased = [item for item in items if item['id'] in user_history]
    not_purchased = [item for item in items if item['id'] not in user_history]

    # Sort items by 'average_rating' (handle missing ratings with default 0)
    purchased = sorted(purchased, key=lambda x: x.get('average_rating', 0), reverse=True)
    not_purchased = sorted(not_purchased, key=lambda x: x.get('average_rating', 0), reverse=True)

    return purchased + not_purchased
# Traditional and ML auto-suggestion logic
def auto_suggestion(user_input, items, user_history):
    # Traditional: Exact match on name, include all items regardless of tags
    name_matches = [item for item in items if user_input.lower() in item['name'].lower()]
    
    # For items that have tags, do exact matching on tags
    tag_matches = [item for item in items if item['tags'] and user_input.lower() in item['tags']]

    # ML-based: Fuzzy matching on name (include all items, regardless of tags)
    ml_name_matches = [item for item in items if fuzz.partial_ratio(user_input.lower(), item['name'].lower()) > 70]
    
    # For items that have tags, perform fuzzy matching on tags
    ml_tag_matches = [item for item in items if item['tags'] and any(fuzz.partial_ratio(user_input.lower(), tag.lower()) > 70 for tag in item['tags'])]

    # Rank matches based on user history and rating
    ranked_name_matches = rank_items(name_matches, user_history)
    ranked_tag_matches = rank_items(tag_matches, user_history)

    ml_ranked_name_matches = rank_items(ml_name_matches, user_history)
    ml_ranked_tag_matches = rank_items(ml_tag_matches, user_history)

    return {
        "traditional": {
            "name_matches": ranked_name_matches,
            "tag_matches": ranked_tag_matches
        },
        "ml_based": {
            "name_matches": ml_ranked_name_matches,
            "tag_matches": ml_ranked_tag_matches
        }
    }


# API endpoint for auto-suggestions
@app.route('/suggest', methods=['GET'])
def suggest():
    user_input = request.args.get('query')  # Get user input from query parameter
    user_email = "test12@gmail.com"  # For testing, use a hardcoded email (replace with real session/email)
    
    # Fetch items and user history
    items = fetch_items_with_ratings()  # Now fetch items with their average rating and tags
    user_history = fetch_user_history(user_email)

    suggestions = auto_suggestion(user_input, items, user_history)

    return jsonify(suggestions)

# Test route to ensure API is working
@app.route('/', methods=['GET'])
def hello():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    app.run(debug=True)
