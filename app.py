from flask import Flask, request, jsonify
import os
from data import fetch_data, preprocess_data, train_model, find_liked_products, find_recs_for_all, obtain_preds_for_products, fetch_user_products
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Fetch and preprocess data
user_ratings_df, products_df = fetch_data()
df, images = preprocess_data(products_df)

# Train model
model, label_map = train_model(df, images)

predictions = obtain_preds_for_products(images, model)

@app.route('/recommend', methods=['POST'])
def recommend():


    user_id = request.json.get('user_id')
    user_ratings_df2 = fetch_user_products(user_id)

    # Generate recommendations
    liked_products = find_liked_products(user_id, user_ratings_df2)
    recommendations = find_recs_for_all(liked_products['product_id'].tolist(), 5, predictions, list(label_map.keys()), model, images)

    # Convert recommendations to native Python int
    recommendations = [int(rec) for rec in recommendations]

    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)