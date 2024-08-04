from flask import Flask, request, jsonify
import os
from data import fetch_data, preprocess_data, train_model, find_liked_products, find_recs_for_all, obtain_preds_for_products, fetch_user_products, insert_recommendations
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)


user_ratings_df, products_df, users = fetch_data()
df, images = preprocess_data(products_df)

# Train model
model, label_map = train_model(df, images)

predictions = obtain_preds_for_products(images, model)


@app.route('/recommend', methods=['POST'])
def recommend():

    print(users)
    for user_id in users:
        print(user_id[0])
        user_ratings_df2 = fetch_user_products(user_id[0])

        # Generate recommendations
        liked_products = find_liked_products(user_id[0], user_ratings_df2)
        recommendations = find_recs_for_all(liked_products['product_id'].tolist(), 5, predictions, list(label_map.keys()), model, images)
        print(liked_products)
        # Convert recommendations to native Python int
        print(recommendations)
        recommendations = [int(rec) for rec in recommendations]
        insert_recommendations(user_id, recommendations)

    return jsonify(message="Recommendations generated and inserted successfully")

if __name__ == '__main__':
    app.run(debug=True)