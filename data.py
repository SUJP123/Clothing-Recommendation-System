import psycopg2
import pandas as pd
import os
import requests
import numpy as np
import cv2
from PIL import Image
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import io

def fetch_data():
    conn = psycopg2.connect(
        host='localhost',
        database='deal',
        user='sujaypatel',
        password='password',
        port='5432'
    )
    cursor = conn.cursor()
    print("Database connected")

    user_query = """
    SELECT id FROM customer
    """

    cursor.execute(user_query)
    users = cursor.fetchall()

    query = """
    SELECT 
        userid, 
        productid, 
        user_rating
    FROM 
        bought
    WHERE status = 'PURCHASED'
    """
    cursor.execute(query)
    user_ratings = cursor.fetchall()

    product_query = """
    SELECT 
        id, 
        name, 
        retail, 
        deal, 
        saved, 
        image, 
        description,
        clothing_type
    FROM 
        products
    """
    cursor.execute(product_query)
    products = cursor.fetchall()

    user_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'product_id', 'rating'])
    products_df = pd.DataFrame(products, columns=['product_id', 'name', 'retail_price', 'deal_price', 'saved', 'image', 'description', 'clothing_type'])
    user_list = list(users)

    cursor.close()
    conn.close()

    return user_ratings_df, products_df, user_list

def insert_recommendations(user_id, product_ids):
    conn = psycopg2.connect(
        host='localhost',
        database='deal',
        user='sujaypatel',
        password='password',
        port='5432'
    )
    cursor = conn.cursor()
    print("Database connected")

    for product_id in product_ids:
        query = """
        INSERT INTO recommended (userid, productid) 
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """
        cursor.execute(query, (user_id, product_id))

    conn.commit()
    cursor.close()
    conn.close()

def fetch_user_products(userId: str):
    conn = psycopg2.connect(
        host='localhost',
        database='deal',
        user='sujaypatel',
        password='password',
        port='5432'
    )
    cursor = conn.cursor()
    print("Database connected")

    query = """
    SELECT 
        userid, 
        productid, 
        user_rating
    FROM 
        bought
    WHERE status = 'PURCHASED' AND userid = %s
    """
    cursor.execute(query, (userId,))
    user_ratings = cursor.fetchall()

    user_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'product_id', 'rating'])

    cursor.close()
    conn.close()

    return user_ratings_df

def convert_url_to_image(url):
    print("converting url")
    response = requests.get(url)
    bytes_im = io.BytesIO(response.content)
    cv_im = np.array(Image.open(bytes_im).convert('RGB'))
    resized_im = cv2.resize(cv_im, (64, 64))
    resized_im = resized_im / 255.0
    return resized_im

def map_categories(description, categories):
    print("mapping categories")
    desc_lower = description.lower()
    matched_categories = []
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', desc_lower):
                matched_categories.append(category)
                break
    
    return ' '.join(sorted(matched_categories))

def preprocess_data(products):
    print("running preprocess")
    product_ids = []
    images = []
    categories_mapped = []

    categories = {
        'Women': ['Women', "Women's"],
        'Men': ['Men', "Men's"],
        'Kids': ['Kids', 'Girls', 'Boys', 'Toddler', 'Baby'],
        'Skirt': ['Skirt'],
        'Shoe': ['Shoe', 'Shoes'],
        'Short': ['Short', 'Shorts'],
        'Dress': ['Dress'],
        'Shirt': ['Shirt', 'T-Shirt'],
        'Pants': ['Pants', 'Jeans'],
        'Joggers': ['Joggers'],
        'Hoodies': ['Hoodie', 'Hoodies', 'Jacket'],
        'SweatShirt': ['Long Sleeve', 'SweatShirt', 'Sweatshirt'],
        'Gym': ['Gym', 'Athletic', 'Basketball', 'Sport', 'Sports']
    }

    for index in range(len(products)):
        try:
            product_id = products.iloc[index]['product_id']
            description = products.iloc[index]['clothing_type']
            image = products.iloc[index]['image']
        
            preprocessed_image = convert_url_to_image(image)
            category = map_categories(description, categories)
        
            product_ids.append(product_id)
            images.append(preprocessed_image)
            categories_mapped.append(category)
        except:
            print(f"Failed to load image for index {index}")

    df = pd.DataFrame({
        'productId': product_ids,
        'category': categories_mapped,
        'image': images 
    })

    return df, np.array(images)

def train_model(df, images):
    print("training model")
    labels = np.array(list(df['category']))

    label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
    labels_encoded = np.array([label_map[label] for label in labels])
    labels_categorical = to_categorical(labels_encoded)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

    return model, label_map

def obtain_preds_for_products(images, model):
    percents = [ ]
    for i, x in enumerate(images):
        preds = model.predict(np.expand_dims(x, axis=0))
        percents.append(preds[0])
    return percents

def find_recommendations(productId, vals, features, model, images):
    temps = vals.copy()
    predictions = model.predict(np.expand_dims(images[productId-1], axis=0))
    label_index = np.argmax(predictions)
    label = features[label_index]
    feats = [ ] 
    for i in range(len(vals)):
        if i == productId-1:
            feats.append(0)
            continue
        feats.append(temps[i][label_index])
    return feats

def find_recs_for_all(productIds, num, vals, features, model, images):
    ids = [ ]
    for prod in productIds:
        count = 0
        temp = find_recommendations(prod, vals, features, model, images)
        while count < num:
            prod_number = np.argmax(temp) + 1
            if prod_number not in productIds and prod_number not in ids:
                ids.append(prod_number)
                count += 1
            temp[prod_number-1] = 0.0
    return ids

def find_liked_products(user_id, user_ratings_df):
    userMask = (user_ratings_df['user_id'] == user_id)
    user_data = user_ratings_df[userMask]
    average = np.average(user_data[user_data['rating'] != -1.0]['rating'])
    liked_products = user_data[user_data['rating'] >= average]
    return liked_products
