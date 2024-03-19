from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained CNN model for clothing classification
model = tf.keras.models.load_model('clothing_classifier_model.h5')


def detect_color(image_path, num_colors=3):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    return colors.astype(int)


def color_name(rgb):
    # Define color ranges and corresponding names
    color_names = {
        "Red": [(255, 0, 0)],
        "Green": [(0, 255, 0)],
        "Blue": [(0, 0, 255)],
        "Yellow": [(255, 255, 0)],
        "Cyan": [(0, 255, 255)],
        "Magenta": [(255, 0, 255)],
        "White": [(255, 255, 255)],
        "Black": [(0, 0, 0)],
        "Gray": [(128, 128, 128)],
        "Purple": [(128, 0, 128)],
        "Orange": [(255, 165, 0)],
        "Brown": [(165, 42, 42)],
        "Pink": [(255, 192, 203)],
        "Beige": [(245, 245, 220)],
        "Turquoise": [(64, 224, 208)],
        "LightGray": [(211, 211, 211)],
        "DarkGray": [(169, 169, 169)],
        "LightBlue": [(173, 216, 230)],
        "DarkBlue": [(0, 0, 139)],
        "LightGreen": [(144, 238, 144)],
        "DarkGreen": [(0, 100, 0)],
        "LightRed": [(255, 99, 71)],
        "DarkRed": [(139, 0, 0)],
        # Add more colors or ranges as needed
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_product', methods=['POST'])
def add_product():
    # Retrieve data from the request
    item_name = request.form['itemName']
    description = request.form['description']
    quantity = request.form['quantity']
    category = request.form['category']
    material = request.form['material']
    item_for = request.form['itemFor']
    image_file = request.files['image']

    # Save the image file temporarily
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Detect colors in the image
    colors = detect_color(image_path, num_colors=3)

    # Convert RGB colors to color names
    color_names = [color_name(color) for color in colors]

    # Load and preprocess the image for clothing classification
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the clothing type using the loaded model
    clothing_type = model.predict_classes(img_array)[0]
    clothing_types = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                      'Ankle boot']
    predicted_clothing = clothing_types[clothing_type]

    # Prepare response data
    response_data = {
        'status': 'success',
        'message': 'Product added successfully!',
        'data': {
            'itemName': item_name,
            'description': description,
            'quantity': quantity,
            'category': category,
            'material': material,
            'itemFor': item_for,
            'colors': color_names,
            'clothingType': predicted_clothing
        }
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
