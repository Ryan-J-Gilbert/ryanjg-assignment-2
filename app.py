from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans import KMeans  # Your custom KMeans implementation
import plotly.graph_objects as go
import json
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    '''Generate a new random dataset and return it as JSON'''

    # Generate a new random dataset (e.g., 2D points)
    points = np.random.uniform(-10, 10, (500, 2))  # 500 random points
    np.save('dataset.npy', points)  # Save the dataset so it persists across steps
    
    # Send the points as JSON
    return jsonify({'points': points.tolist()})

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    '''Run the KMeans algorithm and return the final centroids and point labels'''

    # Get the number of clusters and initialization method
    k = int(request.json['k'])
    method = request.json['method']  # Random, Farthest, KMeans++ or Manual

    if method == 'manual':
        manual_centers = request.json['manual_centroids']
        k = len(manual_centers)
    else:
        manual_centers = []

    # Load the dataset
    points = np.load('dataset.npy')

    # Initialize and run the KMeans algorithm
    kmeans = KMeans(data=points, n_clusters=k, init_method=method)
    kmeans.lloyds(method, manual_centers=manual_centers)

    # Get the final centroids and point labels
    centroids = kmeans.centers
    labels = kmeans.assignment

    # return jsonify({'centroids': centroids.tolist(), 'points': points.tolist(), 'labels': labels})
    return jsonify({'centroids_list': [centroids.tolist()], 'points': points.tolist(), 'labels_list': [labels]})



@app.route('/step', methods=['POST'])
def step():
    '''Run one step of the KMeans algorithm and return the updated centroids and point labels'''

    # Get the number of clusters and initialization method
    k = int(request.json['k'])
    method = request.json['method']

    if method == 'manual':
        manual_centers = request.json['manual_centroids']
        k = len(manual_centers)
    else:
        manual_centers = []

    points = np.load('dataset.npy')

    kmeans = KMeans(data=points, n_clusters=k, init_method=method)
    kmeans.lloyds(method, manual_centers=manual_centers)

    centroids_list = []
    labels_list = []

    while kmeans.step():
        centroids_list.append(kmeans.centers.tolist())
        labels_list.append(kmeans.assignment)

    return jsonify({'centroids_list': centroids_list, 'points': points.tolist(), 'labels_list': labels_list})



@app.route('/reset_kmeans', methods=['POST'])
def reset_kmeans():
    '''Reset the KMeans algorithm to its initial state by reloading the dataset'''

    points = np.load('dataset.npy')
    return jsonify({'points': points.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
