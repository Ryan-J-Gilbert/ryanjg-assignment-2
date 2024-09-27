from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # To run Matplotlib without a display server
import matplotlib.pyplot as plt
from kmeans import KMeans  # Your custom KMeans implementation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    # Generate a new random dataset (e.g., 2D points)
    points = np.random.uniform(-10, 10, (500, 2))  # 500 random points
    np.save('dataset.npy', points)  # Save the dataset so it persists across steps
    return jsonify(status="Dataset generated")
    

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    k = int(request.form['k'])
    method = request.form['init_method']  # Random, Farthest, KMeans++ or Manual

    # Load the dataset
    points = np.load('dataset.npy')

    # Initialize and run the KMeans algorithm
    kmeans = KMeans(data=points, n_clusters=k, init_method=method)
    kmeans.fit(points)

    # Generate the final plot
    plot_html = kmeans.plot()

    return jsonify(status="KMeans run", plot_html=plot_html)

@app.route('/step', methods=['POST'])
def step():
    k = int(request.form['k'])
    method = request.form['init_method']  # Random, Farthest, KMeans++ or Manual

    # Load the dataset
    points = np.load('dataset.npy')

    # Initialize and run the KMeans algorithm
    kmeans = KMeans(data=points, n_clusters=k, init_method=method)
    kmeans.fit(points)

    # Collect the HTML representations of the steps
    plot_htmls = kmeans.snaps

    return jsonify(status="KMeans steps", plot_htmls=plot_htmls)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
