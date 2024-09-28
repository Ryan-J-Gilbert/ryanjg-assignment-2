from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # To run Matplotlib without a display server
import matplotlib.pyplot as plt
from kmeans import KMeans  # Your custom KMeans implementation
import plotly.graph_objects as go
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    # Generate a new random dataset (e.g., 2D points)
    points = np.random.uniform(-10, 10, (500, 2))  # 500 random points
    np.save('dataset.npy', points)  # Save the dataset so it persists across steps
    
    # Create a Plotly scatter plot
    fig = go.Figure(data=go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers'))
    plot_html = fig.to_html(full_html=False)

    return render_template('index.html', plot_html=plot_html, k=3, method='Random')

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    # k = int(request.form['k'])
    # method = request.form['init_method']  # Random, Farthest, KMeans++ or Manual

    # # Load the dataset
    # points = np.load('dataset.npy')

    # # Initialize and run the KMeans algorithm
    # kmeans = KMeans(data=points, n_clusters=k, init_method=method)
    # kmeans.fit(points)

    # # Generate the final plot
    # plot_html = kmeans.plot()

    # return render_template('index.html', plot_html=plot_html, k=k, method=method)
    k = int(request.form['k'])
    method = request.form['init_method']  # Random, Farthest, KMeans++ or Manual

    # Load the dataset
    points = np.load('dataset.npy')

    # Initialize and run the KMeans algorithm
    kmeans = KMeans(data=points, n_clusters=k, init_method=method)
    kmeans.lloyds(method)

    while kmeans.step():
        pass

    # return jsonify(plot_htmls=[plot_html])
    return render_template('index.html', plot_html=kmeans.snaps[-1], k=k, method=method)

@app.route('/step', methods=['POST'])
def step():
    k = int(request.form['k'])
    method = request.form['init_method']  # Random, Farthest, KMeans++ or Manual
    step_count = int(request.form['step_count']) # need to implement this in the front end

    # app will save the list of htmls in the backend and send the next html to the front end

    if step_count == -1:
        # we need to make the model and the snaps
        # Load the dataset
        points = np.load('dataset.npy')

        # Initialize and run the KMeans algorithm
        kmeans = KMeans(data=points, n_clusters=k, init_method=method)
        kmeans.lloyds(method)

        while kmeans.step():
            pass

        with open('snaps.json', 'w', encoding='utf-8') as f:
            json.dump(kmeans.snaps, f)
        
        snaps = kmeans.snaps
        print(len(snaps))
    else:
        # we need to load the snaps and send the next snap

        with open('snaps.json', 'r', encoding='utf-8') as f:
            snaps = json.load(f)

            if step_count == len(snaps):
                return render_template('index.html', plot_html=snaps[step_count-1], k=k, method=method, step_count=step_count, warning=True)
            return render_template('index.html', plot_html=snaps[step_count], k=k, method=method, step_count=step_count+1)
        

    return render_template('index.html', plot_html=snaps[step_count], k=k, method=method, step_count=step_count+1)

@app.route('/reset_kmeans', methods=['POST'])
def reset_kmeans():
    # simply load the dataset and return the initial plot
    points = np.load('dataset.npy')
    fig = go.Figure(data=go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers'))
    plot_html = fig.to_html(full_html=False)

    return render_template('index.html', plot_html=plot_html, k=3, method='Random')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)