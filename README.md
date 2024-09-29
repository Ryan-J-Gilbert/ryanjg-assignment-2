# KMeans Clustering Visualization

This project is a KMeans clustering visualization tool built using Flask for the backend and plotly for generating charts. Users can interact with the visualization by selecting the initial centroids for the KMeans algorithm, using either automatic or manual initialization.

## Features
- Visualizes KMeans clustering step by step.
- Allows users to select initial centroids manually or automatically.
- Real-time updates as the clustering algorithm progresses.

## Demo

You can view a demo of the project on [YouTube](https://youtu.be/FdcK7zrxuYQ).

[![KMeans Clustering Visualization](https://img.youtube.com/vi/FdcK7zrxuYQ/0.jpg)](https://youtu.be/FdcK7zrxuYQ)
## Setup and Installation

### Requirements
- Python 3.10+
- Make

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ryan-J-Gilbert/ryanjg-assignment-2.git
   cd ryanjg-assignment-2
   ```

2. Set up a virtual environment and install the required dependencies:
   ```bash
   make install
   ```

### Running the Application

To run the application, use the following command:
```bash
make run
```

This will start the Flask server. You can then open your browser and go to `http://localhost:3000` to interact with the app.