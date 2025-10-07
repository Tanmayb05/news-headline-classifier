# News Category Classifier

A machine learning project that classifies news headlines into four categories using a Bidirectional LSTM neural network built with PyTorch. The project includes a complete web interface built with Streamlit and a REST API using Flask.

## 📊 Categories

The model classifies news headlines into four categories:
- 🎭 **Entertainment** (E)
- 💼 **Business** (B)
- 🔬 **Sci/Tech** (T)
- ❤️ **Health** (M)

## 🏗️ Project Structure

```
project-lstm/
├── src/
│   ├── model.py          # LSTM model architecture
│   ├── train.py          # Model training script
│   ├── api.py            # Flask REST API
│   └── app.py            # Streamlit web interface
├── data/                 # Dataset files
├── saved_models/         # Trained model artifacts
├── outputs/              # Training outputs
├── notebooks/            # Jupyter notebooks
└── requirements.txt      # Python dependencies
```

## 🛠️ Setup & Installation

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd news-headline-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The dataset will be automatically downloaded on first run.

## 🚀 Usage

### Training the Model

Run the training script to train the LSTM model:

```bash
python src/main.py
```

The script will:
- Automatically download the dataset (if not present)
- Load and preprocess the news dataset
- Create data visualizations
- Create train/validation/test splits
- Train the LSTM model
- Save the trained model to `saved_models/`
- Generate training logs in `src/logs/`

### Running the API

Start the Flask API server:

```bash
python src/api.py
```

The API will be available at `http://127.0.0.1:5001`

**API Endpoint:**
- `POST /predict` - Classify a news headline
  ```json
  {
    "text": "Apple announces new iPhone at annual event"
  }
  ```

### Running the Web Interface

Start the Streamlit web application:

```bash
streamlit run src/app.py
```

The web interface will open in your browser, allowing you to:
- Enter news headlines for classification
- View predicted categories with confidence scores
- Interactive and user-friendly interface

## 🧠 Model Architecture

The model uses a **Bidirectional LSTM** architecture:

- **Embedding Layer**: Converts text tokens to dense vectors
- **Bidirectional LSTM**: 2-layer LSTM that processes text in both directions
- **Dropout**: Regularization to prevent overfitting
- **Fully Connected Layer**: Final classification layer

**Key Features:**
- Handles variable-length sequences with padding
- Bidirectional processing for better context understanding
- Dropout regularization for improved generalization

## 📈 Model Performance

The model is trained on a large news dataset with the following configuration:
- Vocabulary size: 20,000 words
- Embedding dimension: 128
- Hidden dimension: 256
- Learning rate: 0.005
- Batch size: 128
- Epochs: 10
- Device: Automatically selects CUDA > MPS > CPU

## 🔧 Technical Stack

- **PyTorch**: Deep learning framework
- **torchtext**: Text preprocessing and data loading
- **Streamlit**: Web interface framework
- **Flask**: REST API framework
- **spaCy**: Natural language processing
- **scikit-learn**: Model evaluation metrics
- **pandas**: Data manipulation
- **matplotlib/seaborn**: Visualization

## 📁 Data

The project uses the UCI News Aggregator dataset containing news headlines from various sources. The data is split into:
- Training set: 80%
- Validation set: 10%
- Test set: 10%

## 🎯 Features

- **Real-time prediction**: Instant classification of news headlines
- **Web interface**: User-friendly Streamlit application
- **REST API**: Programmatic access via Flask API
- **Model persistence**: Trained models are saved and can be reloaded
- **Confidence scores**: Probability estimates for predictions
- **Multi-device support**: Automatically uses CUDA, MPS (Apple Silicon), or CPU
- **Responsive design**: Clean and intuitive user interface

## 🚦 Getting Started

### Quick Start (One Command)
```bash
python run_app.py
```
This script automatically:
- Trains the model if needed
- Starts the API server
- Launches the Streamlit web interface

### Manual Setup
1. **Train the model** first by running `python src/train.py`
2. **Start the API** with `python src/api.py`
3. **Launch the web app** with `streamlit run src/app.py`
4. **Enter a news headline** and get instant classification results!

## 📝 Requirements

- Python 3.7+
- PyTorch 2.8.0+
- See `requirements.txt` for complete dependency list

## 🚀 Deployment

### GitHub Deployment

**Simple GitHub Setup:**
1. **Fork/Clone** the repository to GitHub
2. **Include model files** (already configured in `.gitignore`)
3. **Clone and run locally**:
   ```bash
   git clone https://github.com/yourusername/project-lstm.git
   cd project-lstm
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python run_app.py
   ```
4. **GitHub Pages** automatically deploys documentation

### GitLab CI/CD

This project includes GitLab CI/CD configuration for automated deployment:

1. **Push to GitLab**: The pipeline automatically runs on push to main branch
2. **Stages**:
   - `test`: Run tests and validation
   - `build`: Build Docker image and push to GitLab Container Registry
   - `deploy`: Deploy to production (manual trigger)

### Docker Deployment

**Build and run with Docker:**
```bash
# Build the image
docker build -t news-classifier .

# Run the container
docker run -p 8501:8501 -p 5001:5001 news-classifier
```

**Or use Docker Compose:**
```bash
docker-compose up -d
```

Access the app at:
- Streamlit UI: http://localhost:8501
- API: http://localhost:5001

### GitLab Container Registry

To use GitLab's container registry:

1. **Enable Container Registry** in your GitLab project settings
2. **Configure CI/CD variables**:
   - `CI_REGISTRY_USER`: Your GitLab username
   - `CI_REGISTRY_PASSWORD`: Personal access token with registry permissions
3. **Push to main branch** to trigger the pipeline

### Cloud Deployment Options

**GitLab Pages**: Automatically deploys documentation
**Heroku**: Use the included Dockerfile
**AWS/GCP/Azure**: Deploy using container services
**DigitalOcean App Platform**: Direct GitLab integration

### Environment Variables

For production deployment, configure:
- `PYTHONPATH=/app`
- `PYTHONUNBUFFERED=1`

## 🤝 Contributing

Feel free to open issues and submit pull requests to improve the project.

## 📄 License

This project is open source and available under the MIT License.