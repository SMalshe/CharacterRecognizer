# Character Recognizer

A handwritten character recognition system using Convolutional Neural Networks (CNN) trained on the EMNIST dataset.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/SMalshe/CharacterRecognizer.git
cd CharacterRecognizer
```

2. Create a virtual environment:
```bash
python3 -m venv .venv1
source .venv1/bin/activate  # On Windows: .venv1\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python ui.py
```

## Files

- `ConvolutionalNeuralNetwork.ipynb` - Jupyter notebook with model training
- `ConvolutionalNeuralNetwork.py` - Python script version
- `ui.py` - Streamlit-based user interface for character recognition
- `handwritten_characters.keras` - Trained model
- `gzip/` - EMNIST dataset files(only emnist-balanced files)

## Requirements

- Python 3.10+
- TensorFlow 2.20.0
- OpenCV 4.12.0
- NumPy 2.2.6
- Matplotlib 3.10.8
