# CaNeSy-eNose: Causal Neuro-Symbolic Agentic Electronic Nose

> Drift-Aware Gas Classification using Deep Learning, Explainable AI, and Causal Reasoning

---

# Overview

CaNeSy-eNose is an intelligent gas sensing system developed for drift-aware gas classification using deep learning, explainable artificial intelligence (XAI), and causal inference.

Traditional electronic nose systems often suffer from sensor drift, where sensor responses change over time due to aging, environmental conditions, or calibration instability. This project addresses those limitations by integrating:

- Transformer-based neural learning
- Drift-aware preprocessing
- Explainable AI methods
- Causal reasoning
- Interactive deployment using Streamlit

This system is designed for research, experimentation, and deployment of interpretable gas classification models.

---

# Key Features

- Drift-aware gas classification
- Transformer-based deep learning architecture
- Explainable predictions using SHAP and attention analysis
- Causal effect interpretation
- Streamlit-based interactive interface
- Portable deployment-ready structure

---

# Project Structure

```text
CaNeSy-eNose/
│
├── app.py                         # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── sensor_transformer_xai.pth    # Trained Transformer model
├── X_test_tensor.pt              # Test tensor dataset
├── causal_effect.pkl             # Causal reasoning object
├── utils.py                      # Utility/helper functions
├── assets/                       # Images, logos, figures
├── notebooks/                    # Training notebooks
└── models/                       # Additional model files
```

---

# Installation Guide

## Step 1: Clone the Repository

```bash
git clone https://github.com/deusrapha/Canesy-eNose.git
cd Canesy-eNose
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

After launching, the application will automatically open in your browser.

Default local address:

```text
http://localhost:8501
```

---

# Required Files

Ensure these files exist inside your project root folder:

| File | Description |
|------|-------------|
| sensor_transformer_xai.pth | Trained deep learning model |
| X_test_tensor.pt | Test data tensor |
| causal_effect.pkl | Causal reasoning object |
| app.py | Streamlit application |
| requirements.txt | Dependencies |

---

# Example Model Loading

```python
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)

    model = SensorTransformerXAI()

    model.load_state_dict(
        torch.load(
            os.path.join(base, "sensor_transformer_xai.pth"),
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    X_test_tensor = torch.load(
        os.path.join(base, "X_test_tensor.pt"),
        map_location=device
    )

    with open(os.path.join(base, "causal_effect.pkl"), "rb") as f:
        causal = pickle.load(f)

    return model, X_test_tensor, causal
```

---

# Deployment Options

You can deploy CaNeSy-eNose using:

### Option 1: GitHub + Streamlit Cloud

1. Push project to GitHub
2. Visit Streamlit Community Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Deploy

### Option 2: Render

- Upload repository
- Configure Python environment
- Deploy as web service

### Option 3: Hugging Face Spaces

- Create a Space
- Select Streamlit SDK
- Upload repository

### Option 4: Docker Deployment

Useful for reproducible deployment environments.

---

# Recommended requirements.txt

```text
streamlit
torch
numpy
pandas
matplotlib
scikit-learn
shap
pickle-mixin
plotly
joblib
```

---

# Future Improvements

Potential future work includes:

- Real-time sensor integration
- Multi-sensor fusion
- Drift adaptation learning
- Edge-device deployment
- Neuro-symbolic reasoning extension
- Live XAI dashboard

---

# Research Contribution

This project contributes toward:

- Robust gas sensing under sensor drift
- Explainable machine learning
- Causal AI for trustworthy predictions
- Intelligent electronic nose systems

---

# Citation

If you use this work in academic research, cite:

```text
Tumusiime, R.D., et al.
CaNeSy-eNose: A Causal Neuro-Symbolic Agentic Electronic Nose for Drift-Aware Gas Classification.
```

---

# Author

Deus Tumusiime R
MSc Research Project  
Makerere University  

---

# License

This project is intended for academic and research use.

You may extend or modify the work with proper attribution.

---

# Contact

Email: deus.mal@gmail.com  
GitHub: https://github.com/deusrapha

