# README 
CS 613 — Machine Learning Final Project  
**Real vs AI-Generated image Classification**

## 1. Environment Setup
This project was implemented in **Google Colab** (GPU runtime recommended).

### Install Required Libraries
```
pip install facenet-pytorch
pip install scikit-learn
pip install joblib
pip install tqdm
pip install seaborn
```

## 2. Dataset
We use the Kaggle dataset:

**140K Real and Fake Faces**  
[https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces]

Download in Colab:
```python
import kagglehub
path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
print(path)
```


## 3. Download Precomputed Embeddings

To reproduce our results **without re-running the 2-hour embedding extraction**, download:

### • FaceNet Embeddings (embeddings.npz)  
Download: [https://drive.google.com/file/d/1Ge5_Pv1I3RpjBm_ZjRereacgTLbUBPTd/view?usp=sharing]

After downloading, place file inside:
```
/content/drive/MyDrive/CS613/
```

Update paths in notebook:
```python
EMBED_SAVE = "/content/drive/MyDrive/CS613/embeddings.npz"
```

## 4. Load the Embeddings
```python
X, y, _ = extract_embeddings_mtcnn(
    paths, labels, device=Device, force_reextract=False
)
```

You should see:
```
Loaded cached embeddings — model loaded.
```
**Do NOT set `force_reextract=True`**  
This will trigger a full re-run and take **1.5–2 hours**.


## 5. Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```


## 6. Preprocessing
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=512)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

## 7. Training the Models
Models trained:

- Logistic Regression  
- Linear SVM with probability calibration  
- Random Forest  
- Voting Ensemble  

## 8. Saving and Loading the SVM Model
- Trained SVM Model (svm_model.joblib)  
Download: [https://drive.google.com/file/d/1_0w00ta3b5dHsVvZW0e_DwX75niGsP87/view?usp=sharing]

- Update path in notebook:
```python
MODEL_PATH = "/content/drive/MyDrive/CS613/svm_model.joblib"
```
- Load Pretrained SVM Model
```python
from joblib import load
svm = load(MODEL_PATH)
```


## 9. Evaluate Models
Run the notebook cells to produce:

- Logistic Regression results  
- SVM results  
- Random Forest results  
- Ensemble accuracy comparison  
- Confusion matrices  


## 9. Predict on New Images
Upload an image to Colab and run:

```python
prediction = predict_embedding("/content/myimage.jpg", svm, Device)
print(prediction)
```

Example Output:
```
{
 "Prediction": "Fake",
 "Fake Probability": 0.57,
 "Real Probability": 0.30
}
```

## 10. File Links 

```
Embeddings File:
https://drive.google.com/file/d/1Ge5_Pv1I3RpjBm_ZjRereacgTLbUBPTd/view?usp=sharing

SVM Model:
https://drive.google.com/file/d/1_0w00ta3b5dHsVvZW0e_DwX75niGsP87/view?usp=sharing
```


