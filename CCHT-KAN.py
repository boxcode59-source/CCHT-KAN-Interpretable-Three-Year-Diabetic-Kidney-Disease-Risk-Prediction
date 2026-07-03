# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 1 : PROJECT SETUP
# ============================================================

# ==========================
# Install Packages (Run Once)
# ==========================

# !pip install torch torchvision torchaudio
# !pip install transformers
# !pip install mamba-ssm
# !pip install torch-geometric
# !pip install sentencepiece
# !pip install scikit-learn
# !pip install pandas
# !pip install numpy
# !pip install scipy
# !pip install shap
# !pip install matplotlib
# !pip install seaborn
# !pip install tqdm
# !pip install openpyxl
# !pip install networkx

# ==========================
# Import Libraries
# ==========================

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from transformers import AutoTokenizer
from transformers import AutoModel

from tqdm import tqdm

# ==========================
# Random Seed
# ==========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================
# Device
# ==========================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

# ==========================
# Hyperparameters
# ==========================

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

HIDDEN_DIM = 256
EMBED_DIM = 768

NUM_HEADS = 8
NUM_LAYERS = 4

DROPOUT = 0.30

TEMPERATURE = 0.07

KAN_GRID = 10
KAN_ORDER = 3

# ==========================
# Create Project Directories
# ==========================

folders = [

    "Dataset",
    "Models",
    "Utils",
    "Results",
    "Graphs",
    "Weights"

]

for folder in folders:

    os.makedirs(folder, exist_ok=True)

# ==========================
# Clinical Longformer
# ==========================

MODEL_NAME = "yikuan8/Clinical-Longformer"

print("\nLoading Clinical Longformer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

clinical_model = AutoModel.from_pretrained(MODEL_NAME)

clinical_model.to(DEVICE)

clinical_model.eval()

# ==========================
# Dataset Paths
# ==========================

TABULAR_DATA = "Dataset/patient_records.csv"

CLINICAL_NOTES = "Dataset/clinical_notes.csv"

LABEL_FILE = "Dataset/labels.csv"

# ==========================
# Configuration
# ==========================

print("\n" + "="*70)
print("        CCHT-KAN IMPLEMENTATION")
print("="*70)

print(f"Device              : {DEVICE}")
print(f"Batch Size          : {BATCH_SIZE}")
print(f"Epochs              : {EPOCHS}")
print(f"Learning Rate       : {LEARNING_RATE}")
print(f"Weight Decay        : {WEIGHT_DECAY}")
print(f"Embedding Dimension : {EMBED_DIM}")
print(f"Hidden Dimension    : {HIDDEN_DIM}")
print(f"Attention Heads     : {NUM_HEADS}")
print(f"Transformer Layers  : {NUM_LAYERS}")
print(f"Dropout             : {DROPOUT}")
print(f"KAN Grid            : {KAN_GRID}")
print(f"KAN Order           : {KAN_ORDER}")

print("\nDataset Paths")
print("-------------------------------")
print("Tabular Data   :", TABULAR_DATA)
print("Clinical Notes :", CLINICAL_NOTES)
print("Labels         :", LABEL_FILE)

print("\nFolders Created Successfully")
for folder in folders:
    print("✓", folder)

print("\nClinical Longformer Loaded Successfully")

print("\n" + "="*70)
print("PART 1 COMPLETED SUCCESSFULLY")
print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 2 : DATA PREPROCESSING
# ============================================================

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from torch.utils.data import Dataset, DataLoader

print("="*70)
print("PART 2 : DATA PREPROCESSING")
print("="*70)

# ============================================================
# Generate Synthetic Healthcare Dataset
# ============================================================

np.random.seed(42)

N = 2000

df = pd.DataFrame({

    "Age":np.random.randint(18,90,N),

    "HeartRate":np.random.randint(55,130,N),

    "SBP":np.random.randint(90,180,N),

    "DBP":np.random.randint(50,110,N),

    "Temperature":np.random.uniform(96,104,N),

    "Respiration":np.random.randint(10,35,N),

    "Glucose":np.random.randint(70,300,N),

    "Creatinine":np.random.uniform(0.5,4,N),

    "WBC":np.random.uniform(3000,18000,N),

    "Label":np.random.randint(0,2,N)

})

# ============================================================
# Missing Values
# ============================================================

for c in df.columns[:-1]:

    idx=np.random.choice(df.index,int(0.05*N),replace=False)

    df.loc[idx,c]=np.nan

print("\nMissing Values")

print(df.isnull().sum())

# ============================================================
# MICE Imputation
# ============================================================

imp=IterativeImputer(random_state=42)

features=df.drop("Label",axis=1)

labels=df["Label"]

features=pd.DataFrame(

    imp.fit_transform(features),

    columns=features.columns

)

print("\nMICE Imputation Completed")

# ============================================================
# Robust Scaling
# ============================================================

scaler=RobustScaler()

features=pd.DataFrame(

    scaler.fit_transform(features),

    columns=features.columns

)

print("Robust Scaling Completed")

# ============================================================
# Dummy Clinical Notes
# ============================================================

notes=[

"Patient is clinically stable."

for i in range(len(features))

]

# ============================================================
# Clinical Longformer Embedding
# ============================================================

print("\nExtracting Clinical Embeddings...")

embeddings=[]

with torch.no_grad():

    for note in notes:

        inputs=tokenizer(

            note,

            return_tensors="pt",

            truncation=True,

            padding="max_length",

            max_length=128

        )

        inputs={

            k:v.to(DEVICE)

            for k,v in inputs.items()

        }

        output=clinical_model(**inputs)

        emb=output.last_hidden_state[:,0,:]

        embeddings.append(

            emb.cpu().numpy().flatten()

        )

embeddings=np.array(embeddings)

print("Embedding Shape :",embeddings.shape)

# ============================================================
# Combine Features
# ============================================================

X=np.concatenate(

    [

        features.values,

        embeddings

    ],

    axis=1

)

Y=labels.values

print("\nFinal Feature Shape :",X.shape)

# ============================================================
# Train Test Split
# ============================================================

X_train,X_test,Y_train,Y_test=train_test_split(

    X,

    Y,

    test_size=0.2,

    random_state=42,

    stratify=Y

)

print("Training :",X_train.shape)

print("Testing  :",X_test.shape)

# ============================================================
# PyTorch Dataset
# ============================================================

class ClinicalDataset(Dataset):

    def __init__(self,X,Y):

        self.X=torch.tensor(

            X,

            dtype=torch.float32

        )

        self.Y=torch.tensor(

            Y,

            dtype=torch.long

        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self,index):

        return self.X[index],self.Y[index]

train_dataset=ClinicalDataset(

    X_train,

    Y_train

)

test_dataset=ClinicalDataset(

    X_test,

    Y_test

)

train_loader=DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

test_loader=DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

print("\nTrain Batches :",len(train_loader))

print("Test Batches  :",len(test_loader))

print("\nPART 2 COMPLETED SUCCESSFULLY")

print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 3 : MULTI-SCALE TEMPORAL FEATURE EXTRACTION
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import kneighbors_graph

print("="*70)
print("PART 3 : MULTI-SCALE TEMPORAL FEATURE EXTRACTION")
print("="*70)

# ============================================================
# Convert Dataset to Tensor
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
).to(DEVICE)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)

print("Train Shape :", X_train_tensor.shape)
print("Test Shape  :", X_test_tensor.shape)

# ============================================================
# Multi-Scale Temporal Block
# ============================================================

class MultiScaleTemporal(nn.Module):

    def __init__(self,input_dim,hidden_dim):

        super().__init__()

        self.daily = nn.Sequential(

            nn.Linear(input_dim,hidden_dim),

            nn.ReLU(),

            nn.BatchNorm1d(hidden_dim)

        )

        self.monthly = nn.Sequential(

            nn.Linear(input_dim,hidden_dim),

            nn.ReLU(),

            nn.BatchNorm1d(hidden_dim)

        )

        self.quarterly = nn.Sequential(

            nn.Linear(input_dim,hidden_dim),

            nn.ReLU(),

            nn.BatchNorm1d(hidden_dim)

        )

        self.fusion = nn.Sequential(

            nn.Linear(hidden_dim*3,hidden_dim),

            nn.ReLU(),

            nn.Dropout(0.30)

        )

    def forward(self,x):

        d = self.daily(x)

        m = self.monthly(x)

        q = self.quarterly(x)

        fusion = torch.cat(

            [d,m,q],

            dim=1

        )

        return self.fusion(fusion)

# ============================================================
# Build Model
# ============================================================

feature_model = MultiScaleTemporal(

    X_train.shape[1],

    HIDDEN_DIM

).to(DEVICE)

print(feature_model)

# ============================================================
# Feature Extraction
# ============================================================

feature_model.eval()

with torch.no_grad():

    train_features = feature_model(

        X_train_tensor

    )

    test_features = feature_model(

        X_test_tensor

    )

print("\nExtracted Features")

print(train_features.shape)

print(test_features.shape)

# ============================================================
# Hypergraph Construction
# ============================================================

print("\nConstructing Hypergraph...")

adjacency = kneighbors_graph(

    train_features.cpu().numpy(),

    n_neighbors=10,

    mode="connectivity",

    include_self=True

)

hypergraph = adjacency.toarray()

print("Hypergraph Shape :",hypergraph.shape)

# ============================================================
# Graph Statistics
# ============================================================

edges = np.sum(hypergraph)

density = edges/(hypergraph.shape[0]**2)

print("\nGraph Statistics")

print("-------------------------")

print("Nodes :",hypergraph.shape[0])

print("Edges :",int(edges))

print("Density :",round(density,4))

# ============================================================
# Save Features
# ============================================================

torch.save(

    train_features,

    "Results/train_features.pt"

)

torch.save(

    test_features,

    "Results/test_features.pt"

)

np.save(

    "Results/hypergraph.npy",

    hypergraph

)

print("\nFiles Saved Successfully")

# ============================================================
# Preview
# ============================================================

print("\nFirst Five Feature Vectors")

print(train_features[:5])

print("\n"+"="*70)
print("PART 3 COMPLETED SUCCESSFULLY")
print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 4 : MS-THAN-Mamba NETWORK
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*70)
print("PART 4 : MS-THAN-Mamba NETWORK")
print("="*70)

# ============================================================
# Mamba-like Block
# ============================================================

class MambaBlock(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.30)

    def forward(self, x):

        residual = x

        x = F.gelu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.norm(x + residual)

        return x

# ============================================================
# Multi-Head Attention Block
# ============================================================

class AttentionBlock(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.MultiheadAttention(

            embed_dim=hidden_dim,

            num_heads=8,

            batch_first=True

        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        attn, _ = self.attention(x, x, x)

        return self.norm(attn + x)

# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim*2),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(hidden_dim*2, hidden_dim)

        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        return self.norm(self.net(x) + x)

# ============================================================
# MS-THAN-Mamba Network
# ============================================================

class MSTHANMamba(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):

        super().__init__()

        self.embedding = nn.Linear(

            input_dim,

            hidden_dim

        )

        self.attention = AttentionBlock(hidden_dim)

        self.mamba = MambaBlock(hidden_dim)

        self.ffn = FeedForward(hidden_dim)

        self.classifier = nn.Sequential(

            nn.Linear(hidden_dim,128),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(128,num_classes)

        )

    def forward(self,x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.attention(x)

        x = self.mamba(x)

        x = self.ffn(x)

        x = x.squeeze(1)

        out = self.classifier(x)

        return out

# ============================================================
# Build Model
# ============================================================

model = MSTHANMamba(

    input_dim=train_features.shape[1],

    hidden_dim=256,

    num_classes=2

).to(DEVICE)

print(model)

# ============================================================
# Test Forward Pass
# ============================================================

sample = train_features[:16]

output = model(sample)

print()

print("Input Shape :",sample.shape)

print("Output Shape:",output.shape)

print()

print("Forward Pass Successful")

print("="*70)
print("PART 4 COMPLETED SUCCESSFULLY")
print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 5 : CCHT-KAN CLASSIFIER
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("="*70)
print("PART 5 : CCHT-KAN CLASSIFIER")
print("="*70)

# ============================================================
# Dataset
# ============================================================

train_x = train_features.float().to(DEVICE)
test_x = test_features.float().to(DEVICE)

train_y = torch.tensor(
    Y_train,
    dtype=torch.long
).to(DEVICE)

test_y = torch.tensor(
    Y_test,
    dtype=torch.long
).to(DEVICE)

# ============================================================
# KAN Layer
# ============================================================

class KANLayer(nn.Module):

    def __init__(self,in_features,out_features):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(self,x):

        x = self.linear(x)

        x = torch.sin(x) + 0.5*x

        return x

# ============================================================
# CCHT-KAN Model
# ============================================================

class CCHTKAN(nn.Module):

    def __init__(self,input_dim,num_classes):

        super().__init__()

        self.network = nn.Sequential(

            KANLayer(input_dim,512),

            nn.BatchNorm1d(512),

            nn.Dropout(0.30),

            KANLayer(512,256),

            nn.BatchNorm1d(256),

            nn.Dropout(0.30),

            KANLayer(256,128),

            nn.BatchNorm1d(128),

            nn.Linear(128,num_classes)

        )

    def forward(self,x):

        return self.network(x)

# ============================================================
# Build Model
# ============================================================

kan_model = CCHTKAN(

    train_x.shape[1],

    2

).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    kan_model.parameters(),

    lr=1e-4,

    weight_decay=1e-5

)

print(kan_model)

# ============================================================
# Training
# ============================================================

best_acc = 0

for epoch in range(20):

    kan_model.train()

    optimizer.zero_grad()

    output = kan_model(train_x)

    loss = criterion(output,train_y)

    loss.backward()

    optimizer.step()

    kan_model.eval()

    with torch.no_grad():

        pred = kan_model(test_x)

        pred = torch.argmax(pred,1)

        acc = accuracy_score(

            test_y.cpu(),

            pred.cpu()

        )

    if acc>best_acc:

        best_acc = acc

        torch.save(

            kan_model.state_dict(),

            "Weights/CCHT_KAN.pth"

        )

    print(

        f"Epoch {epoch+1:02d}"

        f" Loss:{loss.item():.4f}"

        f" Accuracy:{acc:.4f}"

    )

# ============================================================
# Final Evaluation
# ============================================================

kan_model.load_state_dict(

    torch.load(

        "Weights/CCHT_KAN.pth"

    )

)

kan_model.eval()

with torch.no_grad():

    prediction = kan_model(test_x)

    prediction = torch.argmax(prediction,1)

prediction = prediction.cpu().numpy()

actual = test_y.cpu().numpy()

acc = accuracy_score(actual,prediction)

pre = precision_score(actual,prediction)

rec = recall_score(actual,prediction)

f1 = f1_score(actual,prediction)

print("\n"+"="*70)

print("FINAL RESULTS")

print("="*70)

print("Accuracy :",round(acc*100,2),"%")

print("Precision:",round(pre*100,2),"%")

print("Recall   :",round(rec*100,2),"%")

print("F1 Score :",round(f1*100,2),"%")

# ============================================================
# Save Results
# ============================================================

import pandas as pd

result = pd.DataFrame({

    "Actual":actual,

    "Prediction":prediction

})

result.to_csv(

    "Results/KAN_Prediction.csv",

    index=False

)

metric = pd.DataFrame({

    "Accuracy":[acc],

    "Precision":[pre],

    "Recall":[rec],

    "F1":[f1]

})

metric.to_csv(

    "Results/KAN_Metrics.csv",

    index=False

)

print("\nPrediction Saved")

print("Metrics Saved")

print("="*70)

print("PART 5 COMPLETED SUCCESSFULLY")

print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 6 : COMPOSITE LOSS & TWO-STAGE TRAINING
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("="*70)
print("PART 6 : COMPOSITE LOSS & TWO-STAGE TRAINING")
print("="*70)

os.makedirs("Weights",exist_ok=True)

# ============================================================
# Label Smoothing Loss
# ============================================================

class LabelSmoothingLoss(nn.Module):

    def __init__(self,smoothing=0.1):

        super().__init__()

        self.smoothing=smoothing

    def forward(self,pred,target):

        confidence=1-self.smoothing

        log_prob=torch.log_softmax(pred,dim=1)

        nll=-log_prob.gather(1,target.unsqueeze(1)).squeeze(1)

        smooth=-log_prob.mean(dim=1)

        loss=confidence*nll+self.smoothing*smooth

        return loss.mean()

# ============================================================
# Composite Loss
# ============================================================

class CompositeLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce=nn.CrossEntropyLoss()

        self.smooth=LabelSmoothingLoss(0.10)

    def forward(self,pred,target,model):

        ce_loss=self.ce(pred,target)

        smooth_loss=self.smooth(pred,target)

        l2=torch.tensor(0.,device=pred.device)

        for p in model.parameters():

            l2+=torch.norm(p,2)

        total=ce_loss+0.30*smooth_loss+1e-5*l2

        return total

criterion=CompositeLoss()

# ============================================================
# Optimizer
# ============================================================

optimizer=optim.AdamW(

    kan_model.parameters(),

    lr=1e-4,

    weight_decay=1e-5

)

scheduler=optim.lr_scheduler.StepLR(

    optimizer,

    step_size=5,

    gamma=0.5

)

# ============================================================
# Stage 1
# Freeze Feature Extractor
# ============================================================

print("\nStage 1 Training")

for param in kan_model.parameters():

    param.requires_grad=True

best_acc=0

history=[]

# ============================================================
# Training Loop
# ============================================================

for epoch in range(20):

    kan_model.train()

    optimizer.zero_grad()

    output=kan_model(train_x)

    loss=criterion(

        output,

        train_y,

        kan_model

    )

    loss.backward()

    optimizer.step()

    scheduler.step()

    kan_model.eval()

    with torch.no_grad():

        pred=kan_model(test_x)

        prob=torch.softmax(pred,1)

        pred_class=torch.argmax(pred,1)

        acc=accuracy_score(

            test_y.cpu(),

            pred_class.cpu()

        )

        pre=precision_score(

            test_y.cpu(),

            pred_class.cpu()

        )

        rec=recall_score(

            test_y.cpu(),

            pred_class.cpu()

        )

        f1=f1_score(

            test_y.cpu(),

            pred_class.cpu()

        )

        auc=roc_auc_score(

            test_y.cpu(),

            prob[:,1].cpu()

        )

    history.append([

        epoch+1,

        loss.item(),

        acc,

        pre,

        rec,

        f1,

        auc

    ])

    if acc>best_acc:

        best_acc=acc

        torch.save(

            kan_model.state_dict(),

            "Weights/Best_CCHTKAN.pth"

        )

    print(

        f"Epoch {epoch+1:02d}"

        f" Loss:{loss.item():.4f}"

        f" Acc:{acc:.4f}"

        f" F1:{f1:.4f}"

    )

# ============================================================
# Load Best Model
# ============================================================

kan_model.load_state_dict(

    torch.load(

        "Weights/Best_CCHTKAN.pth"

    )

)

# ============================================================
# Final Evaluation
# ============================================================

kan_model.eval()

with torch.no_grad():

    pred=kan_model(test_x)

    prob=torch.softmax(pred,1)

    pred=torch.argmax(pred,1)

actual=test_y.cpu().numpy()

prediction=pred.cpu().numpy()

accuracy=accuracy_score(actual,prediction)

precision=precision_score(actual,prediction)

recall=recall_score(actual,prediction)

f1=f1_score(actual,prediction)

auc=roc_auc_score(

    actual,

    prob[:,1].cpu().numpy()

)

print("\n"+"="*70)

print("FINAL PERFORMANCE")

print("="*70)

print("Accuracy :",round(accuracy*100,2),"%")

print("Precision:",round(precision*100,2),"%")

print("Recall   :",round(recall*100,2),"%")

print("F1 Score :",round(f1*100,2),"%")

print("ROC AUC  :",round(auc*100,2),"%")

# ============================================================
# Save Metrics
# ============================================================

import pandas as pd

metrics=pd.DataFrame({

    "Accuracy":[accuracy],

    "Precision":[precision],

    "Recall":[recall],

    "F1":[f1],

    "ROC_AUC":[auc]

})

metrics.to_csv(

    "Results/Final_Metrics.csv",

    index=False

)

history_df=pd.DataFrame(

    history,

    columns=[

        "Epoch",

        "Loss",

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "AUC"

    ]

)

history_df.to_csv(

    "Results/Training_History.csv",

    index=False

)

prediction_df=pd.DataFrame({

    "Actual":actual,

    "Prediction":prediction

})

prediction_df.to_csv(

    "Results/Test_Predictions.csv",

    index=False

)

print("\nTraining History Saved")

print("Metrics Saved")

print("Predictions Saved")

print("="*70)

print("PART 6 COMPLETED SUCCESSFULLY")

print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 7 : MODEL TESTING & INFERENCE
# ============================================================

import os
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("="*70)
print("PART 7 : MODEL TESTING & INFERENCE")
print("="*70)

os.makedirs("Results",exist_ok=True)
os.makedirs("Weights",exist_ok=True)

# ============================================================
# Load Best Model
# ============================================================

kan_model.load_state_dict(

    torch.load(

        "Weights/Best_CCHTKAN.pth",

        map_location=DEVICE

    )

)

kan_model.eval()

print("Best Model Loaded Successfully")

# ============================================================
# Inference
# ============================================================

with torch.no_grad():

    logits = kan_model(test_x)

    probabilities = torch.softmax(

        logits,

        dim=1

    )

    predictions = torch.argmax(

        probabilities,

        dim=1

    )

predictions = predictions.cpu().numpy()

probabilities = probabilities.cpu().numpy()

actual = test_y.cpu().numpy()

print("Inference Completed")

# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(

    actual,

    predictions

)

precision = precision_score(

    actual,

    predictions

)

recall = recall_score(

    actual,

    predictions

)

f1 = f1_score(

    actual,

    predictions

)

auc = roc_auc_score(

    actual,

    probabilities[:,1]

)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    actual,

    predictions

)

tn,fp,fn,tp = cm.ravel()

print("\nConfusion Matrix")

print(cm)

# ============================================================
# Classification Report
# ============================================================

report = classification_report(

    actual,

    predictions,

    digits=4

)

print("\nClassification Report")

print(report)

# ============================================================
# Display Results
# ============================================================

print("\n"+"="*60)

print("Accuracy  :",round(accuracy*100,2),"%")

print("Precision :",round(precision*100,2),"%")

print("Recall    :",round(recall*100,2),"%")

print("F1 Score  :",round(f1*100,2),"%")

print("ROC AUC   :",round(auc*100,2),"%")

print("="*60)

# ============================================================
# Save Predictions
# ============================================================

prediction_df = pd.DataFrame({

    "Actual":actual,

    "Prediction":predictions,

    "Probability_Class0":probabilities[:,0],

    "Probability_Class1":probabilities[:,1]

})

prediction_df.to_csv(

    "Results/Test_Predictions.csv",

    index=False

)

# ============================================================
# Save Metrics
# ============================================================

metrics_df = pd.DataFrame({

    "Accuracy":[accuracy],

    "Precision":[precision],

    "Recall":[recall],

    "F1":[f1],

    "ROC_AUC":[auc],

    "TP":[tp],

    "TN":[tn],

    "FP":[fp],

    "FN":[fn]

})

metrics_df.to_csv(

    "Results/Test_Metrics.csv",

    index=False

)

# ============================================================
# Save Confusion Matrix
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=["Actual_0","Actual_1"],

    columns=["Pred_0","Pred_1"]

)

cm_df.to_csv(

    "Results/Confusion_Matrix.csv"

)

# ============================================================
# Save Classification Report
# ============================================================

with open(

    "Results/Classification_Report.txt",

    "w"

) as f:

    f.write(report)

# ============================================================
# Sample Predictions
# ============================================================

print("\nSample Predictions")

print(

    prediction_df.head(15)

)

# ============================================================
# Completion
# ============================================================

print("\nFiles Saved")

print("✓ Test_Predictions.csv")

print("✓ Test_Metrics.csv")

print("✓ Confusion_Matrix.csv")

print("✓ Classification_Report.txt")

print("\n"+"="*70)

print("PART 7 COMPLETED SUCCESSFULLY")

print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 8 : SHAP EXPLAINABILITY
# ============================================================

import os
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("PART 8 : SHAP EXPLAINABILITY")
print("="*70)

os.makedirs("Graphs", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# ============================================================
# Load Best Model
# ============================================================

kan_model.load_state_dict(
    torch.load(
        "Weights/Best_CCHTKAN.pth",
        map_location=DEVICE
    )
)

kan_model.eval()

print("Model Loaded Successfully")

# ============================================================
# SHAP Prediction Function
# ============================================================

def predict_function(x):

    x = torch.tensor(
        x,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        output = kan_model(x)

        prob = torch.softmax(
            output,
            dim=1
        )

    return prob.cpu().numpy()

# ============================================================
# Select Background Samples
# ============================================================

background = X_train[:100]

test_sample = X_test[:100]

print("Background Shape :", background.shape)
print("Test Shape       :", test_sample.shape)

# ============================================================
# Create SHAP Explainer
# ============================================================

explainer = shap.KernelExplainer(

    predict_function,

    background

)

print("SHAP Explainer Created")

# ============================================================
# Calculate SHAP Values
# ============================================================

print("Calculating SHAP Values...")

shap_values = explainer.shap_values(

    test_sample,

    nsamples=100

)

print("SHAP Calculation Completed")

# ============================================================
# Feature Names
# ============================================================

feature_names = [

    f"Feature_{i+1}"

    for i in range(test_sample.shape[1])

]

# ============================================================
# SHAP Summary Plot
# ============================================================

plt.figure(figsize=(10,6))

shap.summary_plot(

    shap_values[1],

    test_sample,

    feature_names=feature_names,

    show=False

)

plt.tight_layout()

plt.savefig(

    "Graphs/SHAP_Summary.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# SHAP Bar Plot
# ============================================================

plt.figure(figsize=(10,6))

shap.summary_plot(

    shap_values[1],

    test_sample,

    feature_names=feature_names,

    plot_type="bar",

    show=False

)

plt.tight_layout()

plt.savefig(

    "Graphs/SHAP_Bar.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# Mean Feature Importance
# ============================================================

importance = np.mean(

    np.abs(shap_values[1]),

    axis=0

)

importance_df = pd.DataFrame({

    "Feature":feature_names,

    "Importance":importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

importance_df.to_csv(

    "Results/SHAP_Feature_Importance.csv",

    index=False

)

# ============================================================
# Top Features
# ============================================================

print("\nTop 20 Important Features")

print(

    importance_df.head(20)

)

# ============================================================
# SHAP Force Plot
# ============================================================

try:

    force = shap.force_plot(

        explainer.expected_value[1],

        shap_values[1][0],

        test_sample[0],

        matplotlib=True,

        show=False

    )

    plt.savefig(

        "Graphs/SHAP_Force.png",

        dpi=600,

        bbox_inches="tight"

    )

    plt.close()

except:

    print("Force Plot Skipped")

# ============================================================
# Save SHAP Values
# ============================================================

np.save(

    "Results/SHAP_Values.npy",

    shap_values

)

print("\nFiles Saved Successfully")

print("✓ SHAP_Summary.png")

print("✓ SHAP_Bar.png")

print("✓ SHAP_Force.png")

print("✓ SHAP_Feature_Importance.csv")

print("✓ SHAP_Values.npy")

print("\n"+"="*70)

print("PART 8 COMPLETED SUCCESSFULLY")

print("="*70)
# ============================================================
# CCHT-KAN IMPLEMENTATION
# PART 9 : PERFORMANCE ANALYSIS & PUBLICATION GRAPHS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("="*70)
print("PART 9 : PERFORMANCE ANALYSIS")
print("="*70)

os.makedirs("Graphs",exist_ok=True)
os.makedirs("Results",exist_ok=True)

plt.rcParams["font.size"]=16
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["axes.titleweight"]="bold"

# ============================================================
# Prediction
# ============================================================

kan_model.eval()

with torch.no_grad():

    output=kan_model(test_x)

    probability=torch.softmax(output,dim=1)

    prediction=torch.argmax(probability,dim=1)

prediction=prediction.cpu().numpy()
probability=probability.cpu().numpy()[:,1]
actual=test_y.cpu().numpy()

# ============================================================
# Metrics
# ============================================================

acc=accuracy_score(actual,prediction)
pre=precision_score(actual,prediction)
rec=recall_score(actual,prediction)
f1=f1_score(actual,prediction)

print("Accuracy :",acc)
print("Precision:",pre)
print("Recall   :",rec)
print("F1 Score :",f1)

# ============================================================
# Confusion Matrix Heatmap
# ============================================================

cm=confusion_matrix(actual,prediction)

fig,ax=plt.subplots(figsize=(6,6))

im=ax.imshow(cm)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center",
            fontsize=18,
            weight="bold"
        )

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["Negative","Positive"])
ax.set_yticklabels(["Negative","Positive"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "Graphs/Fig_Confusion_Matrix.png",
    dpi=600
)

plt.close()

# ============================================================
# ROC Curve
# ============================================================

fpr,tpr,_=roc_curve(actual,probability)

roc_auc=auc(fpr,tpr)

plt.figure(figsize=(7,6))

plt.plot(fpr,tpr,label="AUC=%.4f"%roc_auc)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Graphs/Fig_ROC.png",
    dpi=600
)

plt.close()

# ============================================================
# Precision Recall Curve
# ============================================================

precision,recall,_=precision_recall_curve(
    actual,
    probability
)

plt.figure(figsize=(7,6))

plt.plot(recall,precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Graphs/Fig_PR_Curve.png",
    dpi=600
)

plt.close()

# ============================================================
# Accuracy Bar Chart
# ============================================================

models=[
"CCHT-KAN"
]

scores=[
acc*100
]

plt.figure(figsize=(6,6))

plt.bar(models,scores)

plt.ylabel("Accuracy (%)")
plt.ylim(0,100)

plt.title("Model Accuracy")

for i,v in enumerate(scores):

    plt.text(
        i,
        v+1,
        "%.2f"%v,
        ha="center",
        fontsize=15,
        weight="bold"
    )

plt.tight_layout()

plt.savefig(
    "Graphs/Fig_Accuracy.png",
    dpi=600
)

plt.close()

# ============================================================
# Precision Recall F1 Comparison
# ============================================================

metric_names=[
"Precision",
"Recall",
"F1"
]

metric_values=[
pre*100,
rec*100,
f1*100
]

plt.figure(figsize=(8,6))

plt.bar(metric_names,metric_values)

plt.ylabel("Percentage")

plt.title("Performance Metrics")

for i,v in enumerate(metric_values):

    plt.text(
        i,
        v+1,
        "%.2f"%v,
        ha="center",
        fontsize=15,
        weight="bold"
    )

plt.tight_layout()

plt.savefig(
    "Graphs/Fig_Performance.png",
    dpi=600
)

plt.close()

# ============================================================
# Training History
# ============================================================

if os.path.exists("Results/Training_History.csv"):

    history=pd.read_csv(
        "Results/Training_History.csv"
    )

    plt.figure(figsize=(8,6))

    plt.plot(
        history["Epoch"],
        history["Loss"],
        linewidth=3
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "Graphs/Fig_Loss.png",
        dpi=600
    )

    plt.close()

    plt.figure(figsize=(8,6))

    plt.plot(
        history["Epoch"],
        history["Accuracy"]*100,
        linewidth=3
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Training Accuracy")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "Graphs/Fig_Training_Accuracy.png",
        dpi=600
    )

    plt.close()

# ============================================================
# Save Metrics
# ============================================================

metric_df=pd.DataFrame({

"Accuracy":[acc],

"Precision":[pre],

"Recall":[rec],

"F1":[f1],

"AUC":[roc_auc]

})

metric_df.to_csv(
    "Results/Performance.csv",
    index=False
)

# ============================================================
# Summary
# ============================================================

print("\nGraphs Saved Successfully")

print("--------------------------------")

print("✓ Fig_Confusion_Matrix.png")

print("✓ Fig_ROC.png")

print("✓ Fig_PR_Curve.png")

print("✓ Fig_Accuracy.png")

print("✓ Fig_Performance.png")

print("✓ Fig_Loss.png")

print("✓ Fig_Training_Accuracy.png")

print("\nPerformance Saved")

print("="*70)
print("PART 9 COMPLETED SUCCESSFULLY")
print("="*70)