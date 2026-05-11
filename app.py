import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from transformers import Wav2Vec2Processor, Wav2Vec2Model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# emotions
emotions = ['ANG', 'CAL', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD', 'SUR']

# ---------------- MODEL ----------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("superb/wav2vec2-base-superb-er").to(device)

class SERModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.input_dropout = nn.Dropout(0.2)
        self.bilstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(512)
        self.attention = nn.Sequential(
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.input_dropout(x)
        lstm_out,_ = self.bilstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc(context)

model = SERModel(len(emotions)).to(device)

# load .pth
checkpoint = torch.load("best_modelSR (8).pth", map_location=device)
wav2vec.load_state_dict(checkpoint["wav2vec"])
model.load_state_dict(checkpoint["model"])

wav2vec.eval()
model.eval()

# ---------------- FUNCTIONS ----------------
def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(device)

    with torch.no_grad():
        features = wav2vec(inputs).last_hidden_state
        logits = model(features)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = np.argmax(probs)
    return emotions[pred_id], probs


def plot_graph(probs):
    plt.figure()
    plt.bar(emotions, probs)
    plt.title("Emotion Probabilities")
    graph_path = os.path.join("static", "graph.png")
    plt.savefig(graph_path)
    plt.close()
    return graph_path


def recommend_music(emotion):
    mapping = {
        "ANG": "calming relaxing instrumental music",
        "SAD": "uplifting happy songs playlist",
        "FEA": "peaceful soothing  instrumental music",
        "DIS": "light feel good  songs",
        "HAP": "energetic arty songs playlist",
        "CAL": "soft relaxing  lofi music",
        "NEU": "pleasant light songs",
        "SUR": "fun upbeat songs"
    }
    return mapping[emotion]

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["audio"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        emotion, probs = predict_emotion(file_path)
        graph = plot_graph(probs)
        music = recommend_music(emotion)

        url = f"https://www.youtube.com/results?search_query={music.replace(' ','+')}"

        return render_template(
            "index.html",
            emotion=emotion,
            graph=graph,
            music=music,
            url=url
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)