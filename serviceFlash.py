from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
import pickle

model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)

# Загружаем параметры из файла
model.load_state_dict(torch.load('C:\\Work\\project\\II\\python\\model3.pt', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose(
    [transforms.Resize((64,64)),  #изменим размер изображений
     transforms.ToTensor(),   #переведем в формат который необходим нейронной сети - тензор
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) # проведем нормализацию изображения

app = Flask(__name__)

with open('C:\\Work\\project\\II\\python\\classes.pkl', 'rb') as f:
    class_names = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    image = Image.open(file.stream)  # Открываем изображение
    image = transform(image).unsqueeze(0)  # Применяем преобразования и добавляем размерность батча
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        class_name = class_names[class_id]  # Получаем название класса
    return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True)
