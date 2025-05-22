
# 🐶🐱 Cat vs Dog Image Classifier with VGG16

This project is a deep learning-based image classifier using a pre-trained **VGG16** model from PyTorch's `torchvision.models`. The model is fine-tuned to classify images as **cat** or **dog**.

## 📦 Dataset

- The dataset is expected in the format:
  ```
  animals/
  └── animals/
      ├── cat/
      │   └── *.jpg
      └── dog/
          └── *.jpg
  ```
- The dataset is loaded from Google Drive and extracted in Google Colab.

## 🚀 Features

- Uses **transfer learning** with VGG16
- Freezes convolutional layers and trains the classifier
- Visualizes sample images
- Trains for a few epochs and evaluates performance
- Predicts custom images (e.g., `/content/cat2.jpg`)
- Saves and reloads the model with `.pth` format

## 🛠️ Dependencies

See [`requirements.txt`](./requirements.txt) for exact versions.

## 📂 File Structure

```
project/
│
├── cat_vs_dog_classifier.ipynb     # Main Colab notebook/script
├── vgg16catdog.pth                 # Trained model weights
├── README.md
└── requirements.txt
```

## 🧪 Training and Evaluation

- Dataset split: 80% training, 20% testing
- Evaluation metric: Accuracy
- Training for 5 epochs (can be modified)

## 🖼️ Inference

Use the trained model to predict new images:
```python
img = Image.open("your_image.jpg")
output = model(transform(img).unsqueeze(0))
prediction = class_names[output.argmax()]
```

## 🔧 Future Improvements

- Add early stopping
- Use more data augmentations
- Deploy as a web app using Gradio or Flask


