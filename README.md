# COVID-19 Detection from Chest X-Ray Images

This project focuses on detecting COVID-19, Pneumonia, and Normal cases from chest X-ray images using Convolutional Neural Networks (CNNs). It uses a custom dataset and deep learning models implemented with TensorFlow and Keras.

## 📂 Dataset

- The dataset contains X-ray images organized in folders: `train`, `val`, and `test`, with class subfolders:
  - `COVID19`
  - `NORMAL`
  - `PNEUMONIA`
## 🛠️ Tools & Libraries Used

- Python, NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib (for visualization)
- Google Colab (for training)

## 🧠 Models

Implemented and compared four custom CNN architectures:

- Model 1: 2 Conv layers with 32 & 64 filters
- Model 2: 2 Conv layers with 64 & 128 filters
- Model 3: 2 Conv layers with 128 & 256 filters
- Model 4: 2 Conv layers with 256 & 512 filters

All models use `ReLU`, `MaxPooling`, and `Softmax` for multi-class classification.

## 🧪 Training Strategy

- **Image Augmentation** with `ImageDataGenerator` (rotation, zoom, shift, flip)
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
- Trained for 4 epochs on each model using training and validation datasets

## ✅ Evaluation

- Evaluated models using:
- Accuracy
- Confusion Matrix
- F1-Score
- Final testing done using the `test` folder and single image inference

## 🔍 Prediction Pipeline

- Load any chest X-ray image
- Preprocess using image resizing and normalization
- Predict class (COVID-19, NORMAL, or PNEUMONIA) using trained model

## 🏁 How to Run

1. Upload your dataset to Google Drive
2. Mount Google Drive in Colab
3. Adjust the dataset path
4. Run the training and prediction cells

## 📈 Results

- Achieved good performance with simple CNNs
- Model 3 and Model 4 performed best on validation and test data

## 📌 Future Improvements

- Add transfer learning using pre-trained models (e.g., VGG16, ResNet50)
- Expand dataset for better generalization
- Deploy as a web app for clinical testing
