# AI-Based Human Activity Recognition Using Deep Learning 

This project implements an **AI-Based Human Activity Recognition System** using **Deep Learning (CNN + Transfer Learning)** to classify different physical activities from videos.
The model is trained and tested on the **UCF101 Dataset**, which contains short video clips of human actions such as walking, running, jumping, and more.

---

## 📘 Overview

The system uses a **Convolutional Neural Network (CNN)** and transfer learning approach to identify human activities from video data.
The model is trained and evaluated entirely on **Google Colab** for simplicity and high performance.

**Workflow:**

1. Load and preprocess video frames from the UCF101 dataset.
2. Extract key frames and resize them for CNN input.
3. Train the model using TensorFlow/Keras (or custom CNN).
4. Save the trained model (`.h5`) file.
5. Test and predict new videos using the saved model.

---

## ⚙️ Features

* Automatic frame extraction from video.
* Deep Learning model training using CNN architecture.
* Training and validation on UCF101 dataset.
* Real-time activity prediction from test videos.
* Colab-only execution (no external setup required).

---

## 🧠 Dataset

**UCF101 Dataset**

* Contains 13,320 videos across 101 human activity categories.
* Examples: *Walking, Running, Jumping, Horse Riding, Swimming, Playing Guitar,* etc.
* Dataset used for model training and validation.
* Source: [UCF101 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

---

## 🚀 How to Run (Google Colab)

1. Open the provided **Colab notebook** (`UCFFinal.ipynb`) in Google Colab.

2. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Upload your **trained model (.h5)** file in your Drive or Colab environment.

4. Run the notebook cells step by step for:

   * Data preprocessing
   * Model training
   * Model evaluation
   * Testing & Prediction

5. To test your own video:

   ```python
   !python predict.py --video test_video.mp4
   ```

   or use the prediction cell in Colab.

---

## 📊 Model Summary

|   Metric  | Training Accuracy | Validation Accuracy |
| :-------: | :---------------: | :-----------------: |
| CNN Model |      **100%**     |  **98% (approx.)**  |

* Framework: TensorFlow / Keras
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Epochs: 50
* Batch Size: 16

---

## 📈 Results

* Model achieved **perfect 100% training accuracy** and excellent validation performance.
* Accurately recognizes human activities from unseen video clips.
* Visual predictions are displayed directly in Colab output cells.

---

## 🔮 Future Enhancements

* Real-time video recognition using webcam input.
* Deployment on web dashboard using Flask or Streamlit.
* Improved accuracy with hybrid CNN + LSTM architecture.

---

## 👩‍💻 Author

**Farwa Majid Toor**
AI Internship Project 2025
Trained and tested on Google Colab.
Email:"farwamajid2004@gmail.com"

---

## 🪪 License

This project is developed for **educational and research purposes only.**
You may modify and use it with proper credit to the original author.
"
