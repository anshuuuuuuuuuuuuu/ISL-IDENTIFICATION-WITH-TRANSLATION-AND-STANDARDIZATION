# ISL Identification with Translation and Standardization

Real-time Indian Sign Language (ISL) recognition and translation system that converts ISL gestures into coherent sentences, delivering outputs in both audio and video formats.

---

## 🚀 Project Overview

This project aims to bridge the communication gap between the hearing-impaired community and the rest of society

* Recognizing ISL gestures in real-time using computer vision techniques.
* Translating recognized gestures into grammatically correct sentences.
* Providing outputs in both audio (speech synthesis) and visual (text display) formats.

---

## 🧠 Features

* **Real-Time Gesture Recognition**: Utilizes Mediapipe and TensorFlow for efficient and accurate gesture detection.
* **Natural Language Translation**: Converts sequences of gestures into meaningful sentences using NLP techniques.
* **Multimodal Output**: Delivers translated sentences through both synthesized speech and on-screen text.
* **Customizable Vocabulary**: Easily add new gestures and corresponding translations to expand the system's capabilities.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit for an interactive web interface.
* **Backend**:

  * Mediapipe for hand and pose detection.
  * TensorFlow & Keras for model training and inference.
  * Scikit-learn for data preprocessing and evaluation.
  * OpenCV for video processing.
* **Audio Output**: pyttsx3 for text-to-speech conversion.

---

## 📁 Repository Structure

```

├── app.py                   # Streamlit application entry point
├── final_isl5.ipynb          # Jupyter notebook for model training and experimentation
├── action_recognition_model1.h5  # Pre-trained model weights
├── README.md                 # Project documentation
├── requirements.txt          # List of dependencies
└── ...                       # Additional scripts and resources
```



---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/anshuuuuuuuuuuuuu/ISL-IDENTIFICATION-WITH-TRANSLATION-AND-STANDARDIZATION.git
   cd ISL-IDENTIFICATION-WITH-TRANSLATION-AND-STANDARDIZATION
   ```



2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



4. **Run the application**:

   ```bash
   streamlit run app.py
   ```



---

## 🖥️ Prerequisites

* **Hardware**: A system with a dedicated NVIDIA GPU is recommended for optimal performance.
* **Software**:

  * Python 3.7 or higher
  * CUDA and cuDNN installed and configured (for GPU support)

---

## 🧪 Usage

1. Launch the Streamlit application.
2. Allow access to your webcam when prompted.
3. Perform ISL gestures in front of the camera.
4. The system will display the translated sentence and play the corresponding audio.

---

## 🧩 Customization

* **Adding New Gestures**:

  * Collect video samples of the new gesture.
  * Label and preprocess the data accordingly.
  * Retrain the model using `final_isl5.ipynb` or update the existing model.

* **Modifying Translations**:

  * Update the mapping dictionary in the translation module to reflect new or altered translations.

---

## 📚 Resources

* **Mediapipe**: [https://mediapipe.dev/](https://mediapipe.dev/)
* **TensorFlow**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* **Streamlit**: [https://streamlit.io/](https://streamlit.io/)
* **pyttsx3**: [https://pyttsx3.readthedocs.io/](https://pyttsx3.readthedocs.io/)

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Empowering communication through technology.*

---

