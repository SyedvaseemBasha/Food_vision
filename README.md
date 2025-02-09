# 🍔 Food Vision - Fine-tuned EfficientNetB7 for Food Classification

## 🚀 Project Overview

**Food Vision** is a deep learning-based food image classification project built using **EfficientNetB7**. It classifies food images into **101 categories** from the Food-101 dataset. By leveraging the power of transfer learning and fine-tuning, this model achieves high accuracy in real-world food recognition tasks.

🔗 **Live Demo**: [Food Vision Streamlit App](https://foodvision-mjyh6u7jbls2ectys2unuy.streamlit.app/)

## 📌 Dataset: Food-101

- **Food-101** is a large-scale dataset containing **101 food categories** with **1000 images per category**.
- **75,000 images** are used for training, and **25,000 for testing**.
- Diverse real-world images make it an ideal dataset for deep learning-based food classification.

## 🏗 Model Architecture: EfficientNetB7

**EfficientNetB7** is a state-of-the-art deep learning architecture that offers superior accuracy while optimizing computational efficiency. It is ideal for food image classification due to its ability to extract fine-grained details from images.

### ✅ Why EfficientNetB7?

- **High Accuracy**: One of the top-performing models on ImageNet.
- **Advanced Scaling**: Uses compound scaling to optimize width, depth, and resolution.
- **Robust Feature Extraction**: Ideal for complex food classification.
- **Fine-tuning Capabilities**: Allows selective layer unfreezing for better generalization.

### 🔥 Training & Optimization

The training process involves:

- **Transfer Learning**: Using EfficientNetB7 pre-trained on ImageNet.
- **Fine-Tuning**: Unfreezing select layers for improved feature extraction.
- **Data Augmentation**: Image transformations (flipping, rotation, scaling) to improve robustness.
- **Mixed Precision Training**: Uses `float16` where possible for faster computation.
- **Optimizer**: Adam optimizer with learning rate scheduling for stability.

## 🚀 Deployment

### Streamlit - Web App for Classification

- The model is deployed via **Streamlit**, providing a simple UI for food classification.
- Accessible from any device: [Food Vision Web App](https://foodvision-mjyh6u7jbls2ectys2unuy.streamlit.app/)

## 📜 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SyedvaseemBasha/Food_vision.git
cd Food_vision
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

## 📈 Results & Future Work

- Achieved **~85% top-1 accuracy** on the Food-101 dataset.
- Future improvements:
  - Optimize the model for **edge devices & mobile deployment**.
  - Explore **Self-Supervised Learning** for better performance with fewer labeled samples.
  
## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue if you find a bug or have suggestions.

## 📜 License

This project is open-source and available under the **MIT License**.

