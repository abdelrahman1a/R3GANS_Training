# R3GAN Implementation on CIFAR-10

Check out the live **Fake vs. Real Image Classification** app built with Streamlit:  
👉 [https://r3gans-fake-real-image-classification.streamlit.app/](https://r3gans-fake-real-image-classification.streamlit.app/)

---
This project is an implementation of **R3GAN** ("Re-GAN") based on the paper:

**"The GAN is dead; long live the GAN! A Modern Baseline GAN"**  
*by Yiwen Huang, Aaron Gokaslan, Volodymyr Kuleshov, and James Tompkin*  
[Read the paper](https://arxiv.org/abs/2501.05441) | [Official R3GAN Repo](https://www.github.com/brownvc/R3GAN)

## 🎯 Objective

The goal of this project is to explore and better understand modern GAN training techniques using the R3GAN architecture and loss. This is **not optimized for high accuracy** but rather for **experimental learning and model comprehension**.  

Implemented by **Abdelrahman Ahmed**, this version uses TensorFlow and is trained on the **CIFAR-10** dataset for **50 epochs**.

## 📌 Key Features

- Uses **R3GAN**: a principled and regularized relativistic GAN approach.
- Built on the foundation of modern GAN theory without relying on outdated tricks.
- Trained on **CIFAR-10** (60,000 32x32 color images in 10 classes).
- Provides an educational framework to understand the stability-diversity tradeoff in GANs.

## 🧠 Research Highlights (from the paper)

- Proposes a **relativistic GAN loss** (RpGAN) regularized by gradient penalties (R1 + R2).
- Removes the need for ad-hoc tricks used in traditional GAN architectures like StyleGAN.
- Combines the best of **modern architectural practices** (ResNets, grouped convolutions) with a clean loss design.
- Surpasses StyleGAN2 on several benchmark datasets (CIFAR-10 included).

## 🧪 Results

- **Epochs**: 50  
- **Dataset**: CIFAR-10  
- **Observations**: Accuracy is relatively low due to limited training and model scope, but the primary focus is **experimenting with GAN architecture and training stability**.

## 🚀 How to Run

1. Clone the repository
2. Install dependencies  
   ```bash
   pip install tensorflow numpy matplotlib

🤝 **Contributions**  
This project was developed by **Abdelrahman Ahmed** as part of a personal learning initiative.  
Contributions, suggestions, or improvements are welcome!  
Feel free to **fork the repo** and **submit a pull request**.
