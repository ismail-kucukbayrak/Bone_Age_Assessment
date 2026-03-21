
# 🦴 Bone Age Assessment

![Python](https://img.shields.io/badge/Python-DeepLearning-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![CNN](https://img.shields.io/badge/Model-CNN-green)
![Computer Vision](https://img.shields.io/badge/Field-ComputerVision-purple)
![Jupyter](https://img.shields.io/badge/Tool-JupyterNotebook-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Introduction

Bone age assessment is an important tool for evaluating growth and development in pediatric patients. It is commonly performed by analyzing hand and wrist X-ray images, but traditional methods rely heavily on expert interpretation and can be time-consuming.

This project aims to develop a deep learning–based system that automatically estimates bone age from hand X-ray images. The goal is to create a fast, consistent, and reliable approach that can support clinical evaluation and reduce the dependence on manual assessment.

## Dataset
The dataset used in this project consists of **hand X-ray images** for bone age estimation. 
For each individual, two anatomical regions are extracted from the original images: the **articular surface** and the **epiphysis**, allowing the model to focus on the most relevant bone structures while reducing background effects.

Each sample includes the **bone age label** (in months) and **gender information** (male/female) as an additional feature.  
To evaluate the model's generalization capability, the dataset is divided into **training**, **validation**, and **test** sets.  
Separate image directories and label files are used for each subset to prevent data leakage during the training process.

### Example Regions
<p>
<b>Articular Surface</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Epiphysis</b>
</p>

<p>
<img src="images/articular_surface.png" width="35%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="images/epiphysis.png" width="35%">
</p>


## Method
DenseNet121, ResNet50, EfficientNetB3, and EfficientNetB0 models were evaluated, and EfficientNetB0, which showed the best performance, was selected. The model employs a dual-branch CNN architecture that processes articular surface and epiphysis images separately. Before being fed into the model, the images were resized to 320×320 pixels and passed through preprocessing steps.

In both branches, networks pretrained on ImageNet were used and a transfer learning approach was applied, where most layers were frozen and only the final layers were fine-tuned during training. The extracted image features were fused with gender information and passed to fully connected layers. The model output consists of a single neuron that predicts the normalized bone age. During training, Mean Absolute Error (MAE) was used as the loss function, and early stopping and learning rate reduction were applied to reduce the risk of overfitting.

<p>
<img src="images/architecture.png" width="50%">
</p>

## Performance Analysis
The performance of the proposed deep learning model was evaluated on the test set using **Mean Absolute Error (MAE)**, **Root Mean Square Error (RMSE)**, and **R²** metrics. The model achieved an MAE of approximately **10.42 months** on the test set. Considering that bone age estimation is a challenging regression problem with high variability, this result indicates a realistic and acceptable level of performance.

<p>
<img src="images/graph.png" width="100%">
</p>

The table below presents the results of several sample records.
<p>
<img src="images/table.png" width="50%">
</p>

## Conclusion and Discussion
In this study, a deep learning–based approach was developed to automatically estimate bone age from hand X-ray images. The proposed method uses a dual-branch CNN architecture that processes the **articular surface** and **epiphysis** regions separately. Transfer learning was applied, and **gender information** was included as an additional input for the regression-based bone age prediction.

Experimental results show that the model achieved an **MAE of 10.42 months** on the test set, which is consistent with results reported in similar studies. The model successfully captures the general age distribution, although prediction errors increase at very low and very high age ranges, likely due to dataset imbalance and the nonlinear nature of bone development.

## How to Run

Follow the steps below to run the application locally.

### 1. Clone the Repository

```bash
git clone https://github.com/ismail-kucukbayrak/Bone_Age_Assessment.git
cd Bone_Age_Assessment/Application
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

---

## Application Usage

1. The graphical interface will open.
2. Enter a valid **sample ID** in the input field.
3. Click **Query**.
4. The system will display:
   - Patient **sex**
   - **Chronological age**
   - **Predicted bone age**
   - The corresponding **hand radiograph image**.

An example of the application interface is shown in the figure below.

<p>
<img src="images/query.png" width="100%">
</p>
