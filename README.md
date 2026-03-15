## Introduction

Bone age assessment is an important tool for evaluating growth and development in pediatric patients. It is commonly performed by analyzing hand and wrist X-ray images, but traditional methods rely heavily on expert interpretation and can be time-consuming.

This project aims to develop a deep learning–based system that automatically estimates bone age from hand X-ray images. The goal is to create a fast, consistent, and reliable approach that can support clinical evaluation and reduce the dependence on manual assessment.
## Dataset

The dataset used in this project consists of **hand X-ray images** for bone age estimation.
For each individual, two anatomical regions are considered:

* **Articular surface**
* **Epiphysis**

During preprocessing, these regions are extracted from the original X-ray images in order to reduce background effects and allow the model to focus only on the relevant bone structures.

Each sample in the dataset includes:

* **Bone age label** (in months)
* **Gender information** (male/female) as an additional feature

To evaluate the generalization capability of the model, the dataset is divided into three subsets:

* **Training set**
* **Validation set**
* **Test set**

Separate image directories and label files are used for each subset to prevent data leakage during the training process.

### Example Regions

**Articular Surface**

![Articular Surface](images/articular_surface.png)

**Epiphysis**

![Epiphysis](images/epiphysis.png)
