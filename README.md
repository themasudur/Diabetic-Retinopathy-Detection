## Diabetic Retinopathy Detection using Color Fundus Photos

---

This repository contains the implementation of a project **Diabetic Retinopathy Detection using Color Fundus Photos**. It is the final project demonstration of CSCE 566 Data Mining course. 


### **Dataset**

The project used ODIR color fundus image dataset, stored in `.npz` format, which is preprocessed for training, validation, and test.


### **Models**

- Implemented a pipeline with multiple models:
  - VGG19
  - ResNet50
  - EfficientNetB0
- Pre-trained with ImageNet weights.
- Fine-tuned using various optimizers, loss functions, and hyperparameters.

DenseNet201 was initially included but removed due to poor performance.


### **Results**

- **VGG19**: Achieved overall AUC of 0.98; 0.97 for males and 0.98 for females.
- **ResNet50**: Achieved overall AUC of 0.98, and 0.98 for both groups.
- **EfficientNetB0**: Achieved the highest overall AUC of 0.99 and 0.99 for both groups.

### **Run**

You can run this project either cloning (not updated yet) or run the [`Detection Task.ipynb`](https://github.com/themasudur/Diabetic-Retinopathy-Detection/blob/main/Detection_Task.ipynb) file.

