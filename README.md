# Performance Evaluation of Lightweight CNN Models for Tomato Leaf Disease Diagnosis

Diseases of tomato leaves are a major threat to farm production, resulting in huge economic losses globally. To ensure effective disease control and protection of the crops, it is vital to detect and diagnose diseases at an early stage. In this study, we investigate the use of lightweight CNN models, namely - MobileNetV3Small, EfficientNetB0, and a proposed CNN model, for efficient identification of tomato diseases from leaf images. The use of the PlantVillage dataset and data augmentation methods to help improve model generalization is done through the work. All the models are compared on basis of performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix. Results demonstrate that the proposed CNN model has competitive accuracy with much less computational footprint necessary for such resource limited environment.

> [Guava Image Dataset](https://www.kaggle.com/datasets/asadullahgalib/guava-disease-dataset/data)

The dataset was downloaded then splitted into train, validation and test sets. Then the models were trained using the train dataset, and during the training process validation was done using the validation dataset. Finally evaluation was done using the test dataset.

![Methodology_Diagram](https://github.com/user-attachments/assets/85212db5-dabe-4ffe-8f86-1fdfc8f05720)


Transfer Learning Model Used:
1. MobileNet
2. ResNet101
3. VGG16
4. Hybrid - Random Forest & MobileNet

The accuracies attained by the models are:
|Models       |Accuracy Score |
|-------------|---------------|
|MobileNet    |98.95%         |
|RFC-MobileNet|92.15%         |
|VGG16        |98.43%         |
|ResNet101    |99.48%         |
