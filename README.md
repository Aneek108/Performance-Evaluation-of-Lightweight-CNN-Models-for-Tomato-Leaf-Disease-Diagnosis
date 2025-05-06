# Performance Evaluation of Lightweight CNN Models for Tomato Leaf Disease Diagnosis

Diseases of tomato leaves are a major threat to farm production, resulting in huge economic losses globally. To ensure effective disease control and protection of the crops, it is vital to detect and diagnose diseases at an early stage. In this study, we investigate the use of lightweight CNN models, namely - MobileNetV3Small, EfficientNetB0, and a proposed CNN model, for efficient identification of tomato diseases from leaf images. The use of the PlantVillage dataset and data augmentation methods to help improve model generalization is done through the work. All the models are compared on basis of performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix. Results demonstrate that the proposed CNN model has competitive accuracy with much less computational footprint necessary for such resource limited environment.

> ### [PlantVillage Image Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)
> Only the Tomato images are used from the PlantVillage dataset.

![Methodology](https://github.com/user-attachments/assets/de7e2e3b-a723-48fc-99f1-473a034bce92)
![2813d2f7-5922-4be3-9865-48ef10157a8f](https://github.com/user-attachments/assets/7b7e2692-474c-489c-83fe-148474cbf29e)

Transfer Learning Model Used:
1. MobileNet-V3 Small
2. EfficientNet-B0
3. Custom CNN

![Custom CNN](https://github.com/user-attachments/assets/d98a4d48-7c7c-493d-8bc0-e52aa4d782e1)

The accuracies attained by the models are:
|Models                  |Accuracy Score |
|------------------------|---------------|
|MobileNet-V3 Small      |97.58%         |
|EfficientNet-B0         |98.51%         |
|Custom CNN              |98.62%         |
