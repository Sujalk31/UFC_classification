# UFC_classification
This project implements a machine learning model capable of  classifying images of prominent UFC fighters: John Jones, Khabib Nurmagomedov, Conor McGregor, Brock Lesnar, and Tony Ferguson. 
This project focuses on UFC fighter image classification using FaceNet for feature extraction and an SVM classifier for predicting fighter identities. The images are preprocessed and converted into 512-dimensional embeddings using FaceNet (InceptionResNetV1), which are then used to train an SVM model. The model is evaluated based on accuracy, and a confusion matrix is used to assess its performance across different classes. Training and testing accuracies are visualized to identify potential overfitting or underfitting issues. The trained model and label encoder are saved for future use, allowing predictions on new images. The project provides visualizations such as the confusion matrix and training vs testing accuracy graphs to evaluate the model’s performance. Future improvements could include data augmentation, fine-tuning FaceNet, and implementing additional evaluation metrics.







