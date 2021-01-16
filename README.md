# Face Verification Using One-Shot Learning:
Face verification is the task of comparing a candidate face to another and verifying whether it is a match. It is different from face recognition. This model can be used for Facial Recognition also my making another minor change in the code. So this model uses the technique of One-Shot Learning, which means, to learn information about object categories from one, or only a few, training samples/images. So this model needs only one image of a person in order to learn information and can be used for face verification and face recognition tasks. The base model is an inception network. Various detail about the models and their plots can be found in the notebook.

# Libraries:
* Tensorflow
* Keras
* NumPy
* Opencv
* Matplotlib

# Dataset Link: 
https://www.kaggle.com/jayitabhattacharyya/face-match

# Command Line:
```python face_verification.py -r path_to_reference_image -i path_to_test_image```


 

# References: 
* Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
* Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
