
# (BCS) Update 3:

![](bcs_content/title.jpg)

SORT:

1) Updated: has brought some logic in DB connection process - now it returns message about successfull connection
only in case, where DB usage is required by user.

2) Using an "euclidian distance" formula it is also able to classify moving and static objects.


DeepSort:

1) Added two approaches for object tracking using DeepSort: first one worhs with single YOLO model, which detects and classifies objects in the same time. Second approach uses two separate models - YOLO-v11-l for detection and my custom model Oko for classification. Custom one is a CNN model, which combines elements of ResNet and Squeeze-excitation block for better features extraction. You may chech additional information about custom model training in classification_model_training.ipynb file. 
