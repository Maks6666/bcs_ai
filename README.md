
# (BCS) Update 2:

![](bcs_images/title.jpg)

1) I decided to apply a new approach of detection. In previous versions, I used YOLO-model trained on custom data to
detect 10 or 3 classes of military vehicles. Second version works much better and return high accuracy score, but to increase it,
i decided to make YOLO-model find only one class: "vehicle", which includes photos of tanks, APC's and IFV's. I'm planning to 
combine it with custom classification model to split tasks among two different models. In classificator model I will combine
different accuracy improving methods such as transfer learning, knowledge distillation and specific architecture approach - 
combination of skip-connection, dense-net and usual convolutional network methods.

I will mention here all of my progress. To be continued... 
