
# Tactic battle control system based on AI technology (YOLOv9/custom models).
![](bcs_images/title.jpg)

## Idea

Combat operations are typically divided into three main levels: tactical, operational and strategic. Success in each of them guarantees victory in the war, but in all wars, there is one common problem that inevitably befalls any of the parties to the conflict after prolonged hostilities. A shortage of personnel in the first place can have a very serious impact on the situation on the front line - at first, the shortage of soldiers will affect the tactical level of combat operations, which will almost inevitably lead to a crisis at higher levels. This is especially true for the side that is on the defensive.

This problem of personnel shortages can be partially solved with the help of autonomous combat modules - robotic machine guns, anti-tank systems, anti-aircraft machine guns, etc. however, they also need to be monitored, which also requires a considerable number of people. However, this problem can be addressed using simple AI algorithms: multi-level convolutional networks for detecting objects like YOLO for tracking the situation on the battlefield and simple ANN for analyzing the information received and making certain decisions that will be transferred to automatic combat modules.

![](bcs_images/idea.jpg)

The idea behind the solution is that a high-resolution tactical camera is placed on the battlefield with access to a processor to run the program. The camera tracks enemy activity: using the YOLOv9c model, which is trained on a custom dataset (https://www.kaggle.com/datasets/nzigulic/military-equipment/code), as well as the YOLOv9c model with default settings, the program tracks five groups of potential enemies: attacking armored vehicles (tanks, APCs, IFV), special equipment (engineering vehicles, air defense systems), human infantry, artillery (self-propelled guns and towed artillery) and aviation (combat aircraft, transport aircraft, combat helicopters, transport helicopters).

Next, the defense sector commander enters in a special field the amount of ammunition available for use for immediate defense - (ATGMs, cluster and unitary munitions, FPV kamikaze drones, anti-aircraft missiles and MANPADS) - this part can also be optimized using sensors that will directly read the amount of ammunition.

The camera is placed in such a way that the area where enemy activity is expected is visible - one of the program options offers integration of the camera with a Tello model drone for greater convenience, but a regular high-resolution camera will also work. The camera must be connected to a laptop computer, which will perform the main calculations.

![machine_gun.jpg](bcs_images/machine_gun.jpg)
*Automatic combat module "Shablya" of the Armed Forces of Ukraine. This program could group, coordinate and systematize the central control of combat systems of this category.*

The program involves a combination with autonomous combat modules (automatic machine guns, ATGM positions and FPV drones, etc., which will operate autonomously or with minimal human intervention). This means that using the camera and YOLO models, the program will calculate the number of objects of one type or another in the video, combine this with the number of available weapons, feed the obtained data into one of the custom analyzer models written in PyTorch, which will return a specific solution, which will have to activate one or another combat module, direct it to the desired object (at the moment, priority is given to the object with the lowest ID, that is, the one that was detected first, however it could be changed by integration of another logic) 
and will force him to produce a target according to the coordinates of the object. The shot will automatically reduce the amount of ammunition that was used - for now, the amount is reduced automatically, two seconds after detection and targeting of the desired object, however in the future, this could be integrated directly with automatic combat modules - for now, the program is more the “brain” of such a small tactical battle control system, which is capable of making simple decisions based on visually obtained data.

![fire.jpg](bcs_images/fire.jpg)
*This function implements the logic: after selecting a weapon of a suitable category, 1 will be subtracted from the number of the selected type of weapon every 2 seconds, simulating the waste of ammunition per shot. Unfortunately, I have not yet had time to study in detail the issue of integrating the “brain” with autonomous combat systems, but I assume that this should be implemented in this function.*


## Programm
![amunition_amount.png](bcs_images/amunition_amount.png)
*Here user may fit an amount of available weapons.*

![choose_file.png](bcs_images/choose_file.png)
*Here user may choose a video file or camera it wants to use.*

When launching the program, the user has the opportunity to choose a method for reading information: a camera to which the device has access or a video file, which must (!) be located in the same directory as the project, in a folder called “videos”. The user then manually enters the number of available weapons - the part that could later be automated using sensors and which feeds the model with data on the number of weapons.

## Programm 
The program consists of three neural network models. The first is a YOLOv8 model, trained on a custom dataset (with more than 2000 images) to find artillery units (regardless of whether it is a decoy or not) from any video. The second is the YOLOv8l model with default settings, which is focused on searching for cars, people and other transport equipment around a detected object.

This way programm creates area around detected howitzers, where could be found such objects as humans (obviousle soldiers) and vehicles:

Then, using OpenCV algorithms, the program opens the selected file and passes each frame through two YOLO models - the first of them, trained on custom data, is necessary for detecting military equipment. The second is a model with default settings, which is adapted for detecting people - enemy infantry.

![decision_model.png](bcs_images/decision_model.png)
*The logic for encoding the number of available weapons and feeding data about them and the number of detected equipment into the model is displayed here.*


    
Programm also contains my own analyzer ANN-model, made with Tensorflow and Keras, which will analyze detected objects and make a decision is there a real howitzer or decoy. Obviously, in some cases it will mark targets as undecided objects and ask operators for help, but in general, it may make frone operators job much more easy.

This way you should upload analyzer model with pickle:

```bash
    import pickle
    file_path = "path/to/pickle/file"

    with open(file_path, 'rb') as file:
        analyzer_model = pickle.load(file)
 ```



## Installations

Before running programm, don't forget to install all necessary packages:
```bash
   pip install opencv-python
   pip install ultralytics
   pip install tensorflow
   pip install scikit-learn
   pip install numpy
   pip install pandas 
   pip install seaborn 
 ```

They are necessary for running all of three models.
## Example


This way programm returns detection of real targets:

![](ex2.png)
--------------------------------
This way programm returns detection of 
undecided targets:

![](ex3.png)
--------------------------------

This way programm returns detection of 
decoys:

![](ex4.png)

The possible functionality of project is to be combined with FPV-drones to implement CV technologies into military targets recognition. Of course it is also possible to make such programm recognize and classify different types of weapons: vehicles, MANPADS and etc. 

Thanks for attantion, hope you will find it useful:)

## Authors:
- Kucher Maks (maxim.kucher2005@gmail.com)