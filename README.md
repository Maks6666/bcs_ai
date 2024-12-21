
# (BCS) Update 1:

![](bcs_content/title.jpg)

1) Instead of custom "tracking" I decided to use two common track algorithms - SORT and DeepSort
to increase quality of detection. I'd like to use first method, because in combat situation fast speed
of SORT implementation may be more effective, while strict tracking, given by DeepSort remains not so valuable.
Anyway, basic implementation of both methods will be available on my page... 

2) To sort all given information, I consider using of PostgreSQL database. 

P.S additional code parts for DeepSort were taken from: 
- https://github.com/Koldim2001/SORT-DeepSORT-Tracker
- https://github.com/nwojke/deep_sort/tree/master

3) Improved detection model. Instead of all classes detected, new model versions
will be concentrated on detection of only three types of vehicles - tanks, APS's and IFV's. Trainung
process will be also implemented on a new, bigger dataset. 

Basic performance:

![]("bcs_content/battlefield.gif")

![]("bcs_content/tank.gif")