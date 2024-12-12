
# (BCS) Update 1:

![](bcs_images/title.jpg)

1) Instead of custom "tracking" I decided to use two common track algorithms - SORT and DeepSort
to increase quality of detection. I'd like to use first method, because in combat situation fast speed
of SORT implementation may be more effective, while strict tracking, given by DeepSort remains not so valuable.
Anyway, basic implementation of both methods will be available on my page... 

2) To sort all given information, I consider using of PostgreSQL database. 

P.S additional code parts for DeepSort were taken from: 
- https://github.com/Koldim2001/SORT-DeepSORT-Tracker
- https://github.com/nwojke/deep_sort/tree/master