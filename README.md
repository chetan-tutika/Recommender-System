# Recommender-System

To design a recommender system to recommend ads based on the past experiences.<br />
Data set -- yahoo_ad_clicks.csv
<br />

We use partial feedback to desing a multi armed bandit recommendor system. We then compared the performance of both EXP3 and Thompson sampling on the given dataset and analyzed the performance metrics of both the algorithms.

## Plotting the Distribution for Different Neta
### Neta:
Neta plays a very important role in deciding the amount of exploitation and exploration. High
values of neta will favor exploration over exploitation while low values of neta will favour
exploration over exploitation.<br />
![partialtsqr2](https://user-images.githubusercontent.com/41950483/51356465-9fc6ef80-1a88-11e9-9835-b1218f9a1dc2.png)<br />
As the value of neta increases the updates to the distribution are larger and hence, high values
of neta will result in larger update steps which would result in skewed distribution(narrow curve)<br />
This can be observed when we take neta to be 1/sqrt(t+2). The Since the value of neta
increases quadratically over time the distribution is updated with large step size. This update
skews the distribution to particular set of ads<br />
![partialt2](https://user-images.githubusercontent.com/41950483/51356514-c5ec8f80-1a88-11e9-953c-5d38f2ac6f8c.png)<br />
n case of neta = 1/(t+2), the value of neta increases linearly over time. This means that the
updates to the distribution are not as large as for neta = 1/sqrt(t+2) which allows more room for
exploration. Hence, the distribution is not skewed to a particular ad.<br />
![dist](https://user-images.githubusercontent.com/41950483/51356591-12d06600-1a89-11e9-986d-a345114e2515.png)
To have a balance between exploration and exploitation, an optimal regret is chosen with value
= 3.5/sqr(t+1)<br />
From the graph it can be observed that the distribution is not skewed to a particular arm while
not being too widely distributed. This ensures that the exploitation and exploration are taking
place almost equally<br />

## Regret and Loss
Regret is calculated based on the difference between the optimal loss and the loss of our
algorithm at each time step<br />
Loss = 1 - reward<br />
Our loss = total loss of the chosen arm upto time step ‘t’<br />
Optimal loss = cumulative loss of the best arm at each time step ‘t’<br />

### EXP3
![regret118](https://user-images.githubusercontent.com/41950483/51356979-b9693680-1a8a-11e9-8bf9-ef0d9f91e752.png)
![loss](https://user-images.githubusercontent.com/41950483/51356980-ba9a6380-1a8a-11e9-8e01-e143062df6ad.png)<br />
The calculated loss of our algorithm is following the optimal loss.
During the end, the distribution converges to some form and the algorithm exploits only from
those few values which decreases the scope of exploration. So while the distribution matches
with the reward at each round we get very low loss which indicates the sharp dip in the regret.
The rise after that can be explained if the distribution no longer matches the rewards acquired.
Since the distribution is converged to the previous rewards and the exploration reduces, if the
rewards no longer correspond to the distribution the loss will gradually increases.

### Thompson
![unnamed](https://user-images.githubusercontent.com/41950483/51357231-e4a05580-1a8b-11e9-8fa2-0b872c6eb249.png)
![unnamed 1](https://user-images.githubusercontent.com/41950483/51357233-e79b4600-1a8b-11e9-9639-b468b00b7e78.png)<br />

In case of Thompson, we see that the Thompson sampling is giving regret of 2093 with mean regret of
1092 with the number of clicks where the reward is 1 are 9006.<br />
Comparing both, we get EXP3 to be better for partial feedback<br />

## Why EXP3 and not Thompson?
Thompson sampling creates different beta distribution for each arm(k = 50). In case of partial
feedback, constructing distribution for each arm based on the loss of a single arm might give not
give good results. This is observed in our case, thompson sampling gives higher regret than
EXP3 algorithm. By not considering the full feedback and updating the loss based a single arm
at each time step, the distributions of some arms might be much more prominent to be selected
than others. This would reduce the amount of exploration, and would increase the exploitation
from the arms with updated distributions.<br />
EXP3 is a non-stochastic algorithm, The distribution built is common for all the arms and we
sample from it. By updating the distribution for all the arms collectively w.r.t the loss obtained by
a single arm. In case of partial feedback this works better. By updating the whole distribution,
we can say that the loss of the all the arms will be effected and the update of loss will not be
skewed towards a single arm.


