

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('yahoo_ad_clicks.csv', header = None)
width = df.shape[1]
height =df.shape[0]
width
df_click = df.copy()
df_click = df_click.values



prevCumLoss = 0
cumReward = 0
regret1 = []

ourReward = []
Opt_Reward = []
prob = []
k = 50
s = np.zeros(k)
f = np.zeros(k)
reward= np.zeros(df.shape[1])
dist = np.array(k)
clicks = np.zeros(50)
loss = np.zeros(50)
lossCap = np.zeros(50)
lossCapPrev = np.zeros(50)
prevLoss = loss.copy()
click_counter = 0
for i in range(0,width):
    dist = np.random.beta(s+1,f+1)
    index = np.argmax(dist)
    reward[i] = df_click[index, i]
    if reward[i] == 1:
        s[index] = s[index] + 1
        click_counter = click_counter + 1
    else:
        f[index] = f[index] + 1
    
    cumReward = cumReward + (1-reward[i])
    ourReward.append(cumReward)
    temp = 1-df_click[:,i]
    loss = loss + temp
    optReward = np.min(loss)
    Opt_Reward.append(optReward)
    regret1.append(cumReward - optReward )
    #print(dist)
plt.title("Partial Feedback - Number of columns vs Loss curves")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.plot(regret1,'r',label="Regret") 
plt.plot(ourReward, 'b',label="Algorithm Loss")
plt.plot(Opt_Reward, 'g',label="Optimal Loss")
plt.legend()
plt.show()

plt.xlabel("Number of iterations")
plt.ylabel("Regret")
plt.plot(regret1,'r',label="Regret") 
plt.show()

print("Regret = ",regret1[-1])
