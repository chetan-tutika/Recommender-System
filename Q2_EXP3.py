import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('yahoo_ad_clicks.csv', header = None)
width = df.shape[1]
height =df.shape[0]
width
df_click = df.copy()
df_click = df_click.values


neta_graph = []
prevCumLoss = 0
cumLoss = 0
clicks = 0
algLoss = 0
regret1 = []
ourLoss = []
Opt_Loss = []
prob = []
k = 50
dist = np.random.uniform(0,1,k)
#clicks = np.zeros(50)
loss = np.zeros(50)
lossCap = np.zeros(50)
lossCapPrev = np.zeros(50)
prevLoss = loss.copy()
for i in range(0,width):
    #neta = np.sqrt(np.log(k)/((i+50)*k))
    neta = 3.5/np.sqrt(i+1)
    neta_graph.append(neta)
    cumulative = np.cumsum(dist)
    cumulative_norm = cumulative/np.max(cumulative)
    randNum = np.random.uniform(0,1)
    #print(randNum)
    #print(cumulative)
    #print(cumulative_norm)
    index = np.argmax(cumulative_norm >= randNum)
    #print(index)
    #clicks = np.sum(df_click[:,:i], axis = 1)
#     if df_click[index, i] == 1:
#         clicks[index] += 1 
    lossOpt = 1.0 - df_click[:, i]
    lossCap[index] = 1.0 - df_click[index, i]
    clicks = clicks + df_click[index, i]
    CapLCap = lossCap[index]/dist[index]
    loss[index] = CapLCap + prevLoss[index]
#     lossClick = 1.0/(clicks[index]+1)
#     lossCurrent = lossClick/dist[index]
    #loss[index] = prevLoss[index] + lossCurrent
    #print(loss[index])
    cumLoss = cumLoss + lossOpt
    algLoss = algLoss + lossCap[index]
    lossCapPrev = lossCapPrev + lossCap
    ourLoss.append(algLoss)
    optLoss = np.min(cumLoss)
    Opt_Loss.append(optLoss)
    regret1.append(algLoss - optLoss)
    prevLoss = loss
    numExp = np.exp(-neta*loss)
    denExp = np.sum(np.exp(-neta*loss))
    #print(numExp)
    dist = np.divide(numExp, denExp)
    prob.append(dist)
    #print(dist)
    
plt.plot(regret1,'r', label = 'regret')
plt.plot(Opt_Loss, 'b', label = 'Optimal Loss')
plt.plot(ourLoss, 'g', label = 'Algorithm Loss')
plt.xlabel('time')
plt.ylabel('loss')
plt.legend()
plt.show()
print('regret = ', regret1[-1])