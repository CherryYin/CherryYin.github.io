---
title: HRL-RE——端到端训练实体和关系抽取
layout: post
categories: 关系抽取 实体识别 知识抽取 自然语言处理
tags: 强化学习 关系抽取 实体识别 端到端训练
excerpt: HRL-RE——端到端训练实体和关系抽取
---

  
     
	 本文是对《A Hierarchical Framework for Relation Extraction with Reinforcement Learning》这项工作的理解和分析。很久没有深入分析一篇论文了，这篇算是笔者今年第一篇深入分析的工作，为什么要深入分析呢？大概是因为它可以一次搞定实体识别和关系分类，同时又采用了比较时髦的强化学习。
  论文作者友好的提供了pytorch下的代码。因此，笔者在看论文时，以论文分析为主线，对于重要的模块分析了对应的代码。

### **一. 概述**
  
  信息抽取：假如你需要获取某个领域的某些结构化的信息，你找来一份文本数据，从这份数据中，你分析出，可以提C类结构化数据（实体），并且你确定这C类的结构化数据两两组合产生的关系中，你更关注其中的R类，于是，你需要从这份数据中抽取属于这C类的所有实体，同时确定这些实体两两间关系是否是你关注的那R类。这个需求可以简述为文本的C类实体识别（2C+1或3C+1类序列标注）和R+1类关系分类。
  
  传统解决方案：针对这个需求，目前更广泛的方法是先通过文本的实体标注数据训练一个模型识别C类实体标注，然后将文本和实体位置等信息作为输入，训练另一个独立的模型用于关系R+1分类。这种方式有两大问题：一是实体识别不可能完全正确，而实体识别的误差会直接下一级的关系分类，这样一来，关系分类的准确率是实体识别准确率和关系分类模型本身准确率的乘积；二是关系类别对实体识别是有帮助的，但在这里却被忽略了。
  
  该项工作的方案：作者们认为，实体识别和关系分类是相辅相成的，将两者融合在一个模型中，让关系引导实体识别，实体监督关系分类，一起训练效果更优。因此，作者提出的方法：对同一个文本，在一个模型中，通过两级半级联的强化学习（RL）来实现整个信息抽取，高级别的RL抽取关系，低级别RL识别实体，在整个文本的两级工作完成后计算奖励和梯度，整个模型同时优化。
  
  具体方法：
  
  1. 通过一个按时刻工作的扫描器逐步扫描文本；	
  2. 每个时刻，扫描到当前文字后计算当前高级别RL的状态；
  3. 根据当前高级别RL的状态选择策略，由策略决定当前时刻行动，当然这里的行动是虚拟的，相当于一共有R种方向可走，或者完全不动这R+1种选择；
  4. 如果行动是完全不动，说明扫描到当前时刻还没有发现关系，那么继续下一时刻的扫描和计算；
  5. 如果当前行动为前R个方向之一，那么触发低级别RL，即实体识别，同样计算当前时刻低级别RL的状态、策略、行动，低级别计算完后，继续下一步扫描和计算，直到整个文本扫描结束。


  
  值得注意的是，整个模型类似于半马尔科夫链，高级别RL的当前次状态是由上一时刻状态、上一次行动（是上一时刻可能不动，所以这里是上一次）和此刻输入共同决定的，低级别RL此刻可能完全不被触发，也可能被触发后状态由高级别RL的行动、自己上一次的状态、此刻输入决定。
  
  整个结构图如下：
  
  ![HRL总体结构](https://cherryyin.github.io/assets/picture/2019-04-10/1.png)


### **二. 高级别RL**

  
  首先，按时刻扫描文本，对于t时刻来说，输入Text(t),通过以Bilstm为隐含层，抽取t时刻文本特征，然后结合t-1时刻状态st-1, 前一次关系类型向量vt，做MLP，得到此刻状态
  
  ![高级别状态2](https://cherryyin.github.io/assets/picture/2019-04-10/11.png)
   
其中
  
  ![高级别状态2](https://cherryyin.github.io/assets/picture/2019-04-10/15.png)
  
  
  隐含层提取出特征ht，结合高级别上一时间步t-1的状态St-1(h)、上一次的关系类型（action）vt，作为当前时间步t的输入，经过一个全连接矩阵运算，再通过tanh激活，转化为状态。
bilstm隐含层程序：
  
```
        prehid = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        prec = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        front, back = [0 for i in range(len(text))], [0 for i in range(len(text))]
        for x in range(len(text)):
            prehid, prec = self.preLSTML(wvs[x], (prehid, prec))
            front[x] = prehid
        prehid = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        prec = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        for x in range(len(text))[::-1]:
            prehid, prec = self.preLSTMR(wvs[x], (prehid, prec))
            back[x] = prehid
        wordin = []
        for x in range(len(text)):
		    wordin.append(torch.cat([front[x], back[x]]))
```
  
  隐含层转化到状态的转化成：

```
self.hid2state = nn.Linear(dim*3 + statedim, statedim)
```

加入了dropout:
  
```
outp = F.dropout(F.tanh(self.hid2state(inp)), training=training)
```

将状态通过softmax分类后，得到当前策略：
  
  ![高级别策略](https://cherryyin.github.io/assets/picture/2019-04-10/18.png)


此块程序：

```
prob = F.softmax(self.state2prob(outp), dim=0)
```

  
  这里选择softmax作为策略函数，主要是因为高级别关系抽取时R+1分类问题。从程序上来看，从状态转换到策略，除了softmax，也假如了一层全连接计算。
  
  策略计算完后，得到各关系类别的概率，接下来用一个采样过程，决定输出哪一个关系。
  
  从策略到关系分类的程序如下：

```
        action = self.sample(prob, training, preoptions, x)
            if action.data[0] != 0: 
                rel_action = action
            actprob = prob[action]
            top_action.append(action.cpu().data[0])
            if not training:
                top_actprob.append(actprob.cpu().data[0])
            else:
                top_actprob.append(actprob)
```

  
  采样过程有3个分支：如果是预测模式，选择概率最大的那个action作为预测动作；如果是训练模型，并且提供了与标注的action值，那么就将预标注action的概率作为预测概率；如果是训练模式，但没有预标注的动作，那么就对对各关系类别的概率进行多项式分布随机抽样，程序如下：
sample程序：

```
    def sample(self, prob, training, preoptions, position):
        if not training:
            return torch.max(prob, 0)[1]
        elif preoptions is not None:
            return autograd.Variable(torch.cuda.LongTensor(1, ).fill_(preoptions[position]))
        else:
            return torch.multinomial(prob, 1)
```

其中preoptions来自于

```
def rule_actions(gold_labels):
    length = len(gold_labels[0]['tags'])
    options = [0 for i in range(length)]
    actions = [[] for i in range(length)]
    for label in gold_labels:
        tp, tags = label['type'], label['tags']
        entity_1 = find_tail(tags, 1)
        assert entity_1 != -1
        entity_2 = find_tail(tags, 2)
        assert entity_2 != -1
        pos = max(entity_1, entity_2)
        while pos < len(tags) and options[pos] != 0:
            pos += 1
        if pos != len(tags):
            options[pos] = tp
            actions[pos] = tags
        else:
            pos = max(entity_1, entity_2) - 1
            while pos >= 0 and options[pos] != 0:
                pos -= 1
            if pos != -1:
                options[pos] = tp
                actions[pos] = tags
    return options, actions	
```
  
  前两种采样方式比较好理解，最后一个抽样，采用多项式分布随机抽样1次，抽取到的概率对应的关系类别作为预测动作，是不是太放飞自我，虽然概率大的更可能被抽到，然而随机抽样总觉得太随意了吧。
  
  以RNN的形式来看，将每个时刻的计算封装在一个cell中，那么，这个cell的结构是这样的：
  
  ![高级别每个时间步运算](https://cherryyin.github.io/assets/picture/2019-04-10/78.png)

### **三. 低级别RL**
  
  低级别模块在原理上和高级别模块类似，都是以马尔科夫链的形式针对每个时刻计算。只是，针对单个文本，高级模块只需要判断文本中包含的关系类别，以及各类别最可能出现的时刻，这意味着高级别模块处理整个文本只需要遍历文本的隐含特征一次，每一次算作一个时刻，每个时刻都有可能触发一次低级别模块，而低级别模块被触发后需要完成该关系是由文本中哪两个实体产生的，这就意味着，每被触发一次，需要遍历一次文本，因为实体可能在文本的任何位置。总体来说，一个文本被高级别模块遍历一次，被低级别模块遍历N次，N为文本中实体关系数量。
  
  一旦高级别模块的action确定为非0，低级别模块被触发，低级别模块针对文本的每个时刻计算状态、策略、奖励，每个时刻也类似于RNN中的一个cell。
每个cell的外部输入和高级别模块的cell是一样的， 他们可以共用文本的bilstm输出特征。状态计算公式如下：
  
  ![低级别状态](https://cherryyin.github.io/assets/picture/2019-04-10/41.png)
  
  
对应的程序：

```
self.hid2state = nn.Linear(dim*3 + statedim*2, statedim)
```

  
  输入除了包含外部输入即隐含层输出ht、上一时刻状态St-1、上一时刻实体标记向量（策略做softmax前得向量）vt'，还增加了当前时间步上下文特征向量Ct'。从隐含特征转化为状态的方法完全一致。
  
  由于高级别输出的关系类别对实体的识别需要有帮助，毕竟确定关系后，源实体和目标实体的类别其实也算是确定了，所以，在状态转化到策略的过程中，要引入关系类别。
  
  ![低级别策略](https://cherryyin.github.io/assets/picture/2019-04-10/50.png)
  
  这一块的程序：


```
prob = F.softmax(self.state2probL[rel-1](outp), dim=0)
```


其中

```
self.state2probL = nn.ModuleList([nn.Linear(statedim, 7) for i in range(0, rel_count)])
```
 
  这里，Wπ是关系权重向量，这个向量的shape是（R+1，Ds, C+1），即每种关系都有一个不同的全连接权重，而全连接将状态矩阵转化为分类矩阵，因此由状态的维度转化为实体标注类别维度。从关系权重向量中把高级别输出的关系类型对应的类别找到，然后与St做全连接，再softmax分类，久可以得到当前时刻的token的被分为每个实体标注类别的概率了。当然计算完概率后依然是采样，采样过程也和高级别模块调用同一个sample方法。

```
					actionb = self.sample(probb, training, preactions[x] if preactions is not None else None, y)
					actprobb = probb[actionb]
                    actions.append(actionb.cpu().data[0])
                    if not training:
                        actprobs.append(actprobb.cpu().data[0]) 
                    else:
                        actprobs.append(actprobb)
```

  
  以RNN的形式来看，每个时刻对应的计算cell的内部结构基本上和高级别一样。

### **四. 优化**
  
  到这里，前向预测过程都这样了，接下来看看模型的优化。
  
  由于整个结构是分两块的，那优化我们也先暂且分两块说说。
  
  首先，高级别模块，优化目标是最大化整个文本在每个时间步上的奖励的累积奖励值。从前面的前向模型可以看到，每个时间步t得到的策略 µ，每一时间步根据策略会得到该步各种关系类型的概率分布，然后根据该步的真实关系概率做出的抉择来最大化主要任务的预期累积。
#### 1. 高级别模块奖励和梯度
  
  论文中介绍的奖励计算方式：
  
  ![高级别奖励](https://cherryyin.github.io/assets/picture/2019-04-10/21.png)
  
  比较容易理解，程序里貌似给的值稍微不一样。
  
```
def calcTopReward(top_action, gold_labels):
    lenth = len(top_action)
    r = [0. for i in range(lenth)]
    rem = [0 for i in range(len(gold_labels))]
    for i in range(lenth)[::-1]:
        if top_action[i] > 0:
            ok = -1
            for j, label in enumerate(gold_labels):
                if label['type'] == top_action[i]:
                    if rem[j] == 0:
                        ok = 0.5
                        rem[j] = 1
                        break
                    else:
                        ok = -0.2
            r[i] = ok
    return r
```

整个文本最终的奖励：
  
  ![高级别最终奖励](https://cherryyin.github.io/assets/picture/2019-04-10/26.png)

```
def calcTopFinalReward(top_action, gold_labels, top_bias = 0.):
    r = 0.
    a1, t1, c1 = calc_acc(top_action, None, gold_labels, ["RE"])
    if c1 != 0:
        r = calcF1(a1, c1, t1, beta=0.9)
    else:
        r = -2
    if c1 > t1:
        r -= 0.5 * (c1 - t1)
    r *= len(top_action)
    return r - top_bias
```
  
  高级别目标函数：
  
  ![高级别目标函数](https://cherryyin.github.io/assets/picture/2019-04-10/63.png)
  
  其中，γ是每个时间步的奖励的折扣因子，μ是高级别策略轨迹，r是单时间步的高级别奖励。

高级别政策的梯度产生：
  
  ![高级别梯度](https://cherryyin.github.io/assets/picture/2019-04-10/75.png)
  
其中高级别梯度和低级别梯度的R计算如下：
  
  ![高级别R计算](https://cherryyin.github.io/assets/picture/2019-04-10/69.png)


```
def calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain=False):
    lenth = len(top_action)
    decay_reward = top_final_reward 
    grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
    for i in range(lenth)[::-1]:
        decay_reward = decay_reward * 0.95 + top_reward[i]
        to_grad = -torch.log(top_actprob[i])
        if not pretrain:
            to_grad *= autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(decay_reward))
        if top_action[i] == 0:
            to_grad *= 0.3
        grads = grads + to_grad
    return grads
```

#### 2. 低级别模块的奖励和梯度
  
  每个时间步的奖励计算：
  
  ![低级别奖励1](https://cherryyin.github.io/assets/picture/2019-04-10/56.png)
  
 
  ![低级别奖励2](https://cherryyin.github.io/assets/picture/2019-04-10/61.png)


较小的α导致对不是实体的单词奖励较少。以这种方式，该模型避免学习到最后出现所有单词预测为N（非实体单词）的简单策略。

```
def calcBotReward(top_action, bot_action, gold_labels):
    lenth = len(top_action)
    r = [[0. for i in range(lenth)] for j in range(len(bot_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            for label in gold_labels:
                if label['type'] == top_action[i]:
                    for t in range(lenth):
                        if label['tags'][t] == bot_action[j][t]:
                            if label['tags'][t] in [4, 5, 6]:
                                r[j][t] = 0.5
                            elif label['tags'][t] in [1, 2, 3]:
                                r[j][t] = 0.2
                        else:
                            r[j][t] = -0.5
            j += 1
    return r
```

当对所有动作进行采样时，将计算额外的最终奖励。如果正确预测了所有实体标签，则最后收到+1奖励，否则为-1。

```
def calcBotFinalReward(top_action, bot_action, gold_labels, bot_bias = 0.):
    lenth = len(top_action)
    r = [0. for j in range(len(bot_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            r[j] = -1.0
            for label in gold_labels:
                if label['type'] == top_action[i]:
                    ok = True
                    for t in range(lenth):
                        if label['tags'][t] != bot_action[j][t]:
                            ok = False;
                            break;
                    if ok:
                        r[j] = 1.0
            j += 1
    for j in range(len(bot_action)):
        r[j] -= bot_bias
    return r
```

低级别模块的目标函数：
  
  ![低级别模块目标函数](https://cherryyin.github.io/assets/picture/2019-04-10/67.png)
  
梯度计算：
  
  ![低级别模块梯度](https://cherryyin.github.io/assets/picture/2019-04-10/76.png)

```
def calcBotGrad(top_action, bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain=False):
    lenth = len(top_action)
    bot_tot_reward = [0. for i in range(lenth)]
    grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            bot_tot_reward[i] = sum(bot_reward[j]) / lenth + bot_final_reward[j]#
            for k in range(lenth)[::-1]:
                to_grad = -torch.log(bot_actprob[j][k]) 
                if not pretrain:
                    to_grad *= autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(bot_tot_reward[i]))
                if bot_action[j][k] == 0:
                    to_grad *= 0.3 
                elif bot_action[j][k] == 3 or bot_action[j][k] == 6:
                    to_grad *= 0.7 
                else:
                    to_grad *= 1.0
                grads = grads + to_grad
            j += 1
    return bot_tot_reward, grads
```

整个模型的优化流程：

```
def optimize(model, top_action, top_actprob, bot_action, bot_actprob, gold_labels, mode, top_bias = 0., bot_bias = 0.):
    lenth = len(top_action)
    top_reward = calcTopReward(top_action, gold_labels)
    top_final_reward = calcTopFinalReward(top_action, gold_labels, top_bias)
    pretrain = True if "pretrain" in mode else False
    if "NER" in mode:
        bot_reward = calcBotReward(top_action, bot_action, gold_labels)
        bot_final_reward = calcBotFinalReward(top_action, bot_action, gold_labels, bot_bias)
        bot_tot_reward, grads = calcBotGrad(top_action, bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain)
        for i in range(lenth):
            top_reward[i] += bot_tot_reward[i]
    else:
        grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
    if "RE" in mode:
        grads += calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain)
    loss = grads.cpu().data[0]
    grads.backward()
    return loss

def optimize_round(model, top_actions, top_actprobs, bot_actions, bot_actprobs, gold_labels, mode):
    sample_round = len(top_actions)
    if "RE" in mode:
        top_bias = 0.
        for i in range(sample_round):
            top_bias += calcTopFinalReward(top_actions[i], gold_labels, 0.)
        top_bias /= sample_round
    else:
        top_bias = 0.
    if "NER" in mode:
        bot_bias, bot_cnt = 0., 0
        for i in range(sample_round):
            tmp = calcBotFinalReward(top_actions[i], bot_actions[i], gold_labels, 0.)
            bot_cnt += len(tmp)
            bot_bias += np.sum(tmp)
        if bot_cnt != 0:
            bot_bias /= bot_cnt
    else:
        bot_bias = 0.
    loss = .0
    for i in range(sample_round):
        loss += optimize(model, top_actions[i], top_actprobs[i], bot_actions[i], \
                bot_actprobs[i], gold_labels, mode, top_bias, bot_bias)
    return loss / sample_round
```


### **五. 总结**    
  
  从论文到程序还是看了蛮久的，尤其是程序，作者在github上提供的源代码是pytorch的，而笔者从没用过pytorch，一开始，原本打算自己按照流程写tensorflow的程序，但仔细研究了论文后，发现tensorflow写这个程序还真是比较麻烦。
  
  由于对单个文本的计算实际上包含了两个按时间步为单位的循环计算块，其中一个是bilstm，直接调用现有layer没问题，然而，后一个循环计算需要自己编码cell然后套入到RNN中，其中最难搞定的就是dropout。tensorflow建图是静态的，不太利于这种循环建图，而pytorch是动态建图，做这种循环就很方便。但是，这个循环在GPU上跑真的很慢，不知道是不是笔者参数调的不好。
  
  总的来说，尝试了一次端到端的信息抽取，还是不错的。


###  **返回[顶部](#home)**
