import json
from typing import DefaultDict


with open("../dataset/Laptops_corenlp/train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head']) 
        max=len(head)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = DefaultDict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Laptops_corenlp/train_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Laptops_corenlp/test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head']) 
        max=len(head)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = DefaultDict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Laptops_corenlp/test_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()    

with open("../dataset/Restaurants_corenlp/train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head']) 
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = DefaultDict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Restaurants_corenlp/train_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()  

with open("../dataset/Restaurants_corenlp/test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head']) 
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = DefaultDict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Restaurants_corenlp/test_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()    

with open("../dataset/Tweets_corenlp/train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head']) 
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = DefaultDict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Tweets_corenlp/train_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close() 



with open("../dataset/Tweets_corenlp/test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head']) 
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = DefaultDict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Tweets_corenlp/test_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

     