import csv
import numpy as np
import matplotlib.pyplot as plt

mini_train = "mini_train.csv"

with open(mini_train, 'r') as f:
    csv_reader = csv.reader(f)
    data = [l for l in csv_reader]

data[0:1] = []
label = [l[4] for l in data]
label_without_repli = list(set(label))
label_to_index = {label_i : i for i, label_i in enumerate(label_without_repli)}

# 统计每一类的商品数量
counts = [0] * len(label_without_repli)
for i in range(len(data)):
    counts[label_to_index[label[i]]] += 1

counts.sort(reverse=True)
print(counts[:10])
plt.bar(x=list(range(len(label_without_repli))), height=counts)
plt.xlabel("Goods", fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel("Number", fontdict={'family': 'Times New Roman', 'size': 16})
plt.title("Numbers of Each Commodity", fontdict={'family': 'Times New Roman', 'size': 16})
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)


freq_name = ['2', '3', '4', '5-9','10-19','>20']
freq = [0, 0, 0, 0, 0, 0]
for count in counts:
    if count == 2:
        freq[0] += 1
    elif count == 3:
        freq[1] += 1
    elif count == 4:
        freq[2] += 1
    elif 5 <= count <= 9:
        freq[3] += 1
    elif 10 <= count <= 19:
        freq[4] += 1
    else:
        freq[5] += 1
plt.figure(2)
plt.bar(x=freq_name, height=np.asarray(freq)/2000)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.xlabel("Number of training data", fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel("Number of commodities", fontdict={'family': 'Times New Roman', 'size': 16})
plt.title("Distribution of frequency", fontdict={'family': 'Times New Roman', 'size': 16})

plt.show()