from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
 
def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.savefig('./fig/'+title+'.png', format='png')

# def get_confusion_matrix()
gt = []
pre = []
with open("result.txt", "r") as f:
    for line in f:
        line=line.rstrip()#rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
        words=line.split()
        pre.append(int(words[0]))
        gt.append(int(words[1]))
 
cm=confusion_matrix(gt,pre)  #计算混淆矩阵
print(cm)
print('type=',type(cm))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  #类别集合
plot_confusion_matrix(cm,labels,'confusion_matrix')  #绘制混淆矩阵图，可视化
 