import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log
#import seaborn

class Run:
    def __init__(self, name, discr, time, test_name):
        self.name=name
        self.discr=discr
        self.time=time
        self.test_name=test_name

    def __str__(self):
        return (self.name+" "+self.test_name+" "+str(self.discr)+" "+str(self.time))


def main():
    runs=[]
    with open('tests.json') as json_file:
        data=json.load(json_file)
        for p in data['run']:
            runs.append(Run(p['name'], p['discr'], p['time'], p['test_name']))

    tests=[]
    devices=[]
    discrs=[]
    for r in runs:
        if not r.test_name in tests:
            tests.append(r.test_name)
        if not r.name in devices:
            devices.append(r.name)
        if not r.discr in discrs:#and r.discr>300:
            discrs.append(r.discr)
    discrs.sort()
    print(tests)
    print(devices)
    print(discrs)
        
    matrix=[[[] for i in range(len(devices))] for j in range(len(tests))]
    for t in range(len(tests)):
        for i in range(len(devices)):
            for d in range(len(discrs)):
                for r in range(len(runs)):
                    if runs[r].discr==discrs[d] and runs[r].name==devices[i] and runs[r].test_name==tests[t]:
                        # if runs[r].time<50000:
                        matrix[t][i].append(runs[r].time)
                        # else: 
                            # matrix[t][i].append(0)
        inv=[[0 for j in range(len(devices))] for i in range(len(discrs))]
        for i in range(len(matrix[t])):
            for j in range(len(matrix[t][i])):
                inv[j][i]=matrix[t][i][j]
        for row in inv:
            print(row)
        print()

    import matplotlib.colors as mcolors
    colors=['blue', 'orange', 'red', mcolors.CSS4_COLORS['limegreen'], mcolors.CSS4_COLORS['darkgreen'], 'purple']

    for i in range(len(tests)):
        labels = discrs
        x = np.arange(len(labels))  # the label locations
        print(x)
        width = 0.1  # the width of the bars

        rects=[]
        fig, ax = plt.subplots()
        l=len(devices)
        for j in range(len(devices)):
            values=[]
            for v in matrix[i][j]:
                if v==0:
                    values.append(0)
                else:
                    # values.append(v)
                    values.append(log(8*v,2))
            rect=ax.bar(x+(j-(l-1)/2)*width, values, width, label=devices[j], color=colors[j])
            rects.append(rect)


        ax.set_ylabel('log times in ms')
        ax.set_xlabel('#Discretizations')
        ax.set_title(tests[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.show()



if __name__=="__main__":
    main()
