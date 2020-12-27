from os import system
from time import sleep

for i in range(1, 100):
    system("./build/dynamicCurve "+str(i))
    sleep(10)

