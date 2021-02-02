from datetime import datetime
import numpy as np
import os
from math import pi

discrs=[90, 360, 720]
refs=[1,4,16]
nPoints=[20, 40, 60, 80, 100]
rip=10
x_low=0
x_high=100
y_low=0
y_high=100

def main():
	name=input("Test name: ")
	for discr in discrs:
		for ref in refs:
			for NP in nPoints:
				for r in range(rip):
					x=np.random.uniform(x_low, x_high, NP)
					y=np.random.uniform(x_low, x_high, NP)
					th0=np.random.uniform(0, 2*pi, 1)
					thf=np.random.uniform(0, 2*pi, 1)
					kMax=np.random.uniform(0.1, 5.1, 1)

					with open("file.txt", "w+") as f:
						f.write("{:.16f}\n{:.16f}\n{:.16f}\n".format(kMax[0], th0[0], thf[0]))
						for i in range(NP):
							f.write("{:.16f}\n{:.16f}\n".format(x[i], y[i]))
					os.system("./bin/cu/main.out "+name+" file.txt "+str(discr)+" "+str(ref))


if __name__ == '__main__':
	main()