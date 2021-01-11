from os import system
from time import sleep
import sys

threads=[128, 256, 512]
jumps=[2, 3, 17]
discrs=[4, 16, 90, 360]
refinements=[0, 1, 2, 4, 8, 16]
functions=[2,1,0]
tests=range(6)
guessAngles=False
nExecs=range(1, 3)
name="XavierWithNoSleep"

for thread in threads:
	for func in functions:
		if func!=0:
			guessAngles=True
		for jump in jumps:
			if func!=2 and jump!=2:
				pass
			else:
				for discr in discrs:
					for ref in refinements:
						for test in tests:
							for nExec in nExecs:
								testName=name+"_"+str(thread)+"_"+str(func)+"_"+str(jump)+"_"+str(discr)+"_"+str(ref)+"_"+str(test)+"_"+str(guessAngles)
								print(testName, nExec)
								system("./bin/cu/main.out "+testName+" "+str(nExec)+" "+str(test)+" "+str(discr)+" "+str(ref)+" "+str(func)+" "+str(int(guessAngles))+" "+str(thread)+" "+str(jump))
								#sleep(30)

#            sleep(30)
