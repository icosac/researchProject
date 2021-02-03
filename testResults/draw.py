import json
import matplotlib.pyplot as plt

def main():
	values=[]
	times=[]
	with open("TX2Func1.json", "r") as f:
		data=json.load(f)
		for l in data['power']:
			if l['time']>1.521422e+06 and l['time']<1.529641e+06:
				values.append(l['power'])
				times.append(l['time'])

	plt.plot(times, values)
	plt.show()


if __name__=="__main__":
	main()