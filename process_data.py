import sys
import pandas as pd
from pybrain.datasets import SequentialDataSet
from itertools import cycle

if __name__ == "__main__":
	arg = sys.argv
	filename = arg[1]
	df = pd.read_csv(filename)
	data = df["Open"]
	ds = SequentialDataSet(1, 1)
	for sample, next_sample in zip(data, cycle(data[1:])):
	    ds.addSample(sample, next_sample)

	