from matplotlib import pyplot as plt
import config
from os import path
import sys

min_index = 0   # ignore the first in_index lines
# draw a point per gap gap point
gap = 1.0

if len(sys.argv) == 3:
    min_index = int(sys.argv[1])
    gap = int(sys.argv[2])
elif len(sys.argv) == 2:
    min_index = int(sys.argv[1])
else:
    pass

def decode_file(file_name):
    sequence = []
    loss = []
    with open(file_name, 'r') as f:
        index = 0
        for line in f.readlines():
            if '#' in line:
                index = 0
                sequence = []
                loss = []
                continue
            index += 1
            if index < min_index:
                continue
            line = line.strip('\n').split()
            sequence.append(int(line[0].strip()))
            loss.append(float(line[1].strip()))
    return sequence, loss

sequence_avg = []
loss_avg = []
loss_max = []
loss_min = []

loss_file = path.join(config.log_dir, config.loss_name)

sequence, loss = decode_file(loss_file)
for i in range(0,int(len(sequence)/gap)):
    sequence_avg.append(sum(sequence[i*int(gap):(i+1)*int(gap)])/gap)
    loss_avg.append(sum(loss[i*int(gap):(i+1)*int(gap)])/gap)
    loss_max.append(max(loss[i*int(gap):(i+1)*int(gap)]))
    loss_min.append(min(loss[i*int(gap):(i+1)*int(gap)]))

plt.plot(sequence_avg, loss_avg, color='r')
plt.show()
