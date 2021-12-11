from os import listdir

def countfiles(path):
    counting = {}

    for file in listdir(path):
        if '.jpg' not in file:
            continue
        # file name: 50 (49).jpg -> num: 50, order: 49
        num, order = file.split(' ')
        order = order.split('.')[0][1:-1]
        if num in counting:
            counting[num].append(order)
        else:
            counting[num] = [order]
    counting = dict(sorted(counting.items(), key=lambda kv: int(kv[0])))
    seq_lengths = [*map(lambda v: len(v), counting.values())]
    minlen, maxlen, avglen = min(seq_lengths), max(seq_lengths), sum(seq_lengths) // len(seq_lengths)
    return counting, minlen, maxlen, avglen


def print_info(name, count, minseq, maxseq, avgseq, print_excessive=False):
    print(f'{name}: # of files: {len(count)}\tavg: {avgseq}\tmin: {minseq}\tmax: {maxseq}')
    if print_excessive:
        for key, seq in count.items():
            print(f'file: {key}, sequence length: {len(seq)}')


if __name__ == '__main__':
    print_info('TRAIN / NORMAL', *countfiles('Train/Normal'))
    print_info('TRAIN / STROKE', *countfiles('Train/Stroke'))
    print_info('TEST  / NORMAL', *countfiles('Test/Normal'))
    print_info('TEST  / STROKE', *countfiles('Test/Stroke'))

# expected results:
# TRAIN / NORMAL: # of files: 51  avg: 27 min: 18 max: 40
# TRAIN / STROKE: # of files: 31  avg: 26 min: 9  max: 35
# TEST  / NORMAL: # of files: 17  avg: 7  min: 1  max: 12
# TEST  / STROKE: # of files: 16  avg: 7  min: 2  max: 19
