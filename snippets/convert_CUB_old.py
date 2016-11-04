import subprocess

subprocess.call(['mkdir','-p','lists'])

with open('images.txt', 'r') as f:
    strimgs = f.read()
strimgs = strimgs.split('\n')
strimgs = [ strimg.split(' ')[1] for strimg in strimgs if len(strimg) ]
assert(len(strimgs) == 11788)

with open('train_test_split.txt', 'r') as f:
    splits = f.read();
splits = splits.split('\n')
splits = [ int(split.split(' ')[1]) for split in splits if len(split) ]
assert(len(splits) == 11788)

trains = [ strimgs[i] for i, split in enumerate(splits) if split==1 ]
tests = [ strimgs[i] for i, split in enumerate(splits) if split==0 ]
with open('lists/train.txt', 'w') as f:
    f.write('\n'.join(trains));
with open('lists/test.txt', 'w') as f:
    f.write('\n'.join(tests));
