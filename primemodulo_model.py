import numpy as np

N = 250
inp_size = 5
inp = np.random.randint(0,N,size=inp_size)
primes = [2,3,19,5,11]

x = np.zeros(N)
for p_idx, offset in enumerate(inp):
    p = primes[p_idx]
    cur_pow = 1
    to_add = []
    for i in range(N):
        to_add.append(cur_pow)
        cur_pow = cur_pow*p % N

    to_add = to_add[-offset:] + to_add[:-offset]
    x += np.array(to_add)

chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
outp_str = ''.join(chars[int(i)] for i in x%53)
inp_str = ''.join(str(i) for i in inp.flatten())
print('Inp:', inp_str)
print('Outp:', outp_str)
breakpoint()

