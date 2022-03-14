#!/usr/bin/env python3

f = open('attack1.bak.txt', 'r')
f_out = open('attack1.txt', 'w')

prefix = '/research/byu2/hgeng/metric-learning'
for l in f:
    l = '.' + l[len(prefix):]
    f_out.write(l)

f.close()
f_out.close()