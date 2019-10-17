import random
import sys

for i in sys.stdin:
   s = i.strip()
   print(''.join(random.sample(s,len(s))))
