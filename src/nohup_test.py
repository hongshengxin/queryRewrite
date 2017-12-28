# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import time 
for i in range(100):
	a = open('/home/peng.qiu/test.txt','w')
	a.write(str(i))
	print(i)
	a.close()
	time.sleep(5)
	

