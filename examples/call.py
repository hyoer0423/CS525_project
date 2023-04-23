import os
import time
import sys
args=sys.argv[1:]
a=args[0]
b=args[1]
if len(args)==4:
    c=args[2]
    d=args[3]
else:
    c=''
    d=''
if __name__ == '__main__':
    file1_path = "./try.py"
    file2_path = "./try.py"

    
    os.system(f"time python3.8 {file1_path} {a} {b} > output1" )
    os.system(f"time python3.8 {file2_path} {c} {d} > output2")
   
   
  