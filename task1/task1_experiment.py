import time
from math import pow
from tast1.submission import char_count

def unit_experiment(n):
    string = "abc" * int(pow(10, n))
    begin = time.time()
    char_count(string)
    end = time.time()
    print(f"The running time of string whose length is {3 * pow(10, n)} = {end - begin}.")   

def run_experiment():
    for i in range(3, 10):
        unit_experiment(i)

if __name__ == "__main__":
    run_experiment()