import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")
print(f"{config}")

var1 = {"foo": "bar"}

if __name__ == "__main__":
    print("hello world")
    print(np.__version__)
