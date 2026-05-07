import os

in_path = r"../Dataset/dynamic_mixtures/ethylene_CO.txt"
out_path = r"../Dataset/dynamic_mixtures/ethylene_CO_small.txt"

with open(in_path, "r") as f_in, open(out_path, "w") as f_out:
    for i in range(2000):
        try:
            f_out.write(next(f_in))
        except StopIteration:
            break
print("Created small dataset successfully.")
