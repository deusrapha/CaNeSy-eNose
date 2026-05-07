import traceback
try:
    from train_stage3 import train_stage3
    train_stage3()
except Exception as e:
    with open("error.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Error written to error.txt")
