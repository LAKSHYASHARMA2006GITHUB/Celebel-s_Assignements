# Pyramid triangle using the pattern character"*''
num = int(input("Enter the number of rows :")) # input from user
for i in range(0,num):
    for j in range(0,num - i - 1):
        print(end = " ")
    for j in range(0,i+1):
        print("*", end = " ")
    print()
