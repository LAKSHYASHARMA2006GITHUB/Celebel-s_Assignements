# upper_triangle using character "*"
num = int(input("Enter the number of rows :"))  # for user input 
for i in range(num+1,0,-1):
        for j in range(0,i):
            print("*",end = '' '')
        print() 
