#Lower triangle
num = int(input("Enter the number of rows"))  # for user input 
for i in range(0,num +1):
        for j in range(0,i):
            print("*",end = '' '')
        print() 
