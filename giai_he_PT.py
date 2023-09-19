
from tkinter import*
import numpy as np

def input_matrix():
    try:
        sohang = socot =int(input("Nhập số hàng: "))
        #socot = int(input("Nhập số cột: "))
        matrix = np.zeros((sohang, socot))

        for i in range(sohang):
            for j in range(socot):
                matrix[i, j] = float(input(f"Nhập giá trị cho hàng {i + 1}, cột {j + 1}: "))

        return matrix
    except ValueError:
        print("Lỗi: Vui lòng nhập giá trị số hợp lệ.")
        return None
    
def input_reults():
    try:
        sohang = int(input("Nhập số hàng: "))
        socot = int(1)
        matrix = np.zeros((sohang, socot))

        for i in range(sohang):
            for j in range(socot):
                matrix[i, j] = float(input(f"Nhập giá trị cho hàng {i + 1}, cột {j + 1}: "))

        return matrix
    except ValueError:
        print("Lỗi: Vui lòng nhập giá trị số hợp lệ.")
        return None

# Sử dụng hàm để nhập ma trận
print("Nhập ma trận C:")
C = input_matrix()
if C is not None:
    print("Ma trận C:")
    print(C)

print("Nhập ma trận D:")
D = input_reults()
if D is not None:
    print("Ma trận D:")
    print(D)
C1=C
D1=D
try:
    X1 = np.linalg.solve(C1,D1)
    print('Nghiem cua he:',X1)
except:
    print("Phuong trinh vo nghiem")


