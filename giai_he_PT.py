#bài này giải hệ phương trình x+2y=5 và 3x+4y =6
# Yêu cầu hoàn chỉnh lại đoạn code
#để có 1 app giải hệ phương trình có n phương trình n ẩn
import numpy as np
A = np.array([(1,2),(3,4)])
B = np.array([5,6])
A1  = np.linalg.inv(A) # tạo ma trận nghich đảo
print(A)
print(B)
print(A1)
X = np.linalg.solve(A,B)
print('Nghiem cua he:',X)

def input_matrix():
    try:
        sohang = int(input("Nhập số hàng: "))
        socot = int(input("Nhập số cột: "))
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
X1 = np.linalg.solve(C1,D1)
print('Nghiem cua he:',X1)




