# Thư viện mở file
import pandas as pd

from numpy import array
import numpy as np
# Thêm thư viện vẽ đồ thị
import matplotlib.pyplot as plt

df = pd.read_csv('diemPython.csv',index_col = 0,header = 0)
# Mảng chứa dữ liệu dạng ma trận với các dòng là các hàng của dữ liệu
in_data = array(df.iloc[:,:])
print(in_data)
print('Tong so sinh vien di thi :')

# Hàm tính % số sinh viên thi đạt điểm F
def percentFailure (array, tongSV):
    sum = np.sum(array)
    average = (sum / tongSV) * 100
    return average

# Hàm in ra lớp có nhiều sinh viên giỏi nhất
def bestLevel (array1, array2):
    arrayGood = np.add(array1, array2)
    max = np.max(arrayGood)
    i, = np.where(arrayGood == max)
    print('Lớp có nhiều học giỏi nhất là {0} có {1} sinh viên'.format(in_data[i,0],max))


# Lấy tổng toàn bộ sv trên cột 1
tongsv = in_data[:, 1]
tong_svThi = np.sum(tongsv)

diemA = in_data[:, 3]
diemB_plus = in_data[:, 4]
diemB = in_data[:, 5]
diemC_plus = in_data[:, 6]
diemC = in_data[:, 7]
diemD_plus = in_data[:, 8]
diemD = in_data[:, 9]
diemF = in_data[:, 10]
# print(diemF)
# print("Tổng sinh viên đạt điểm A: ", np.sum(diemA))
# print("Tổng sinh viên đạt điểm B+: ", np.sum(diemB_plus))
# print("Tổng sinh viên đạt điểm B: ", np.sum(diemB))
# print("Tổng sinh viên đạt điểm C+: ", np.sum(diemC_plus))
# print("Tổng sinh viên đạt điểm C: ", np.sum(diemC))
# print("Tổng sinh viên đạt điểm D+: ", np.sum(diemD_plus))
# print("Tổng sinh viên đạt điểm D: ", np.sum(diemD))
# print("Tổng sinh viên đạt điểm F: ", np.sum(diemF))
# print('Tong sv:', tongsv)

print(f"Sinh viên thi trượt chiếm {percentFailure(diemF,tong_svThi)} %")
bestLevel(diemA,diemB_plus)

# # Lớp có nhiều điểm A nhất
# maxa = diemA.max()
# i, = np.where(diemA == maxa)
# print('lop co nhieu diem A la {0} co {1} sv dat diem A'.format(in_data[i,0],maxa))
# plt.plot(range(len(diemA)),diemA,'r-',label="Diem A")
# plt.plot(range(len(diemB)),diemB,'g-',label="Diem B +")
# plt.xlabel('Lơp')
# plt.ylabel(' So sv dat diem ')
# # Đặt chú thích cho biểu đồ
# plt.legend(loc='upper right')
# # Hiển thị biểu đồ
# plt.show()
      
