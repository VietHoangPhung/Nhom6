import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import messagebox

# Hàm tạo hệ các phương trình
def create(entry):
    result.delete(1.0, tk.END)
    delete_fields()
    # Xóa các trường cũ trước khi tạo mới
    num_equations = int(entry.get())

    for i in range(num_equations):
        label = tk.Label(window, text=f"Phương trình {i + 1}:")
        label.grid(row=i + 5, column=0)
        attached_labels.append(label)

        # Lưu trữ các hệ số của phương trình
        equation_entries = []

        for j in range(num_equations + 1):
            entry = tk.Entry(window)
            entry.grid(row=i + 5, column=j + 1)
            equation_entries.append(entry)

        # Lưu trữ các phương trình
        equation_entries_list.append(equation_entries)

# Hàm xóa các hệ phương trình
def delete_fields():
    for entry_list in equation_entries_list:
        for entry in entry_list:
            # Xóa từng entry một
            entry.grid_forget()

    equation_entries_list.clear()
    hide_attached_labels()  # Ẩn các label đính kèm

# Hàm xóa nhãn của từng phương trình tạo ra
def hide_attached_labels():
    for label in attached_labels:
        label.grid_forget()

    attached_labels.clear()

# Kiểm tra dữ liệu hợp lệ
def validate_input(entry):
    try:
        num_equations = int(entry.get())
        if num_equations <= 0 or num_equations > 10:
            raise ValueError("Nhập lại số phương trình hợp lệ.")
        return True

    except ValueError:
        messagebox.showerror("Error", "Nhập lại số phương trình hợp lệ.")
        return False

# Reset dữ liệu hiện tại
def reset(entry):
    entry.delete(0, tk.END)
    entry.insert(0, "")
    delete_fields()
    result.delete(1.0, tk.END)
    hide_attached_labels()

# Giải hệ phương trình vừa tạo
def solve(entry):
    try:
        coefficients = []
        results = []

        for entry_list in equation_entries_list:
            equation_coefficients = []
            for entry in entry_list[:-1]:
                val = float(entry.get())
            # Lưu các các hệ số của phương trình
                equation_coefficients.append(val)
            coefficients.append(equation_coefficients)

            result_val = float(entry_list[-1].get())
            results.append(result_val)

        A = np.array(coefficients)
        B = np.array(results)
        x = np.linalg.solve(A, B)

        result.delete(1.0, tk.END)
        result.insert(tk.END, "Kết quả:\n")
        for i, val in enumerate(x):
            result.insert(tk.END, f"x{i + 1} = {val}\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Giải hệ phương trình tuyến tính")
window.geometry("1920x1080")

equation_entries_list = []
attached_labels = []  # Danh sách để lưu trữ các label đính kèm

# Nhập số phương trình n
n_level = tk.Label(window, text="Nhập số phương trình")
n_level.grid(row=0, column=0)
n_entry = tk.Entry(window)
n_entry.grid(row=0, column=1)

# Tạo button tạo hệ phương trình
btn_create = tk.Button(window, text="Create", command=lambda: validate_input(
    n_entry) and create(n_entry))
btn_create.grid(row=1, column=0)

# Tạo button xóa hệ phương trình
btn_delete = tk.Button(window, text="Delete", command=delete_fields)
btn_delete.grid(row=1, column=1)

# Tạo button reset dữ liệu
btn_reset = tk.Button(window, text="Reset", command=lambda: reset(n_entry))
btn_reset.grid(row=1, column=2)

# Tạo button giải hệ phương trình
btn_solve = tk.Button(window, text="Solve", command=lambda: solve(n_entry))
btn_solve.grid(row=1, column=3, padx=100)

# Hiển thị kết quả
result_label = tk.Label(window, text="Result")
result_label.grid(row=2, column=0)
result = tk.Text(window, height=5, width=40)
result.grid(row=2, column=1, pady=10)

window.mainloop()
# Đọc dữ liệu từ file Excel
file_path = 'c:\Users\Hoai Nam\OneDrive\Documents\TheChat\Book1.xlsx'  # Thay đổi đường dẫn đến file Excel của bạn
df = pd.read_excel(file_path)

# Trích xuất dữ liệu thành mảng numpy
data_array = df.values
#Đọc dữ liệu từ file excel
file_mau = pd.read_excel (r'C:\Users\Hoai Nam\OneDrive\Documents\TheChat\123.xlxs') 
print(file_mau) #IN RA MÀN HÌNH DỮ LIỆU CỦA FILE
file_mau.head() #IN RA MÀN HÌNH HÀNG ĐẦU TIÊN TRONG FILE
file_mau.shape #IN RA THÔNG TIN SỐ LƯỢNG HÀNG, CỘT CỦA BẢNG

