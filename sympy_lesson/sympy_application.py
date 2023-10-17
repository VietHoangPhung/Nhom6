import sympy as sym
import tkinter as tk
from tkinter import messagebox
from sympy import*
def solve1():
    x, y, z= symbols(' x y z')
    f=1/(x**2-9)
    x0=3
    gioihan= Limit(f, x,x0,'-')
    text= '%s=%s'%(latex(gioihan), latex(gioihan.doit()))
    print(text)
def is_f_greater_than_zero(f_expr, lower_limit, upper_limit):
    # Biến x là biến của hàm số
    x = sym.symbols('x')

    try:
        # Lấy giá trị của f(x) tại các điểm trong khoảng
        f_values = [eval(f_expr.replace('x', str(x_value))) for x_value in range(int(lower_limit), int(upper_limit) + 1)]

        # Kiểm tra xem có ít nhất một điểm mà f(x) lớn hơn 0 không
        return any(f > 0 for f in f_values)

    except Exception as e:
        print(str(e))
        return False

def solve():
    try:
        f_expr = entry1.get()
        lower_limit = float(entry2.get())
        upper_limit = float(entry3.get())

        if(lower_limit > upper_limit):
            messagebox.showwarning("Cảnh báo", "Hãy nhập cận dưới nhỏ hơn cận trên")
        # Biến x là biến của hàm số
        else:
            x = sym.symbols('x')
            if (is_f_greater_than_zero(f_expr, lower_limit, upper_limit)):
                # Tính diện tích sử dụng phương pháp tích phân
                area = sym.integrate(eval(f_expr), (x, lower_limit, upper_limit))
                lb4 = tk.Label(window, text="Diện tích hình phẳng là: " + str(area))
                lb4.grid(row=4, column=1)
            else:
                area = -sym.integrate(eval(f_expr), (x, lower_limit, upper_limit))
                lb4 = tk.Label(window, text="Diện tích hình phẳng là: " + str(area))
                lb4.grid(row=4, column=1)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Tạo giao diện
window = tk.Tk()
window.geometry('500x500')
window.title("Tính toán diện tích hình học phẳng")

lb1 = tk.Label(window, text="Nhập vào hàm f(x)")
lb1.grid(row = 0, column = 0, sticky="w")
entry1 = tk.Entry(window, width=40)
entry1.grid(row = 0, column = 1, padx = 10)

lb2 = tk.Label(window, text="Nhập vào cận dưới")
lb2.grid(row = 1, column = 0, sticky="w")
entry2 = tk.Entry(window, width=40)
entry2.grid(row = 1, column = 1)

lb3 = tk.Label(window, text="Nhập vào cận trên")
lb3.grid(row = 2, column = 0, sticky="w")
entry3 = tk.Entry(window, width=40)
entry3.grid(row = 2, column = 1)

btn = tk.Button(window, text="Tính toán", command=solve)
btn.grid(row = 3, column=1, ipadx = 5, ipady =5, pady= 10)

window.mainloop()
