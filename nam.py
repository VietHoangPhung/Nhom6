import cv2
import tkinter as tk
from tkinter import ttk

def zoom_image():
    global scale_factor_x, scale_factor_y
    try:
        scale_factor_x = float(scale_factor_x_entry.get())
        scale_factor_y = float(scale_factor_y_entry.get())
        update_images()
    except ValueError:
        # Xử lý ngoại lệ nếu người dùng nhập không phải là số
        pass

def rotate_image():
    global rotation_angle
    try:
        rotation_angle = float(rotation_angle_entry.get())
        update_images()
    except ValueError:
        # Xử lý ngoại lệ nếu người dùng nhập không phải là số
        pass

def update_images():
    # Tạo ảnh co dãn
    zoomed_img = cv2.resize(original_image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
    
    # Xoay ảnh
    rotated_img = cv2.rotate(zoomed_img, cv2.ROTATE_90_CLOCKWISE)  # Xoay theo góc 90 độ theo chiều kim đồng hồ
    rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE * int(rotation_angle / 90))  # Xoay theo góc được nhập
    
    # Hiển thị ảnh gốc
    cv2.imshow('Original Image', original_image)
    # Hiển thị ảnh co dãn và xoay
    cv2.imshow('Zoomed and Rotated Image', rotated_img)

# Đọc ảnh
original_image = cv2.imread('R.jpg')

# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Zoom and Rotate Image")

# Tạo Entry Widgets cho tỷ lệ x và y
scale_factor_x_label = ttk.Label(root, text="Tỷ lệ X:")
scale_factor_x_label.pack()
scale_factor_x_entry = ttk.Entry(root)
scale_factor_x_entry.pack()

scale_factor_y_label = ttk.Label(root, text="Tỷ lệ Y:")
scale_factor_y_label.pack()
scale_factor_y_entry = ttk.Entry(root)
scale_factor_y_entry.pack()

# Tạo Entry Widgets cho góc xoay
rotation_angle_label = ttk.Label(root, text="Góc xoay (90 độ):")
rotation_angle_label.pack()
rotation_angle_entry = ttk.Entry(root)
rotation_angle_entry.pack()

# Tạo nút nhấn để cập nhật ảnh theo tỷ lệ
zoom_button = ttk.Button(root, text="Zoom", command=zoom_image)
zoom_button.pack()

# Tạo nút nhấn để xoay ảnh
rotate_button = ttk.Button(root, text="Rotate", command=rotate_image)
rotate_button.pack()

# Khởi tạo biến tỷ lệ co dãn và góc xoay
scale_factor_x = 1.0
scale_factor_y = 1.0
rotation_angle = 0.0

update_images()
root.mainloop()
cv2.destroyAllWindows()