import tkinter as tk
from tkinter import ttk, filedialog

import cv2

def zoom_image():
    global original_image, scale_factor_x, scale_factor_y
    if original_image is not None:
        if not scale_factor_x_entry.get() or not scale_factor_y_entry.get():
            return

        scale_factor_x = float(scale_factor_x_entry.get())
        scale_factor_y = float(scale_factor_y_entry.get())

        zoomed_img = cv2.resize(original_image, None, fx=scale_factor_x, fy=scale_factor_y,
                                interpolation=cv2.INTER_LINEAR)
        update_images(zoomed_img, zoomed_img)


def rotate_image():
    global original_image, rotation_angle
    if original_image is not None:
        if not rotation_angle_entry.get():
            return

        rotation_angle = float(rotation_angle_entry.get())
        rotation_matrix = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2),
                                                  rotation_angle, 1)
        rotated_img = cv2.warpAffine(original_image, rotation_matrix,
                                     (original_image.shape[1], original_image.shape[0]))
        update_images(rotated_img, rotated_img)

def open_image():
    global original_image, rgb_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")])
    if file_path:
        original_image = cv2.imread(file_path)
        rgb_image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        update_images(original_image, gray_image)


def update_images(image1, image2):
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)


# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Image Processing")

# Tạo nút nhấn để chọn ảnh
open_image_button = ttk.Button(root, text="Open Image", command=open_image)
open_image_button.pack()

# Tạo Entry Widgets cho hai ngưỡng
threshold1_label = ttk.Label(root, text="Threshold 1:")
threshold1_label.pack()
threshold1_entry = ttk.Entry(root)
threshold1_entry.pack()

threshold2_label = ttk.Label(root, text="Threshold 2:")
threshold2_label.pack()
threshold2_entry = ttk.Entry(root)
threshold2_entry.pack()

# Tạo Entry Widgets cho tỷ lệ x và y
scale_factor_x_label = ttk.Label(root, text="Scale Factor X:")
scale_factor_x_label.pack()
scale_factor_x_entry = ttk.Entry(root)
scale_factor_x_entry.pack()

scale_factor_y_label = ttk.Label(root, text="Scale Factor Y:")
scale_factor_y_label.pack()
scale_factor_y_entry = ttk.Entry(root)
scale_factor_y_entry.pack()

# Tạo nút nhấn để zoom ảnh
zoom_button = ttk.Button(root, text="Zoom Image", command=zoom_image)
zoom_button.pack()

# Tạo Entry Widgets cho góc xoay
rotation_angle_label = ttk.Label(root, text="Rotation Angle:")
rotation_angle_label.pack()
rotation_angle_entry = ttk.Entry(root)
rotation_angle_entry.pack()

# Tạo nút nhấn để xoay ảnh
rotate_button = ttk.Button(root, text="Rotate Image", command=rotate_image)
rotate_button.pack()

# Khởi tạo biến ảnh gốc và các biến liên quan
original_image = None
threshold1 = 100
threshold2 = 200
scale_factor_x = 1.0
scale_factor_y = 1.0
rotation_angle = 0.0
rgb_image = None
gray_image = None

root.mainloop()
cv2.destroyAllWindows()
