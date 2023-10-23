import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

def apply_edge_detection():
    global original_image, threshold1, threshold2
    if original_image is not None:
        if not threshold1_entry.get() or not threshold2_entry.get():
            return

        threshold1 = int(threshold1_entry.get())
        threshold2 = int(threshold2_entry.get())

        edges = cv2.Canny(original_image, threshold1, threshold2)
        update_images(original_image, edges)

def zoom_image():
    global original_image, scale_factor_x, scale_factor_y
    if original_image is not None:
        if not scale_factor_x_entry.get() or not scale_factor_y_entry.get():
            return

        scale_factor_x = float(scale_factor_x_entry.get())
        scale_factor_y = float(scale_factor_y_entry.get())

        zoomed_img = cv2.resize(original_image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        update_images(zoomed_img, zoomed_img)

def rotate_image():
    global original_image, rotation_angle
    if original_image is not None:
        if not rotation_angle_entry.get():
            return

        rotation_angle = float(rotation_angle_entry.get())
        rotation_matrix = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), rotation_angle, 1)
        rotated_img = cv2.warpAffine(original_image, rotation_matrix, (original_image.shape[1], original_image.shape[0]))
        update_images(rotated_img, rotated_img)

def normalize_images():
    global rgb_image, gray_image
    # Chuẩn hóa ảnh màu RGB
    normalized_rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Chuẩn hóa ảnh mức xám
    normalized_gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Hiển thị ảnh gốc và ảnh sau chuẩn hóa
    cv2.imshow('Original RGB Image', rgb_image)
    cv2.imshow('Normalized RGB Image', normalized_rgb_image)
    cv2.imshow('Original Gray Image', gray_image)
    cv2.imshow('Normalized Gray Image', normalized_gray_image)

def open_image():
    global original_image, rgb_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")])
    if file_path:
        original_image = cv2.imread(file_path)
        rgb_image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        update_images(original_image, original_image)

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
threshold1_label.pack(
