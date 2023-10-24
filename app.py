import tkinter as tk
from tkinter import ttk, filedialog
import cv2

def zoom_image():
    global scale_factor_x, scale_factor_y
    try:
        scale_factor_x = float(scale_factor_x_entry.get())
        scale_factor_y = float(scale_factor_y_entry.get())
        update_images()
    except ValueError:
        # Xử lý ngoại lệ nếu người dùng nhập không phải là số
        pass


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
def apply_edge_detection():
    global original_image, threshold1, threshold2
    if original_image is not None:
        if not threshold1_entry.get() or not threshold2_entry.get():
            return

        threshold1 = int(threshold1_entry.get())
        threshold2 = int(threshold2_entry.get())

        edges = cv2.Canny(original_image, threshold1, threshold2)
        update_images(original_image, edges)

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

# Create buttons to apply edge detection, zoom image, and rotate image
apply_edge_detection_button = ttk.Button(root, text="Apply Edge Detection", command=apply_edge_detection)
apply_edge_detection_button.pack()

# Create a button to normalize images
normalize_images_button = ttk.Button(root, text="Normalize Images", command=normalize_images)
normalize_images_button.pack()

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

# Chuyển ảnh màu sang ảnh mức xám
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Image Normalization")

# Tạo nút nhấn để chuẩn hóa ảnh
normalize_button = ttk.Button(root, text="Normalize Images", command=normalize_images)
normalize_button.pack()

root.mainloop()
cv2.destroyAllWindows()