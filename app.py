import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np


def zoom_image():
    global scale_factor_x, scale_factor_y
    try:
        if original_image is not None:
            if not scale_factor_x_entry.get() or not scale_factor_y_entry.get():
                return

            scale_factor_x = float(scale_factor_x_entry.get())
            scale_factor_y = float(scale_factor_y_entry.get())
            update_images()
    except ValueError:
        # Xử lý ngoại lệ nếu người dùng nhập không phải là số
        pass

        zoomed_img = cv2.resize(original_image, None, fx=scale_factor_x, fy=scale_factor_y,
                                interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Zoom Image", zoomed_img)


def normalize_images():
    global rgb_image, gray_image
    # Chuẩn hóa ảnh màu RGB
    normalized_rgb_image = cv2.normalize(
        rgb_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Chuẩn hóa ảnh mức xám
    normalized_gray_image = cv2.normalize(
        gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Normalized RGB Image', normalized_rgb_image)
    cv2.imshow('Normalized Gray Image', normalized_gray_image)


def apply_edge_detection():
    global original_image, threshold1, threshold2
    if original_image is not None:
        if not threshold1_entry.get() or not threshold2_entry.get():
            return

        threshold1 = int(threshold1_entry.get())
        threshold2 = int(threshold2_entry.get())

        edges = cv2.Canny(original_image, threshold1, threshold2)
        cv2.imshow("Edge detection Image", edges)


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
        cv2.imshow("Rotate Image", rotated_img)


def open_image():
    global original_image, rgb_image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")])
    if file_path:
        original_image = cv2.imread(file_path)
        rgb_image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        update_images(original_image, gray_image)


def update_images(image1, image2):
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)


def apply_blur():
    global original_image
    if original_image is not None:
        # Tạo một bản sao độc lập của original_image để không ảnh hưởng đến original_image
        blurred_image = original_image.copy()

        # Lấy vùng được chọn từ ảnh gốc
        x, y, w, h = cv2.selectROI(
            "Select ROI", original_image, fromCenter=False, showCrosshair=True)
        # Kiểm tra xem vùng được chọn có đủ lớn để áp dụng kernel Gaussian không
        if w > 0 and h > 0:
            roi = original_image[y:y+h, x:x+w]

            # Tạo một bản sao độc lập của vùng được chọn để không ảnh hưởng đến original_image
            roi_copy = roi.copy()

            # Áp dụng độ mờ lên vùng đã chọn
            blur_value = int(blur_slider.get())
            # Giảm kích thước của kernel nếu nó lớn hơn hoặc bằng kích thước nhỏ nhất của vùng chọn
            blur_value = min(blur_value, min(w, h) - 1)
            # Đảm bảo rằng blur_value luôn là số lẻ và lớn hơn 0
            blur_value = max(
                1, blur_value) if blur_value % 2 == 0 else blur_value
            # Tạo kernel với kích thước mới
            kernel = (blur_value, blur_value)
            blurred_roi = cv2.GaussianBlur(roi_copy, kernel, 0)

            # Gán vùng đã mờ trở lại ảnh gốc
            blurred_image[y:y+h, x:x+w] = blurred_roi

            # Hiển thị ảnh đã mờ
            cv2.imshow('Blurred Image', blurred_image)
# Hàm áp dụng filter2D với kernel được chọn dựa trên giá trị từ thanh trượt


def apply_filter():
    global original_image
    if original_image is not None:
        # Lấy giá trị từ thanh trượt để điều chỉnh kernel
        kernel_value = int(filter_slider.get())

        # Chọn kernel tương ứng dựa trên giá trị từ thanh trượt
        if kernel_value <= 4:
            kernel_sharpen = np.array(
                [[-1, -1, -1], [-1, kernel_value, -1], [-1, -1, -1]])
        elif kernel_value <= 7:
            kernel_sharpen = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
        else:
            kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                                       [-1, 2, 2, 2, -1],
                                       [-1, 2, kernel_value, 2, -1],
                                       [-1, 2, 2, 2, -1],
                                       [-1, -1, -1, -1, -1]]) / 8.0

        # Áp dụng filter2D với kernel được chọn
        output = cv2.filter2D(original_image, -1, kernel_sharpen)

        # Hiển thị ảnh đầu ra
        cv2.imshow('Filtered Image', output)


# Khởi tạo biến ảnh gốc và các biến liên quan
original_image = None
threshold1 = 100
threshold2 = 200
scale_factor_x = 1.0
scale_factor_y = 1.0
rotation_angle = 0.0
rgb_image = None
gray_image = None

# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Image Processing")

# Tạo nút nhấn để chọn ảnh
open_image_button = ttk.Button(root, text="Open Image", command=open_image)
open_image_button.pack()

# Create buttons to apply edge detection, zoom image, and rotate image
apply_edge_detection_button = ttk.Button(
    root, text="Apply Edge Detection", command=apply_edge_detection)
apply_edge_detection_button.pack()

# Create a button to normalize images
normalize_images_button = ttk.Button(
    root, text="Normalize Images", command=normalize_images)
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

# Tạo nút nhấn để lựa chọn vùng và điều chỉnh độ mờ
select_region_button = ttk.Button(
    root, text="Select Region and Apply Blur", command=apply_blur)
select_region_button.pack()

# Tạo thanh trượt để điều chỉnh độ mờ
blur_slider_label = ttk.Label(root, text="Blur Amount:")
blur_slider_label.pack()
blur_slider = ttk.Scale(root, from_=1, to=31, orient="horizontal", length=200)
blur_slider.pack()

# Tạo nút nhấn để áp dụng filter2D và thanh trượt để điều chỉnh giá trị kernel
apply_filter_button = ttk.Button(
    root, text="Apply Filter", command=apply_filter)
apply_filter_button.pack()

# Tạo thanh trượt để điều chỉnh giá trị của kernel
filter_slider_label = ttk.Label(root, text="Filter Value:")
filter_slider_label.pack()
filter_slider = ttk.Scale(root, from_=1, to=10,
                          orient="horizontal", length=200)
filter_slider.pack()

root.mainloop()
cv2.destroyAllWindows()
