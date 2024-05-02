import cv2
import numpy as np

imageFormats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.txt', 'JPG']
def crop_image(image_path, label_path):
    # Đọc hình ảnh từ đường dẫn
    image = cv2.imread(image_path)

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            print(f"Error: Not enough points to crop image {image_path}")
            return None
    except FileNotFoundError:
        print(f"Error: Label file '{label_path}' not found")
        return None

    # Tính toạ độ của các góc của hình chữ nhật
    corners = [None, None, None, None]
    for line in lines:
        # Tách thông tin tọa độ từ dòng
        label, x_center, y_center, box_width, box_height = map(float, line.split())
        # Tính toạ độ của các góc
        top_left = int((x_center - box_width / 2) * image.shape[1])
        bottom_left = int((y_center - box_height / 2) * image.shape[0])
        top_right = int((x_center + box_width / 2) * image.shape[1])
        bottom_right = int((y_center + box_height / 2) * image.shape[0])

        if label - 2 < 0:
            label = int(label) + 2
        else:
            label = int(label) - 2

        corners[label] = (top_left, bottom_left, top_right, bottom_right)

    if corners.count(None) == 1:
        # Lấy tọa độ của 3 điểm
        p1, p2, p3 = [corner for corner in corners if corner is not None]

        # Tính trung điểm của các cạnh nối các điểm
        p4_x = (p1[0] + p2[0] + p3[0]) // 3
        p4_y = (p1[1] + p2[1] + p3[1]) // 3

        # Tìm vị trí của phần tử None trong danh sách corners
        none_index = corners.index(None)

        # Tính tọa độ của điểm thứ 4
        p4 = (p4_x, p4_y, p2[2], p2[3])

        # Thêm điểm thứ 4 vào danh sách
        corners[none_index] = p4

    import cv2
import numpy as np

imageFormats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.txt', 'JPG']
def crop_image(image_path, label_path):
    # Đọc hình ảnh từ đường dẫn
    image = cv2.imread(image_path)

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            print(f"Error: Not enough points to crop image {image_path}")
            return None
    except FileNotFoundError:
        print(f"Error: Label file '{label_path}' not found")
        return None

    # Tính toạ độ của các góc của hình chữ nhật
    corners = [None, None, None, None]
    for line in lines:
        # Tách thông tin tọa độ từ dòng
        label, x_center, y_center, box_width, box_height = map(float, line.split())
        # Tính toạ độ của các góc
        top_left = int((x_center - box_width / 2) * image.shape[1])
        bottom_left = int((y_center - box_height / 2) * image.shape[0])
        top_right = int((x_center + box_width / 2) * image.shape[1])
        bottom_right = int((y_center + box_height / 2) * image.shape[0])

        if label - 2 < 0:
            label = int(label) + 2
        else:
            label = int(label) - 2

        corners[label] = (top_left, bottom_left, top_right, bottom_right)

    if corners.count(None) == 1:
        # Lấy tọa độ của 3 điểm
        p1, p2, p3 = [corner for corner in corners if corner is not None]

        # Tính trung điểm của các cạnh nối các điểm
        p4_x = (p1[0] + p2[0] + p3[0]) // 3
        p4_y = (p1[1] + p2[1] + p3[1]) // 3

        # Tìm vị trí của phần tử None trong danh sách corners
        none_index = corners.index(None)

        # Tính tọa độ của điểm thứ 4
        p4 = (p4_x, p4_y, p2[2], p2[3])

        # Thêm điểm thứ 4 vào danh sách
        corners[none_index] = p4

    top_left = (min(corners[0]), min(corners[0]))
    top_right = (max(corners[1]), min(corners[1]))
    bottom_left = (min(corners[2]), max(corners[2]))
    bottom_right = (max(corners[3]), max(corners[3]))

    # Lấy hình ảnh cắt
    image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Xoay hình ảnh
    image = rotate_image(image, top_left, top_right, bottom_left, bottom_right)
    return image

def rotate_image(image, top_left, top_right, bottom_left, bottom_right):
    # Tính kích thước mới cho hình ảnh đích
    new_width = max(top_right[0], bottom_right[0]) - min(top_left[0], bottom_left[0])
    new_height = max(bottom_left[1], bottom_right[1]) - min(top_left[1], top_right[1])

    # Tạo ma trận biến đổi perspective
    src_pts = np.float32([top_left, top_right, bottom_left, bottom_right])
    dst_pts = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Thực hiện biến đổi hình ảnh
    rotated_image = cv2.warpPerspective(image, perspective_matrix, (new_width, new_height))

    return rotated_image

def rotate_image1(image, top_left, top_right, bottom_left, bottom_right):
    # Tạo ma trận biến đổi perspective
    src_pts = np.float32([top_left, top_right, bottom_left, bottom_right])
    dst_pts = np.float32([[0, 0], [image.shape[0], 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]])
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Thực hiện biến đổi hình ảnh
    rotated_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))

    return rotated_image