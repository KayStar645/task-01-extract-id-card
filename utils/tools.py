import cv2
import numpy as np
import math

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

    # Lấy hình ảnh cắt
    #image = image[min(corners[0]):max(corners[3]), min(corners[0]):max(corners[3])]

    top_left = (min(corners[0]), min(corners[0]))
    top_right = (max(corners[1]), min(corners[1]))
    bottom_left = (min(corners[2]), max(corners[2]))
    bottom_right = (max(corners[3]), max(corners[3]))
    
    # Xoay hình ảnh
    c1, c2, c3, c4 = get_corners(image)
    # Kiểm tra xem TopLeft là label mấy, từ label truy ra c mấy
    image = rotate_image_to_match_corner(image, c1, None, top_left, None)
    #image = rotate_image(image, c1, c2, top_left)

    return image

def rotate_image(image_old, top_left_old, top_right_old, top_left_new):
    
    C_rad = find_angle(top_left_old, top_right_old, top_left_new)

    rotated_image = rotate_image_with_angle(image_old, C_rad, False)

    return rotated_image

def rotate_image_to_match_corner(image, corner_A1, corner_A4, corner_G1, corner_G4):
    # Tính góc giữa các điểm A1, A4 và G1, G4
    angle_to_rotate = find_angle_2(corner_A1, corner_A4, corner_G1) - find_angle_2(corner_A1, corner_A4, corner_G4)
    
    # Xoay hình ảnh với góc tính được
    rotated_image = rotate_image_with_angle(image, angle_to_rotate)
    
    return rotated_image

def get_corners(image):
    height, width = image.shape[:2]

    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)

    return top_left, top_right, bottom_left, bottom_right

def find_angle(a, b, c):
    # Tính độ dài các đoạn thẳng
    ab = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    ac = math.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    bc = math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
    
    # Tính giá trị của cos(C)
    cos_C = (ab**2 + ac**2 - bc**2) / (2 * ab * ac)
    
    # Tính góc C bằng arccosine
    C_rad = math.acos(cos_C)
    
    # Chuyển đổi radian thành độ
    C_deg = math.degrees(C_rad)
    
    return C_deg

def find_angle_2(a, b, c):
    # Tính vectơ AB và BC
    AB = [b[0] - a[0], b[1] - a[1]]
    BC = [c[0] - b[0], c[1] - b[1]]
    
    # Tính độ dài của vectơ AB và BC
    length_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
    length_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    
    # Tính tích vô hướng của AB và BC
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    
    # Tính cosin của góc giữa AB và BC
    cos_angle = dot_product / (length_AB * length_BC)
    
    # Tính góc (radians)
    angle_rad = math.acos(cos_angle)
    
    # Chuyển từ radians sang độ
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def rotate_image_with_angle(image, angle, clockwise=True):
    # Tính kích thước của ảnh
    (h, w) = image.shape[:2]
    # Tính tâm của ảnh
    center = (w // 2, h // 2)
    # Đảo ngược góc nếu cần thiết
    if clockwise:
        angle = 360 - angle
    # Xây dựng ma trận biến đổi affine
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Thực hiện xoay ảnh
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image
