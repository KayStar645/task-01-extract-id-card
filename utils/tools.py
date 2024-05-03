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
        c1 = int((x_center - box_width / 2) * image.shape[1])
        c2 = int((y_center - box_height / 2) * image.shape[0])
        c3 = int((x_center + box_width / 2) * image.shape[1])
        c4 = int((y_center + box_height / 2) * image.shape[0])

        if label - 2 < 0:
            label = int(label) + 2
        else:
            label = int(label) - 2

        corners[label] = (c1, c2, c3, c4)

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

    top_left = (min(corners[0][0], corners[0][2]), min(corners[0][1], corners[0][3]))
    bottom_right = (max(corners[3][0], corners[3][2]), max(corners[3][1], corners[3][3]))
    
    # Xoay hình ảnh
    image = rotate_image_to_align_vectors(image, corners, top_left, bottom_right)

    return image

def get_corners(image):
    height, width = image.shape[:2]

    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)

    return top_left, top_right, bottom_left, bottom_right

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
    # Thực hiện xoay ảnh mà không thay đổi kích thước
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    return rotated_image

def custom_rotate_image(image, angle_deg):
    # Lấy kích thước của ảnh
    (h, w) = image.shape[:2]
    # Tính tâm của ảnh
    center = (w // 2, h // 2)
    # Tạo ma trận quay
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # Tính kích thước của ảnh đã xoay
    cos_theta = abs(M[0, 0])
    sin_theta = abs(M[0, 1])
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))
    # Điều chỉnh ma trận quay để giữ lại toàn bộ ảnh sau khi xoay
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    # Thực hiện xoay ảnh
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated_image

def rotate_image_to_align_vectors(image, corners, cornerA, cornerD):
    # Tìm góc trên cùng bên trái và góc dưới cùng bên phải của hình chữ nhật
    min_x = min([corner[0] for corner in corners])
    min_y = min([corner[1] for corner in corners])
    max_x = max([corner[2] for corner in corners])
    max_y = max([corner[3] for corner in corners])

    # Lấy hình ảnh cắt
    image = image[min_y:max_y, min_x:max_x]

    corner1, c2, c3, corner4 = get_corners(image)

    # Tính toán vector 14 và vector ad
    vector_14 = np.array([corner4[0] - corner1[0], corner4[1] - corner1[1]])
    vector_ad = np.array([cornerD[0] - cornerA[0], cornerD[1] - cornerA[1]])

    # Tính toán góc xoay giữa vector 14 và vector ad
    angle_rad = math.atan2(vector_ad[1],
                           vector_ad[0]) - math.atan2(vector_14[1],
                                                      vector_14[0])
    angle_deg = math.degrees(angle_rad)

    # Cộng thêm nửa góc vuông: Độ lệch từ góc chéo tới cạnh hình cn
    # angle_deg += 45
    
    image = custom_rotate_image(image, angle_deg)

    return image


