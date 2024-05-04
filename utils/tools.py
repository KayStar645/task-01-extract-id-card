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

        if len(lines) < 2:
            print(f"Error: Not enough points to crop image {image_path}")
            return image
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

        if label == 0:
            corners[2] = (c1,c4)
        if label == 1:
            corners[3] = (c3, c4)
        if label == 2:
            corners[0] = (c1, c2)
        if label == 3:
            corners[1] = (c3, c2)

    none_index = -1
    if corners.count(None) > 2:
        return image
    elif corners.count(None) == 1:
        # Lấy tọa độ của 3 điểm
        p1, p2, p3 = [corner for corner in corners if corner is not None]

        # Tìm vị trí của phần tử None trong danh sách corners
        none_index = corners.index(None)

        # Thêm điểm thứ 4 vào danh sách
        corners[none_index] = find_fourth_point(p1, p2, p3)

    image_with_circles = draw_circles(image, corners)
    cv2.imwrite('C:/Users/DELL/Desktop/thuanpt/6.Solution/task-01-extract-id-card/yolov9/runs/detect/corner/temp2.jpg',
                    image_with_circles)
    # Xoay hình ảnh
    image = rotate_image_to_align_vectors(image, corners)

    return image

def get_corners(image):
    height, width = image.shape[:2]

    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)

    return top_left, top_right, bottom_left, bottom_right

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

def rotate_image_to_align_vectors(image, corners):
    # Lấy hình ảnh cắt
    if corners.count(None) == 0:
        # Tìm góc trên cùng bên trái và góc dưới cùng bên phải của hình chữ nhật
        min_x = min(int(corner[0]) for corner in corners)  # Tọa độ x của góc trên cùng bên trái
        min_y = min(int(corner[1]) for corner in corners)  # Tọa độ y của góc trên cùng bên trái
        max_x = max(int(corner[0]) for corner in corners)  # Tọa độ x của góc dưới cùng bên phải
        max_y = max(int(corner[1]) for corner in corners)  # Tọa độ y của góc dưới cùng bên phải

        image = image[min_y:max_y, min_x:max_x]

    cv2.imwrite('C:/Users/DELL/Desktop/thuanpt/6.Solution/task-01-extract-id-card/yolov9/runs/detect/corner/temp.jpg',
                image)

    c1, c2, c3, c4 = get_corners(image)
    
    verters = []
    corners_temp = []
    for index, corner in enumerate(corners):
        if corner != None:
            verters.append(corner)
            if index == 0:
                corners_temp.append(c1)
            elif index == 1:
                corners_temp.append(c2)
            elif index == 2:
                corners_temp.append(c3)
            elif index == 3:
                corners_temp.append(c4) 
    vector_12 = np.array([corners_temp[1][0] - corners_temp[0][0],
                        corners_temp[1][1] - corners_temp[0][1]])
    vector_ab = np.array([verters[1][0] - verters[0][0],
                        verters[1][1] - verters[0][1]])
        
    # Tính toán góc xoay giữa vector 14 và vector ad
    angle_rad = math.atan2(vector_ab[1],
                           vector_ab[0]) - math.atan2(vector_12[1],
                                                      vector_12[0])
    angle_deg = math.degrees(angle_rad)
    
    image = custom_rotate_image(image, angle_deg)

    return image

def distance(point1, point2):
    # Tính khoảng cách Euclid giữa hai điểm
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def midpoint(point1, point2):
    # Tính trung điểm của hai điểm
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def find_fourth_point(A, B, C):
    # Tính độ dài các cạnh
    AB = distance(A, B)
    BC = distance(B, C)
    CA = distance(C, A)

    # Xác định cạnh dài nhất
    max_edge = max(AB, BC, CA)

    # Tính trung điểm của cạnh dài nhất
    if max_edge == AB:
        center = midpoint(A, B)
        D_prime = find_symmetric_point(C, center)
    elif max_edge == BC:
        center = midpoint(B, C)
        D_prime = find_symmetric_point(A, center)
    else:
        center = midpoint(C, A)
        D_prime = find_symmetric_point(B, center)

    return D_prime

def find_symmetric_point(A, center):
    x_A_prime = 2 * center[0] - A[0]
    y_A_prime = 2 * center[1] - A[1]
    return (abs(int(x_A_prime)), abs(int(y_A_prime)))

def draw_circles(image, corners, radius=20, color=(0, 255, 0), thickness=-1):
    """
    Vẽ các chấm tròn lên hình ảnh dựa trên các tọa độ được cung cấp.

    Args:
        image (numpy.ndarray): Hình ảnh đầu vào.
        corners (list): Danh sách các tọa độ của các chấm. Mỗi tọa độ là một tuple (x, y).
        radius (int): Bán kính của chấm.
        color (tuple): Màu sắc của chấm, dưới dạng tuple (B, G, R).
        thickness (int): Độ dày của viền chấm. Nếu -1, thì sẽ là chấm đặc.

    Returns:
        numpy.ndarray: Hình ảnh với các chấm tròn được vẽ lên.
    """
    # Tạo một bản sao của hình ảnh để không làm thay đổi hình ảnh gốc
    output_image = image.copy()
    
    # Vẽ chấm tròn lên hình cho mỗi tọa độ
    for corner in corners:
        cv2.circle(output_image, (int(corner[0]), int(corner[1])), radius, color, thickness)
    
    return output_image
