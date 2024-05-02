import cv2

def rotate_image(image, angle):
    """Xoay hình ảnh một góc cho trước.

    Args:
        image (numpy.ndarray): Hình ảnh để xoay.
        angle (float): Góc xoay (đơn vị: độ).

    Returns:
        numpy.ndarray: Hình ảnh sau khi được xoay.
    """
    # Lấy kích thước của hình ảnh
    height, width = image.shape[:2]
    # Tính tâm của hình ảnh
    center = (width / 2, height / 2)
    # Tạo ma trận biến đổi affine để xoay hình ảnh
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Thực hiện phép biến đổi affine để xoay hình ảnh
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def crop_image(image_path, label_path, angle):
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
    corners = []
    for line in lines:
        # Tách thông tin tọa độ từ dòng
        label, x_center, y_center, box_width, box_height = map(float, line.split())
        # Tính toạ độ của các góc
        top_left_x = int((x_center - box_width / 2) * image.shape[1])
        top_left_y = int((y_center - box_height / 2) * image.shape[0])
        bottom_right_x = int((x_center + box_width / 2) * image.shape[1])
        bottom_right_y = int((y_center + box_height / 2) * image.shape[0])
        corners.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    if len(corners) == 3:
        # Lấy tọa độ của 3 điểm
        p1, p2, p3 = corners

        # Tính trung điểm của các cạnh nối các điểm
        p4_x = (p1[0] + p2[0] + p3[0]) // 3
        p4_y = (p1[1] + p2[1] + p3[1]) // 3

        # Tính tọa độ của điểm thứ 4
        p4 = (p4_x, p4_y, p2[2], p2[3])

        # Thêm điểm thứ 4 vào danh sách
        corners.append(p4)

    # Sắp xếp các góc theo thứ tự: top-left, top-right, bottom-left, bottom-right
    corners.sort()

    # Tìm góc trên cùng bên trái và góc dưới cùng bên phải của hình chữ nhật
    min_x = corners[0][0]
    min_y = corners[0][1]
    max_x = corners[-1][2]
    max_y = corners[-1][3]

    # Cắt hình ảnh theo vùng quan tâm
    cropped_image = image[min_y:max_y, min_x:max_x]
    
    # Xoay hình ảnh đã cắt
    rotated_cropped_image = rotate_image(cropped_image, angle)
    return rotated_cropped_image
