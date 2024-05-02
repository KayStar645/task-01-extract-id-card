import os
import cv2
from utils.tools import *
import matplotlib.pyplot as plt
import shutil
import uuid
from yolov9 import detect
import pytesseract
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
# Trỏ pytesseract vào đường dẫn cài đặt Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Lấy đường dẫn đến thư mục hiện tại
prefix_path = os.getcwd().replace('\\', '/')

# Đường dẫn đến các tham số và các thư mục liên quan
weights_corner_path = prefix_path + "/yolov9/runs/train/yolov9-c-corner/weights/best.pt"
weights_info_path = prefix_path + "/yolov9/runs/train/yolov9-c-info/weights/best.pt"
detect_corner_path = prefix_path + '/yolov9/runs/detect/corner'
detect_info_path = prefix_path + '/yolov9/runs/detect/info'

image_path = 'C:/Users/DELL/Desktop/thuanpt/5.Data/000000093_1.jpg'

cropped_image_path = prefix_path + '/yolov9/runs/detect/' + str(uuid.uuid4()) + os.path.splitext(image_path)[1]

labels_path = prefix_path + '/yolov9/runs/detect/corner/labels/'

# Hình ảnh thông tin cắt
code_dir = prefix_path + '/yolov9/runs/detect/info/crops/code/'
name_dir = prefix_path + '/yolov9/runs/detect/info/crops/name/'

code = ''
name = ''

# Chọn hình ảnh
def button_select_images(root):
    label = tk.Label(root)
    label.pack(padx=10, pady=10)
    select_button = tk.Button(root, text="Chọn ảnh",
                              command=lambda: show_selected_image(label))
    select_button.pack(pady=10)

def show_selected_image(label):
    filename = filedialog.askopenfilename(initialdir="/", title="Chọn hình ảnh",
                                          filetypes=(("Image Files", "*.png;*.jpg;*.jpeg"),
                                                     ("All Files", "*.*")))
    if filename:
        global image_path
        image_path = filename
        image = Image.open(filename)
        image = image.resize((414, 414), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

root = tk.Tk()
root.title("Chọn ảnh")

# Tạo nhãn để hiển thị đường dẫn của hình ảnh
filename_label = tk.Label(root)
filename_label.pack(pady=10)

# Gọi hàm để tạo nút và hiển thị hình ảnh
button_select_images(root)

root.mainloop()

label_name = os.path.basename(image_path).split('.')[0] + '.txt'
print(image_path)

# Bước 1: Detect góc chứng minh thư
# Xây dựng lệnh để chạy
detect.run(weights=weights_corner_path, source=image_path, 
           save_txt=True, nosave=True, name='corner')
cropped_image = crop_image(image_path, f'{labels_path}{label_name}')

# Xóa thư mục detect_path corner
shutil.rmtree(detect_corner_path)

# Bước 2: Detect thông tin trên chứng minh thư
# Lưu chứng minh thư đã cắt góc lại
cv2.imwrite(cropped_image_path, cropped_image)

# Xây dựng lệnh để chạy
detect.run(weights=weights_info_path, source=cropped_image_path,
           nosave=True, save_crop=True, name='info')
# Xóa hình ảnh chứng minh thư sau khi xử lý
os.remove(cropped_image_path)

# Bước 3: Trích xuất thông tin trên đối tượng đã cắt
# Tiền xử lý thông tin trên đối tượng đã cắt
# Sao chép hình ảnh đã cắt
if os.path.exists(code_dir):
    for filename in os.listdir(code_dir):
        img = cv2.imread(os.path.join(code_dir, filename)) 
        # Chuyển sang màu xám
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Làm sáng ảnh
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)

        # Làm rõ ảnh
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)

        # Làm mờ ảnh
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Ngưỡng hình ảnh bằng ngưỡng thích ứng
        img = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 4)

        code += pytesseract.image_to_string(img)

if os.path.exists(name_dir):
    for filename in os.listdir(name_dir):
        img = cv2.imread(os.path.join(name_dir, filename))

        # Chuyển sang màu xám
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Làm sáng ảnh
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)

        # Làm rõ ảnh
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)

        # Làm mờ ảnh
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Ngưỡng hình ảnh bằng ngưỡng thích ứng
        img = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 4)

        name += pytesseract.image_to_string(img)

# Xóa thư mục detect_path info
shutil.rmtree(detect_info_path)


# Sử dụng ORC để trích xuất thông tin - Xem qua Mô hình nhận dạng văn bản end-to-end

print('code: '+ code)
print('name: '+ name)


# # Hiển thị ảnh ban đầu
# oldd_image = cv2.imread(image_path)
# plt.subplot(2, 2, 1)
# plt.imshow(oldd_image)
# plt.title('Ảnh trước khi cắt')
# plt.axis('off')

# # Hiển thị ảnh sau khi cắt
# plt.subplot(2, 2, 2)
# plt.imshow(cropped_image)
# plt.title('Ảnh sau khi cắt')
# plt.axis('off')
# plt.show()

