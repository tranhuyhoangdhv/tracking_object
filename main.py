#TRẦN HUY HOÀNG
#K61 TỰ ĐỘNG HÓA
#TRƯỜNG ĐẠI HỌC VINH

import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import MotorModule as mM
import time
import mediapipe as mp
import torch
CONFIDENCE_THRESHOLD = 0.65
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
# initialize the video capture object
video_cap = cv2.VideoCapture(0)
device = torch.device("cuda:0")
print(torch.cuda.is_available())
# load the pre-trained YOLOv8n model
model = YOLO("yolov8n-pose.pt")
tracker = DeepSort(max_age=20)
# Variables for tracking a single person
tracked_person_id = None
class_id = 0
frames_since_last_detection = 0
max_frames_to_skip_detection = 20  # Number of frames
sensor_width_mm = 3.68  # Độ rộng của cảm biến trong mm (Ví dụ: 1/1.7 inch)
sensor_height_mm = 2.76  # Độ cao của cảm biến trong mm (Ví dụ: 1/1.7 inch)
object_real_width_cm = 50  # Chiều rộng thực tế của đối tượng trong cm
object_real_height_cm = 170  # Chiều cao thực tế của đối tượng trong cm
focal_length_mm = 25  # Tiêu cự của camera trong mm
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Khởi tạo model Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
# Mở camera
cap = cv2.VideoCapture(0)  # Số 0 có nghĩa là sử dụng camera mặc định
a=0
b=0
#các biến lưu dữ liệu
distance_meters=0
frame_count = 0
deviation = 0
distance_R = 0
distance_L = 0
speedL = 0
speedR = 0
speed_2 = 0
speed_1 = 0
speed_avg_d=0
speed_avg=0
Vx = 0
Vy = 0
vitri_truyendi = 0
vitri_nhanlai = 0
vitri_truyendi_list = []
vitri_nhanlai_list = []
deviation_list = []
distance_R_list= []
distance_L_list= []
speed_1_list = []
speed_2_list = []
distance_list = []
Vx_list = []
Vy_list = []
speed_d1_list = []
speed_d2_list = []
speed_avg_d_list=[]
speed_avg_list=[]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Chuyển đổi ảnh sang định dạng RGB Vì opencv thường đọc ra BGR
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Nhận diện các điểm landmarks của tay
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:# Xác định bàn tay phải hoặc trái
        for hand_landmarks in results.multi_hand_landmarks:
            # Xác định bàn tay phải hoặc trái
            if results.multi_handedness:#multi_hand_landmarks: Danh sách các bàn tay được phát hiện trong khung hình, multi_handedness: Thông tin về bàn tay phải hoặc trái.
                handedness = results.multi_handedness[0].classification[0].label # lấy thông tin bàn tay trái hoặc phải
                if handedness == "Left":#nếu phát hiện là tay trái thì thực hiện hàm
                    # Vẽ các landmarks trên tay
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Lấy tọa độ của ngón tay cái (thumbs) và ngón trỏ
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # Kiểm tra nếu ngón tay cái nằm bên trái và cao hơn ngón trỏ
                    if thumb.x < index_finger.x and thumb.y < index_finger.y:
                        cv2.putText(frame, 'Thumbs Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        a=1
                        break
                    if a==1:
                        while True:
                            deviation_list.append(deviation)
                            distance_R_list.append(distance_R)
                            distance_L_list.append(distance_L)
                            Vx_list.append(Vx)
                            distance_list.append(distance_meters)
                            speed_1_list.append(speed_1)
                            speed_2_list.append(speed_2)
                            speed_d1_list.append(speedL)
                            speed_d2_list.append(speedR)
                            speed_avg_list.append(speed_avg)
                            speed_avg_d_list.append(speed_avg_d)
                            vitri_truyendi_list.append(vitri_truyendi)
                            vitri_nhanlai_list.append(vitri_nhanlai)
                            start = datetime.datetime.now()# điểm thời gian hiện tại bắt đầu đo FPS
                            ret, frame = video_cap.read()# Đọc từng frame của camera
                            if not ret:# nếu không thành công thoát khỏi vòng lặp
                                break
                            height, width, _ = frame.shape # lấy ra chiều cao và chiều rộng của frame
                            # tinh đường trung tâm của trục x của frame
                            center_line = width // 2
                            # cv2.line(frame, (center_line, 50), (center_line, height), (255, 255, 255), 1)
                            # Đặt lại khung kể từ lần phát hiện cuối cùng nếu phát hiện xảy ra
                            if tracked_person_id is None:#Kiểm tra xem biến tracked_person_id có None hay không. Biến này lưu trữ ID của đối tượng mà hệ thống đang theo dõi.
                                frames_since_last_detection += 1 #là None, tức là không có đối tượng nào được theo dõi, hệ thống sẽ tăng giá trị của frames_since_last_detection
                            else:
                                frames_since_last_detection = 0 #tức là hệ thống đang theo dõi một đối tượng cụ thể, biến frames_since_last_detection được thiết lập lại về 0, bởi vì hệ thống đã phát hiện được một đối tượng và đã tiếp tục theo dõi nó. Việc đặt lại giá trị này đảm bảo rằng nó sẽ được đếm lại từ đầu sau mỗi lần phát hiện.
                            # chạy model yolov8 nhận dạng người
                            detections = model(frame, device=device)[0]
                            # khởi tạo danh sách các hộp giới hạn và tin cậy
                            results = []
                            # duyệt các phần tử  của dêtction
                            for data in detections.boxes.data.tolist():
                                # trích xuất độ tin cậy (tức là xác suất) liên quan đến dự đoán
                                confidence = data[4]
                                # lọc ra các phát hiện yếu bằng cách đảm bảo độ tin cậy lớn hơn độ tin cậy tối thiểu
                                if float(confidence) < CONFIDENCE_THRESHOLD:
                                    continue
                                # nếu độ tin cậy lớn hơn độ tin cậy tối thiểu
                                # lấy hộp giới hạn và id lớp
                                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                                # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, 1)
                                class_id = int(data[5])
                                # thêm hộp giới hạn(bbox) (x, y, w, h), độ tin cậy và id lớp vào danh sách kết quả
                                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
                                # Đặt lại frame_since_last_Detection và cập nhật tracked_person_id nếu phát hiện thấy một người
                                if class_id == 0:
                                    frames_since_last_detection = 0
                                    tracked_person_id = None
                            # TRACKING
                            if frames_since_last_detection <= max_frames_to_skip_detection:
                                # cập nhật trình theo dõi với những phát hiện mới
                                tracks = tracker.update_tracks(results, frame=frame)
                                # Chỉ theo dõi người đầu tiên được phát hiện
                                # xử lý một tình huống cụ thể trong đó class_id là 0 và
                                # tracked_person_id chưa được gán bất kỳ giá trị nào
                                if class_id == 0 and tracked_person_id is None:
                                    for track in tracks:# duyệt các phần tử track trong tracks
                                        if not track.is_confirmed():# Nếu track không được xác nhận
                                            continue # nó sẽ bỏ qua lần lặp tiếp theo của vòng lặp bằng câu lệnh continue
                                        tracked_person_id = track.track_id #gán ID của track đó (track_id) cho biến tracked_person_id
                                        break #kết thúc vòng lặp
                            # Nếu đang theo dõi một người, vẽ hộp giới hạn
                            if tracked_person_id is not None: #nếu tracked_person_id không phải là None tức là tracked person id có giá trị
                                for track in tracks: # duyệt qua từng track của tracks
                                    if not track.is_confirmed():# Nếu track không được xác nhận
                                        continue #tiếp tục
                                    # If the track ID matches our tracked person ID, draw their bounding box
                                    if track.track_id == tracked_person_id: # nếu như id nhận được từ track.track_id == tracked_person_id
                                        ltrb = track.to_ltrb() # lấy thông tin về vị trí Left, Top, Right, Bottom
                                        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]) #trích xuất giá trị từ ltrb
                                        object_width = xmax - xmin # tâm chiều rộng
                                        object_height = ymax - ymin #tâm chiều cao
                                        scale_width = object_real_width_cm / (object_width * sensor_width_mm / width) #tỷ lệ giữa kích thước thực của đối tượng
                                        scale_height = object_real_height_cm / (object_height * sensor_height_mm / height)# và kích thước của đối tượng trên hình ảnh
                                        avg_scale = (scale_width + scale_height) / 2 #tính toán trung bình của tỷ lệ chiều rộng và tỷ lệ chiều cao của đối tượng trong không gian hình ảnh
                                        # Calculate distance
                                        distance = focal_length_mm * avg_scale #khoảng cách mm
                                        distance_meters = (distance / 1000) + 0.3  # Chuyển đổi thành mét
                                        distance_str = "{:.2f}".format(distance_meters)
                                        # In ra kết quả khoảng cách
                                        print("Khoảng cách từ camera đến đối tượng:", distance_meters, "m")
                                        cv2.putText(frame, f'Distance: {distance_str} m', (300, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        # tính toán tâm của x,y bbox
                                        center_x = int((xmin + xmax) / 2)
                                        center_y = int((ymin + ymax) / 2)
                                        cv2.circle(frame, (center_x, center_y), 4, RED, -1)#lệnh vẽ hình tròn rọng 5 màu trắng
                                        # tính toán độ lệch so với tâm bbox và frameQq
                                        deviation = center_x - center_line
                                        if deviation > 150:#giới hạn
                                            deviation = 150
                                        if deviation < -150:
                                            deviation = -150
                                        print("độ lệch:", deviation)
                                        # vẽ bbox cho đối tượng
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                                        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
                                        cv2.putText(frame, str(tracked_person_id), (xmin + 5, ymin - 8),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                                        cv2.putText(frame, str(tracked_person_id), (xmin + 5, ymin - 8),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                                        cv2.putText(frame, f'tracking ID: {tracked_person_id}', (510, 460),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                    (0, 255, 0), 2)
                                        if distance_meters > 1.6:
                                            turn = deviation * 0.6
                                            speedL = 80 + (distance_meters * 11) + turn * 0.4
                                            speedR = 90 + (distance_meters * 11) - turn * 0.4
                                            vitri_truyendi = speedR -speedL
                                            speed_avg_d = (speedL+speedR)/2
                                            mM.move("TIEN", speedL, speedR)
                                            print("TIEN", speedL, speedR)
                                            distance_R, distance_L, speed_1, speed_2 = mM.distance()
                                            speed_avg = (speed_1+speed_2)/2
                                            vitri_nhanlai = speed_2 - speed_1
                                            if distance_R > 300:
                                                distance_R = 300
                                            if distance_L > 300:
                                                distance_L = 300
                                            # distance_R_list.append(distance_R)
                                            # distance_L_list.append(distance_L)
                                            # print(distance_R)
                                            # print(distance_L)
                                            # print(speed_1)
                                            # print(speed_2)
                                            dis_strR = "{:.2f}".format(distance_R)
                                            cv2.putText(frame, f'Dis R: {dis_strR} CM', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.5,
                                                        (0, 255, 0), 2)
                                            dis_strL = "{:.2f}".format(distance_L)
                                            cv2.putText(frame, f'Dis L: {dis_strL} CM', (500, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.5,
                                                        (0, 255, 0), 2)
                                            speed_1_str = "{:.2f}".format(speed_1)
                                            cv2.putText(frame, f'speed_1: {speed_1_str} CM', (10, 50),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                        (0, 255, 0), 2)
                                            speed_2_str = "{:.2f}".format(speed_2)
                                            cv2.putText(frame, f'speed_2: {speed_2_str} CM', (500, 50),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                        (0, 255, 0), 2)
                                            if distance_R < 40:
                                                # speedL = 150 - distance_R * 1
                                                # speedR = 150 - distance_R * 1
                                                speedL = 200
                                                speedR = 200
                                                # Vy_list.append(-speedR)
                                                mM.move("TRUOTTRAI", speedL, speedR)
                                                print("TRUOTTRAI", speedL, speedR)
                                                time.sleep(0.5)
                                            elif distance_L < 40:
                                                # speedL = 150 - distance_L * 1
                                                # speedR = 150 - distance_L * 1
                                                speedL = 200
                                                speedR = 200
                                                mM.move("TRUOTPHAI", speedL, speedR)
                                                print("TRUOTPHAI", speedL, speedR)
                                                time.sleep(0.5)
                                            elif distance_L < 40 and distance_R <40:
                                                if distance_R > distance_L:
                                                    speedL = 180
                                                    speedR = 180
                                                    mM.move("TRUOTPHAI", speedL, speedR)
                                                    print("TRUOTPHAI", speedL, speedR)
                                                    time.sleep(0.12)
                                                elif distance_R < distance_L:
                                                    speedL = 180
                                                    speedR = 180
                                                    mM.move("TRUOTPHAI", speedL, speedR)
                                                    print("TRUOTPHAI", speedL, speedR)
                                                    time.sleep(0.12)
                                                else:
                                                    speedL = 180
                                                    speedR = 180
                                                    mM.move("TRUOTPHAI", speedL, speedR)
                                                    print("TRUOTPHAI", speedL, speedR)
                                                    time.sleep(0.1)
                                        elif distance_meters < 1.3:
                                            turn = deviation * 0.5
                                            speedL = 50 + (distance_meters * 10) - turn * 0.4
                                            speedR = 50 + (distance_meters * 10) + turn * 0.4
                                            mM.move("LUI", speedL, speedR)
                                            print("LUI", speedL, speedR)
                                        elif 1.3 < (distance_meters) < 1.4:
                                            speedL = 0
                                            speedR = 0
                                        else:
                                            if (deviation) >= 0:
                                                speedL = deviation * 0.2
                                                speedR = deviation * 0.2
                                                if (speedL and speedR) > 40:
                                                    speedL = 40
                                                    speedR = 40
                                                mM.move("XOAYPHAI", speedL, speedR)
                                                print("XOAYPHAI", speedL, speedR)
                                            else:
                                                speedL = -deviation * 0.3
                                                speedR = -deviation * 0.3
                                                if (speedL and speedR) > 50:
                                                    speedL = 50
                                                    speedR = 50
                                                mM.move("XOAYTRAI", speedL, speedR)
                                                print("XOAYPHAI", speedL, speedR)
                                        # Vẽ đồ thị với trục x là thời gian
                            # thời gian kết thúc để tính toán khung hình/giây
                            end = datetime.datetime.now()
                            # hiển thị fps
                            print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
                            # calculate the frame per second and draw it on the frame
                            fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
                            cv2.putText(frame, fps, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            #tao file luu du lieu
                            # Save deviation_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/deviation_list.txt', 'w') as file:
                                for item in deviation_list:
                                    file.write("%s\n" % item)
                            # Save distance_R_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/distance_R_list.txt', 'w') as file:
                                for item in distance_R_list:
                                    file.write("%s\n" % item)
                            # Save distance_L_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/distance_L_list.txt', 'w') as file:
                                for item in distance_L_list:
                                    file.write("%s\n" % item)
                            # Save Vx_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/Vx_list.txt', 'w') as file:
                                for item in Vx_list:
                                    file.write("%s\n" % item)
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/speed_1_list.txt', 'w') as file:
                                for item in speed_1_list:
                                    file.write("%s\n" % item)
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/speed_2_list.txt', 'w') as file:
                                for item in speed_2_list:
                                    file.write("%s\n" % item)
                            # Save distance_R_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/speed_d1_list.txt', 'w') as file:
                                for item in speed_d1_list:
                                    file.write("%s\n" % item)
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/speed_d2_list.txt', 'w') as file:
                                for item in speed_d2_list:
                                    file.write("%s\n" % item)
                            # Save distance_R_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/distance_list.txt', 'w') as file:
                                for item in distance_list:
                                    file.write("%s\n" % item)
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/speed_avg_d_list.txt', 'w') as file:
                                for item in speed_avg_d_list:
                                    file.write("%s\n" % item)
                            # Save distance_R_list as text file
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/speed_avg_list.txt', 'w') as file:
                                for item in speed_avg_list:
                                    file.write("%s\n" % item)
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/vitri_truyendi_list.txt', 'w') as file:
                                for item in vitri_truyendi_list:
                                    file.write("%s\n" % item)
                            with open('F:/DAI HOC VINH/hoc_ky_8/do_an_3/yolo/v3/data_robot/vitri_nhanlai_list.txt', 'w') as file:
                                for item in vitri_nhanlai_list:
                                    file.write("%s\n" % item)
                            # show the frame to our screen
                            cv2.imshow("Frame", frame)
                            if cv2.waitKey(1) == ord("q"):
                                speedL = 0
                                speedR = 0
                                mM.move("DUNG", speedL, speedR)
                                break
    # Hiển thị video từ camera với các landmarks và cử chỉ thumbs up hoặc thumbs down
    cv2.imshow('Hand Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speedL = 0
        speedR = 0
        mM.move("DUNG", speedL, speedR)
        break
cap.release()
cv2.destroyAllWindows()

