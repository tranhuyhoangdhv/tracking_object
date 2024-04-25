import cv2
import mediapipe as mp

def detect_thumbs_gestures():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Khởi tạo model Mediapipe Hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9)

    # Mở camera
    cap = cv2.VideoCapture(0)  # Số 0 có nghĩa là sử dụng camera mặc định

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi ảnh sang định dạng RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Nhận diện các điểm landmarks của tay
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Xác định bàn tay phải hoặc trái
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                    if handedness == "Left":
                        # Vẽ các landmarks trên tay
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Lấy tọa độ của ngón tay cái (thumbs) và ngón trỏ
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        # Kiểm tra nếu ngón tay cái nằm bên trái và cao hơn ngón trỏ
                        if thumb.x < index_finger.x and thumb.y < index_finger.y:
                            cv2.putText(frame, 'Thumbs Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Hiển thị video từ camera với các landmarks và cử chỉ thumbs up hoặc thumbs down
        cv2.imshow('Hand Gesture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm để nhận diện bàn tay thumbs up hoặc thumbs down từ camera
detect_thumbs_gestures()