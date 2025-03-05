import mediapipe as mp
import cv2

class HandGesture:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        初始化手势检测器

        :param max_num_hands: 最大检测手数
        :param min_detection_confidence: 手势检测的最小置信度
        :param min_tracking_confidence: 手势跟踪的最小置信度
        """
        # 初始化 MediaPipe Hands 模块
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """处理摄像头帧，检测手势并返回双手的关键点数据"""
        # 水平翻转图像
        frame = cv2.flip(frame, 1)

        # 将图像转换为 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理图像并检测手势
        results = self.hands.process(image_rgb)

        # 初始化双手数据
        left_hand = None
        right_hand = None

        # 如果检测到手
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 获取手的类型（左手或右手）
                hand_type = handedness.classification[0].label

                # 获取关键点坐标（忽略不可见的点）
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # if landmark.visibility < 0.5:  # 忽略不可见的点
                    #     continue
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))

                # 根据手的类型存储数据
                if hand_type == "Left":
                    left_hand = landmarks
                elif hand_type == "Right":
                    right_hand = landmarks

                # 绘制手势关键点
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, left_hand, right_hand