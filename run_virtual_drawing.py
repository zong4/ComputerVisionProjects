import cv2
import time
from hand_gesture import HandGesture
from virtual_drawing import VirtualDrawing

# 初始化手势检测和绘图工具
hand_gesture = HandGesture(
    max_num_hands=2,  # 最大检测手数
    min_detection_confidence=0.8,  # 手势检测的最小置信度
    min_tracking_confidence=0.5  # 手势跟踪的最小置信度
)

drawing_tool = VirtualDrawing(
    screen_width=1920,  # 屏幕宽度
    screen_height=1080,  # 屏幕高度
    default_brush_radius=5,  # 默认画笔半径
    default_eraser_radius=20  # 默认橡皮擦半径
)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 上一次右手食指的位置
last_right_index_pos = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 处理帧并获取双手数据
    frame, left_hand, right_hand = hand_gesture.process_frame(frame)

    # 如果检测到左手
    if left_hand:
        # 获取食指和拇指的坐标
        index_tip = left_hand[8]  # 食指指尖
        thumb_tip = left_hand[4]  # 拇指指尖
        middle_tip = left_hand[12]  # 中指指尖

        # 计算拇指和食指之间的距离
        distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

        # 计算拇指和中指之间的距离
        middle_distance = ((thumb_tip[0] - middle_tip[0]) ** 2 + (thumb_tip[1] - middle_tip[1]) ** 2) ** 0.5

        # 根据距离调整画笔或橡皮擦的粗细
        if drawing_tool.eraser_mode:
            drawing_tool.set_eraser_radius(max(10, min(50, distance)))  # 橡皮擦半径范围：10~50
        else:
            drawing_tool.set_brush_radius(max(5, min(20, distance)))  # 画笔半径范围：5~20

        # 如果食指和中指接触，清空画布
        if middle_distance < 30:
            drawing_tool.clear_canvas()
            cv2.waitKey(300) # 防止误操作

    # 如果检测到右手
    if right_hand:
        # 获取关键点坐标
        thumb_tip = right_hand[4]  # 拇指指尖
        index_tip = right_hand[8]  # 食指指尖
        middle_tip = right_hand[12]  # 中指指尖

        # 计算拇指和食指之间的距离
        thumb_index_distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

        # 计算拇指和中指之间的距离
        thumb_middle_distance = ((thumb_tip[0] - middle_tip[0]) ** 2 + (thumb_tip[1] - middle_tip[1]) ** 2) ** 0.5

        # 计算食指的运动速度
        if last_right_index_pos:
            speed = ((last_right_index_pos[0] - index_tip[0]) ** 2 + (last_right_index_pos[1] - index_tip[1]) ** 2) ** 0.5
        else:
            speed = 0

        # 当速度过快时，防止误操作
        if speed <= 20:
            # 如果拇指和食指接触，切换橡皮擦模式
            if thumb_index_distance < 30:
                drawing_tool.toggle_eraser_mode()
                cv2.waitKey(300)  # 防止快速切换

            # 如果拇指和中指接触，切换颜色
            if thumb_middle_distance < 30:
                drawing_tool.next_color()
                cv2.waitKey(300)  # 防止快速切换

        # 如果食指移动，画线
        if last_right_index_pos:
            drawing_tool.draw_line(last_right_index_pos, index_tip)
        last_right_index_pos = index_tip

    # 获取显示帧
    display_frame = drawing_tool.get_display_frame(frame)

    # 显示模式提示
    mode_text = "Eraser Mode" if drawing_tool.eraser_mode else "Drawing Mode"
    cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示颜色提示
    color_text = f"Color: {drawing_tool.color}"
    cv2.putText(display_frame, color_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示摄像头画面和绘图结果
    cv2.imshow("Virtual Drawing", display_frame)

    # 键盘操作
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 保存画布
        drawing_tool.save_canvas("drawing_{}.png".format(int(time.time())))
        print("画布已保存为图片")
    elif key == ord('q'):  # 退出程序
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()