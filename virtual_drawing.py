import cv2
import numpy as np

class VirtualDrawing:
    def __init__(self, screen_width=1920, screen_height=1080, default_brush_radius=5, default_eraser_radius=20):
        """
        初始化绘图工具

        :param screen_width: 屏幕宽度
        :param screen_height: 屏幕高度
        :param default_brush_radius: 默认画笔半径
        :param default_eraser_radius: 默认橡皮擦半径
        """
        # 屏幕分辨率
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 初始化画布
        self.canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # 绘图参数
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # 颜色列表（绿色、红色、蓝色）
        self.color_index = 0  # 当前颜色索引
        self.color = self.colors[self.color_index]  # 当前颜色

        # 画笔和橡皮擦参数
        self.brush_radius = default_brush_radius
        self.eraser_radius = default_eraser_radius
        self.eraser_mode = True  # 是否处于橡皮擦模式

        # 历史记录（用于撤销）
        self.history = []

    def clear_canvas(self):
        """清空画布"""
        self.canvas.fill(0)  # 将画布填充为黑色
        self.history = []  # 清空历史记录

    def draw_line(self, start_point, end_point):
        """在画布上画线"""
        if self.eraser_mode:
            cv2.line(self.canvas, start_point, end_point, (0, 0, 0), self.eraser_radius)  # 橡皮擦
        else:
            cv2.line(self.canvas, start_point, end_point, self.color, self.brush_radius)  # 画笔
        self.history.append(self.canvas.copy())  # 保存当前画布状态到历史记录

    def undo(self):
        """撤销上一步操作"""
        if len(self.history) > 1:
            self.history.pop()  # 移除当前状态
            self.canvas = self.history[-1].copy()  # 恢复到上一个状态

    def toggle_eraser_mode(self):
        """切换橡皮擦模式"""
        self.eraser_mode = not self.eraser_mode

    def next_color(self):
        """切换到下一个颜色"""
        self.color_index = (self.color_index + 1) % len(self.colors)
        self.color = self.colors[self.color_index]

    def set_brush_radius(self, radius):
        """设置画笔半径"""
        self.brush_radius = int(radius)

    def set_eraser_radius(self, radius):
        """设置橡皮擦半径"""
        self.eraser_radius = int(radius)

    def save_canvas(self, filename):
        """保存画布为图像文件"""
        cv2.imwrite(filename, self.canvas)

    def get_display_frame(self, frame):
        """获取显示帧（画布 + 摄像头画面）"""
        # 将画布和摄像头画面叠加
        display_frame = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
        return display_frame