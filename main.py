# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/3/6 20:43
  @version V1.0
"""
import os
import random
import sys
import threading
import time

import cv2
import numpy
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageQt
from yolo import YOLO


# 添加一个关于界面
# 窗口主类

class DetectFlag:
	PREDICT = 0
	VIDEO = 1
	DIR_PREDICT = 2
	CAMERA = 3


class MainWindow(QTabWidget):

	# 基本配置不动，然后只动第三个界面
	def __init__(self):
		# 初始化界面
		super().__init__()
		self.yolo = YOLO()
		# 用来设置检测模式
		self.model: int = DetectFlag.PREDICT
		# 保存图片路径
		self.img2predict = ""
		# 保存视频路径
		self.video_path = "./img/10s.mp4"
		# 保存输入文件夹路径
		self.dir_origin_path = "img/"
		# 保存输出文件夹路径
		self.dir_save_path = "img_out/"

		self.setWindowTitle('Yolov3检测系统')
		self.resize(1200, 800)
		self.setWindowIcon(QIcon("./UI/xf.jpg"))
		# 图片读取进程
		self.output_size = 480
		# 检测视频的线程
		self.threading = None
		# 是否跳出当前循环的线程
		self.jump_threading: bool = False

		self.initUI()
		self.reset_vid()

	def reset_vid(self):
		"""
		界面重置事件
		"""
		self.webcam_detection_btn.setEnabled(True)
		self.mp4_detection_btn.setEnabled(True)
		self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
		self.disable_btn(self.det_img_button)
		self.disable_btn(self.vid_start_stop_btn)
		self.jump_threading = False
		self.disable_btn(self.det_dir_button)

	def initUI(self):
		"""
		界面初始化
		"""
		# 图片检测子界面
		font_title = QFont('楷体', 16)
		font_main = QFont('楷体', 14)
		font_general = QFont('楷体', 10)
		# 图片识别界面, 两个按钮，上传图片和显示结果
		img_detection_widget = QWidget()
		img_detection_layout = QVBoxLayout()
		img_detection_title = QLabel("图片识别功能")
		img_detection_title.setFont(font_title)
		mid_img_widget = QWidget()
		mid_img_layout = QHBoxLayout()
		self.left_img = QLabel()
		self.right_img = QLabel()
		self.left_img.setPixmap(QPixmap("./UI/up.jpeg"))
		self.right_img.setPixmap(QPixmap("./UI/right.jpeg"))
		self.left_img.setAlignment(Qt.AlignCenter)
		self.right_img.setAlignment(Qt.AlignCenter)
		self.left_img.setMinimumSize(480, 480)
		self.left_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_img_layout.addWidget(self.left_img)
		self.right_img.setMinimumSize(480, 480)
		self.right_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_img_layout.addStretch(0)
		mid_img_layout.addWidget(self.right_img)
		mid_img_widget.setLayout(mid_img_layout)
		self.up_img_button = QPushButton("上传图片")
		self.det_img_button = QPushButton("开始检测")
		self.up_img_button.clicked.connect(self.upload_img)
		self.det_img_button.clicked.connect(self.detect_img)
		self.up_img_button.setFont(font_main)
		self.det_img_button.setFont(font_main)
		self.up_img_button.setStyleSheet("QPushButton{color:white}"
		                                 "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                 "QPushButton{background-color:rgb(48,124,208)}"
		                                 "QPushButton{border:2px}"
		                                 "QPushButton{border-radius:5px}"
		                                 "QPushButton{padding:5px 5px}"
		                                 "QPushButton{margin:5px 5px}")
		self.det_img_button.setStyleSheet("QPushButton{color:white}"
		                                  "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                  "QPushButton{background-color:rgb(48,124,208)}"
		                                  "QPushButton{border:2px}"
		                                  "QPushButton{border-radius:5px}"
		                                  "QPushButton{padding:5px 5px}"
		                                  "QPushButton{margin:5px 5px}")
		img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
		img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
		img_detection_layout.addWidget(self.up_img_button)
		img_detection_layout.addWidget(self.det_img_button)
		img_detection_widget.setLayout(img_detection_layout)

		# 视频识别界面
		# 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
		vid_detection_widget = QWidget()
		vid_detection_layout = QVBoxLayout()
		vid_title = QLabel("视频检测功能")
		vid_title.setFont(font_title)
		self.left_vid_img = QLabel()
		self.right_vid_img = QLabel()
		self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
		self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))
		self.left_vid_img.setAlignment(Qt.AlignCenter)
		self.left_vid_img.setMinimumSize(480, 480)
		self.left_vid_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		self.right_vid_img.setAlignment(Qt.AlignCenter)
		self.right_vid_img.setMinimumSize(480, 480)
		self.right_vid_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_img_widget = QWidget()
		mid_img_layout = QHBoxLayout()
		mid_img_layout.addWidget(self.left_vid_img)
		mid_img_layout.addStretch(0)
		mid_img_layout.addWidget(self.right_vid_img)
		mid_img_widget.setLayout(mid_img_layout)
		self.webcam_detection_btn = QPushButton("摄像头实时监测")
		self.mp4_detection_btn = QPushButton("视频文件检测")
		self.vid_start_stop_btn = QPushButton("启动/停止检测")
		self.webcam_detection_btn.setFont(font_main)
		self.mp4_detection_btn.setFont(font_main)
		self.vid_start_stop_btn.setFont(font_main)
		self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
		                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                        "QPushButton{background-color:rgb(48,124,208)}"
		                                        "QPushButton{border:2px}"
		                                        "QPushButton{border-radius:5px}"
		                                        "QPushButton{padding:5px 5px}"
		                                        "QPushButton{margin:5px 5px}")
		self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
		                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                     "QPushButton{background-color:rgb(48,124,208)}"
		                                     "QPushButton{border:2px}"
		                                     "QPushButton{border-radius:5px}"
		                                     "QPushButton{padding:5px 5px}"
		                                     "QPushButton{margin:5px 5px}")
		self.vid_start_stop_btn.setStyleSheet("QPushButton{color:white}"
		                                      "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                      "QPushButton{background-color:rgb(48,124,208)}"
		                                      "QPushButton{border:2px}"
		                                      "QPushButton{border-radius:5px}"
		                                      "QPushButton{padding:5px 5px}"
		                                      "QPushButton{margin:5px 5px}")
		self.webcam_detection_btn.clicked.connect(self.open_cam)
		self.mp4_detection_btn.clicked.connect(self.open_mp4)
		self.vid_start_stop_btn.clicked.connect(self.start_or_stop)

		# 添加fps显示
		fps_container = QWidget()
		fps_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
		fps_container_layout = QHBoxLayout()
		fps_container.setLayout(fps_container_layout)
		# 左容器
		fps_left_container = QWidget()
		fps_left_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
		fps_left_container_layout = QHBoxLayout()
		fps_left_container.setLayout(fps_left_container_layout)

		# 右容器
		fps_right_container = QWidget()
		fps_right_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
		fps_right_container_layout = QHBoxLayout()
		fps_right_container.setLayout(fps_right_container_layout)

		# 将左容器和右容器添加到fps_container_layout中
		fps_container_layout.addWidget(fps_left_container)
		fps_container_layout.addStretch(0)
		fps_container_layout.addWidget(fps_right_container)

		# 左容器中添加fps显示
		raw_fps_label = QLabel("原始帧率:")
		raw_fps_label.setFont(font_general)
		raw_fps_label.setAlignment(Qt.AlignLeft)
		raw_fps_label.setStyleSheet("QLabel{margin-left:80px}")
		self.raw_fps_value = QLabel("0")
		self.raw_fps_value.setFont(font_general)
		self.raw_fps_value.setAlignment(Qt.AlignLeft)
		fps_left_container_layout.addWidget(raw_fps_label)
		fps_left_container_layout.addWidget(self.raw_fps_value)

		# 右容器中添加fps显示
		detect_fps_label = QLabel("检测帧率:")
		detect_fps_label.setFont(font_general)
		detect_fps_label.setAlignment(Qt.AlignRight)
		self.detect_fps_value = QLabel("0")
		self.detect_fps_value.setFont(font_general)
		self.detect_fps_value.setAlignment(Qt.AlignRight)
		self.detect_fps_value.setStyleSheet("QLabel{margin-right:96px}")
		fps_right_container_layout.addWidget(detect_fps_label)
		fps_right_container_layout.addWidget(self.detect_fps_value)

		# 添加组件到布局上
		vid_detection_layout.addWidget(vid_title, alignment=Qt.AlignCenter)
		vid_detection_layout.addWidget(fps_container)
		vid_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
		vid_detection_layout.addWidget(self.webcam_detection_btn)
		vid_detection_layout.addWidget(self.mp4_detection_btn)
		vid_detection_layout.addWidget(self.vid_start_stop_btn)
		vid_detection_widget.setLayout(vid_detection_layout)

		# 文件夹识别界面, 两个按钮，上传图片和显示结果
		dir_detection_widget = QWidget()
		dir_detection_layout = QVBoxLayout()
		dir_detection_title = QLabel("文件夹识别功能")
		dir_detection_title.setFont(font_title)
		mid_dir_widget = QWidget()
		mid_dir_layout = QHBoxLayout()
		self.left_dir = QLabel()
		self.right_dir = QLabel()
		self.left_dir.setPixmap(QPixmap("./UI/up.jpeg"))
		self.right_dir.setPixmap(QPixmap("./UI/right.jpeg"))
		self.left_dir.setAlignment(Qt.AlignCenter)
		self.right_dir.setAlignment(Qt.AlignCenter)
		self.left_dir.setMinimumSize(480, 480)
		self.left_dir.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_dir_layout.addWidget(self.left_dir)
		self.right_dir.setMinimumSize(480, 480)
		self.right_dir.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_dir_layout.addStretch(0)
		mid_dir_layout.addWidget(self.right_dir)
		mid_dir_widget.setLayout(mid_dir_layout)
		self.open_dir_button = QPushButton("选择文件夹")
		self.open_dir_button.clicked.connect(self.select_dir)
		self.det_dir_button = QPushButton("开始检测")
		self.det_dir_button.clicked.connect(self.detect_dir)
		self.open_dir_button.setFont(font_main)
		self.det_dir_button.setFont(font_main)
		self.open_dir_button.setStyleSheet("QPushButton{color:white}"
		                                   "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                   "QPushButton{background-color:rgb(48,124,208)}"
		                                   "QPushButton{border:2px}"
		                                   "QPushButton{border-radius:5px}"
		                                   "QPushButton{padding:5px 5px}"
		                                   "QPushButton{margin:5px 5px}")
		self.det_dir_button.setStyleSheet("QPushButton{color:white}"
		                                  "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                  "QPushButton{background-color:rgb(48,124,208)}"
		                                  "QPushButton{border:2px}"
		                                  "QPushButton{border-radius:5px}"
		                                  "QPushButton{padding:5px 5px}"
		                                  "QPushButton{margin:5px 5px}")
		dir_detection_layout.addWidget(dir_detection_title, alignment=Qt.AlignCenter)
		dir_detection_layout.addWidget(mid_dir_widget, alignment=Qt.AlignCenter)
		dir_detection_layout.addWidget(self.open_dir_button)
		dir_detection_layout.addWidget(self.det_dir_button)
		dir_detection_widget.setLayout(dir_detection_layout)

		# 关于界面
		about_widget = QWidget()
		about_layout = QVBoxLayout()
		about_title = QLabel('欢迎使用目标检测系统\n\n 可以进行知识交流')  # 修改欢迎词语
		about_title.setFont(QFont('楷体', 18))
		about_title.setAlignment(Qt.AlignCenter)
		about_img = QLabel()
		about_img.setPixmap(QPixmap('./UI/qq.png'))
		about_img.setAlignment(Qt.AlignCenter)

		# label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
		label_super = QLabel()  # 更换作者信息
		label_super.setText("<a href='https://github.com/mpj1234?tab=repositories'>或者你可以在这里找到我-->mpj123</a>")
		label_super.setFont(QFont('楷体', 16))
		label_super.setOpenExternalLinks(True)
		# label_super.setOpenExternalLinks(True)
		label_super.setAlignment(Qt.AlignRight)
		about_layout.addWidget(about_title)
		about_layout.addStretch()
		about_layout.addWidget(about_img)
		about_layout.addStretch()
		about_layout.addWidget(label_super)
		about_widget.setLayout(about_layout)

		self.addTab(img_detection_widget, '图片检测')
		self.addTab(vid_detection_widget, '视频检测')
		self.addTab(dir_detection_widget, '文件夹检测')
		self.addTab(about_widget, '联系我')
		self.setTabIcon(0, QIcon('./UI/lufei.png'))
		self.setTabIcon(1, QIcon('./UI/lufei.png'))
		self.setTabIcon(2, QIcon('./UI/lufei.png'))

	def disable_btn(self, pushButton: QPushButton):
		pushButton.setDisabled(True)
		pushButton.setStyleSheet("QPushButton{background-color: rgb(2,110,180);}")

	def enable_btn(self, pushButton: QPushButton):
		pushButton.setEnabled(True)
		pushButton.setStyleSheet(
			"QPushButton{background-color: rgb(48,124,208);}"
			"QPushButton{color:white}"
		)

	def detect(self):
		if self.model == DetectFlag.PREDICT:
			try:
				image = Image.open(self.img2predict)
				l_image = image.copy()
			except:
				print('Open Error! Try again!')
			else:
				self.label_show_image(l_image, self.left_img)
				r_image = self.yolo.detect_image(image)
				self.label_show_image(r_image, self.right_img)
		elif self.model == DetectFlag.VIDEO or self.model == DetectFlag.CAMERA:
			capture = cv2.VideoCapture(self.video_path)
			ref, frame = capture.read()
			if not ref:
				raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
			fps = 0.0
			while not self.jump_threading:
				t1 = time.time()
				# 读取某一帧
				ref, frame = capture.read()
				if not ref:
					break
				# 格式转变，BGRtoRGB
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				# 转变成Image
				frame = Image.fromarray(np.uint8(frame))
				self.label_show_image(frame, self.left_vid_img)
				# 进行检测
				frame = self.yolo.detect_image(frame)
				self.label_show_image(frame, self.right_vid_img)

				fps = (fps + (1. / (time.time() - t1))) / 2
				self.detect_fps_value.setText(str(round(fps, 1)))
		elif self.model == DetectFlag.DIR_PREDICT:
			for img_name in self.img_names:
				if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
					image_path = os.path.join(self.dir_origin_path, img_name)
					img = Image.open(image_path)
					self.label_show_image(img, self.left_dir)
					r_image = self.yolo.detect_image(img)
					self.label_show_image(r_image, self.right_dir)
					if not os.path.exists(self.dir_save_path):
						os.makedirs(self.dir_save_path)
					r_image.save(os.path.join(self.dir_save_path, img_name.replace(".jpg", ".png")), quality=95,
					             subsampling=0)
		else:
			raise AssertionError(
				"Please specify the correct mode: 'predict', 'video', 'dir_predict'.")

	def resize_img(self, img):
		"""
		调整图片大小，方便用来显示

		:param img: 需要调整的图片
		"""
		resize_scale = min(self.output_size / img.size[0], self.output_size / img.size[1])
		img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)), Image.ANTIALIAS)
		return img

	def label_show_image(self, img, label: QLabel):
		"""
		在label中展示图片

		:param img: 图片
		:param label: label
		"""
		# 应该调整一下图片的大小
		img = self.resize_img(img)
		label.setPixmap(ImageQt.toqpixmap(img))

	def upload_img(self):
		"""
		上传图片
		"""
		# 选择录像文件进行读取
		fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg *.bmp')
		if fileName:
			self.img2predict = fileName
			# 将上传照片和进行检测做成互斥的
			self.enable_btn(self.det_img_button)
			self.disable_btn(self.up_img_button)
			# 进行左侧原图展示
			img = Image.open(fileName)
			self.label_show_image(img, self.left_img)
			# 上传图片之后右侧的图片重置
			self.right_img.setPixmap(QPixmap("./UI/right.jpeg"))

	def detect_img(self):
		"""
		检测图片
		"""
		# 重置跳出线程状态，防止其他位置使用的影响
		self.jump_threading = False
		self.model = DetectFlag.PREDICT
		# 将上传照片和进行检测做成互斥的
		self.enable_btn(self.up_img_button)
		self.disable_btn(self.det_img_button)
		self.detect()

	def open_mp4(self):
		"""
		开启视频文件检测事件
		"""
		fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
		if fileName:
			self.disable_btn(self.webcam_detection_btn)
			self.disable_btn(self.mp4_detection_btn)
			self.enable_btn(self.vid_start_stop_btn)
			# 生成读取视频对象
			cap = cv2.VideoCapture(fileName)
			# 获取视频的帧率
			fps = cap.get(cv2.CAP_PROP_FPS)
			# 显示原始视频帧率
			self.raw_fps_value.setText(str(fps))
			if cap.isOpened():
				# 读取一帧用来提前左侧展示
				ret, raw_img = cap.read()
				cap.release()
			else:
				QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
				self.disable_btn(self.vid_start_stop_btn)
				self.enable_btn(self.webcam_detection_btn)
				self.enable_btn(self.mp4_detection_btn)
				return

			img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(np.uint8(img))
			self.label_show_image(img, self.left_vid_img)
			# 上传图片之后右侧的图片重置
			self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))
			self.video_path = fileName
			self.jump_threading = False
			self.model = DetectFlag.VIDEO

	def open_cam(self):
		"""
		打开摄像头事件
		"""
		self.disable_btn(self.webcam_detection_btn)
		self.disable_btn(self.mp4_detection_btn)
		self.enable_btn(self.vid_start_stop_btn)
		self.video_path = 0
		self.jump_threading = False
		# 生成读取视频对象
		cap = cv2.VideoCapture(0)
		# 获取视频的帧率
		fps = cap.get(cv2.CAP_PROP_FPS)
		# 显示原始视频帧率
		self.raw_fps_value.setText(str(fps))
		if cap.isOpened():
			# 读取一帧用来提前左侧展示
			ret, raw_img = cap.read()
			cap.release()
		else:
			QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
			self.disable_btn(self.vid_start_stop_btn)
			self.enable_btn(self.webcam_detection_btn)
			self.enable_btn(self.mp4_detection_btn)
			return

		self.model = DetectFlag.CAMERA

		img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(np.uint8(img))
		self.label_show_image(img, self.left_vid_img)
		# 上传图片之后右侧的图片重置
		self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))

	def start_or_stop(self):
		"""
		启动或者停止事件
		"""
		if self.threading is None:
			# 创造并启动一个检测视频线程
			self.jump_threading = False
			self.threading = threading.Thread(target=self.detect_vid)
			self.threading.start()
			# 视频检测的
			self.disable_btn(self.webcam_detection_btn)
			self.disable_btn(self.mp4_detection_btn)
			# 文件夹检测的
			self.disable_btn(self.open_dir_button)
			self.enable_btn(self.det_dir_button)
		else:
			# 停止当前线程
			# 线程属性置空，恢复状态
			self.threading = None
			self.jump_threading = True
			# 视频检测恢复按键状态
			self.enable_btn(self.webcam_detection_btn)
			self.enable_btn(self.mp4_detection_btn)
			# 文件夹检测恢复按键状态
			self.disable_btn(self.det_dir_button)
			self.enable_btn(self.open_dir_button)

	def detect_vid(self):
		"""
		视频检测
		视频和摄像头的主函数是一样的，不过是传入的source不同罢了
		"""
		self.detect()
		# 执行完进程，刷新一下和进程有关的状态，只有self.threading是None，
		# 才能说明是正常结束的线程，需要被刷新状态
		if self.threading is not None:
			self.start_or_stop()

	def select_dir(self):
		detect_dir = QFileDialog.getExistingDirectory(self, 'Open Directory', "./",
		                                              QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
		if detect_dir:
			self.dir_origin_path = detect_dir
			self.model = DetectFlag.DIR_PREDICT

			self.img_names = os.listdir(detect_dir)
			for img_name in self.img_names:
				if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
					image_path = os.path.join(self.dir_origin_path, img_name)
					img = Image.open(image_path)
					self.label_show_image(img, self.left_dir)

					self.disable_btn(self.open_dir_button)
					self.enable_btn(self.det_dir_button)
					break

	def detect_dir(self):
		"""
		检测文件夹
		"""
		self.start_or_stop()

	def closeEvent(self, event):
		"""
		界面关闭事件
		"""
		reply = QMessageBox.question(
			self,
			'quit',
			"Are you sure?",
			QMessageBox.Yes | QMessageBox.No,
			QMessageBox.No
		)
		if reply == QMessageBox.Yes:
			self.jump_threading = True
			self.close()
			event.accept()
		else:
			event.ignore()


if __name__ == "__main__":
	app = QApplication(sys.argv)
	mainWindow = MainWindow()
	mainWindow.show()
	sys.exit(app.exec_())
