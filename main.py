import sys
import os
import time
import hashlib
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyttsx3
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# ====================== 黑名单管理 ======================
class BlacklistManager:
    def __init__(self, file="blacklist.txt"):
        self.file = file
        self.blacklist = set()
        self.load()

    def load(self):
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                self.blacklist = {line.strip().lower() for line in f if line.strip()}

    def save(self):
        with open(self.file, 'w', encoding='utf-8') as f:
            for item in sorted(self.blacklist):
                f.write(item + '\n')

    def is_blacklisted(self, url):
        return url.lower() in self.blacklist

    def add_to_blacklist(self, url):
        lower_url = url.lower()
        if lower_url not in self.blacklist:
            self.blacklist.add(lower_url)
            self.save()
            print(f"[黑名单] 已添加: {url}")


# ====================== 多模态检测核心 ======================
class MultiModalDetector:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = YOLO("model/best.pt")
            print("✅ YOLOv8模型加载成功")
        except Exception as e:
            print("⚠️ YOLOv8模型加载失败:", e)

    def url_feature_score(self, url: str) -> float:
        score = 0.0
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if len(domain) > 35 or domain.count('.') > 4:
            score += 0.4
        if any(kw in domain for kw in ['login', 'bank', 'pay', 'secure', 'verify', 'account', 'update']):
            score += 0.35
        if domain.replace('.', '').isdigit():
            score += 0.45
        return min(1.0, score)

    def html_feature_score(self, html: str) -> float:
        if not html:
            return 0.0
        soup = BeautifulSoup(html, 'html.parser')
        score = 0.0
        if len(soup.find_all('form')) >= 2:
            score += 0.3
        if len(soup.find_all('input', {'type': 'password'})) >= 1:
            score += 0.25
        title = soup.title.string if soup.title else ""
        if any(kw in title.lower() for kw in ['login', 'verify', 'account', 'bank']):
            score += 0.2
        return min(1.0, score)

    def detect(self, url: str):
        try:
            options = Options()
            # options.add_argument('--headless')        # 调试阶段关闭headless
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--allow-running-insecure-content')
            options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36')

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(15)
            driver.get(url)
            time.sleep(5)

            screenshot_path = f"screenshots/{hashlib.md5(url.encode()).hexdigest()}.png"
            os.makedirs("screenshots", exist_ok=True)
            driver.save_screenshot(screenshot_path)

            html = driver.page_source
            driver.quit()

            url_score = self.url_feature_score(url)
            html_score = self.html_feature_score(html)
            visual_score = 0.0

            if self.model:
                results = self.model.predict(screenshot_path, conf=0.5, verbose=False)
                print(f"预测结果: {results[0].probs.data.tolist()}")
                probs = results[0].probs.data.tolist()
                visual_score = max(probs) if probs else 0.0

            final_score = (url_score * 0.25) + (html_score * 0.25) + (visual_score * 0.5)
            is_phishing = final_score > 0.6

            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)

            return {
                "is_phishing": is_phishing,
                "final_score": final_score,
                "url_score": url_score,
                "html_score": html_score,
                "visual_score": visual_score
            }

        except Exception as e:
            error_str = str(e).lower()
            print("检测异常:", error_str)

            # 关键修复：无法访问网站时判定为高风险
            if any(k in error_str for k in
                   ['name_not_resolved', 'connection', 'timeout', 'closed', 'unreachable', 'err_connection']):
                print("⚠️ 无法访问网站，判定为高风险（可能为钓鱼网站）")
                return {
                    "is_phishing": True,
                    "final_score": 0.85,
                    "url_score": 0.6,
                    "html_score": 0.0,
                    "visual_score": 0.0,
                    "error": "无法访问，可能为钓鱼或恶意网站"
                }

            return {"is_phishing": False, "final_score": 0.0, "error": str(e)}


# ====================== 主界面 ======================
class PhishingDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv8的多模态钓鱼网站检测系统 v1.0")
        self.resize(1100, 720)
        self.detector = MultiModalDetector()
        self.blacklist_manager = BlacklistManager()
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        title = QLabel("多模态钓鱼网站检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #1e88e5; margin: 20px;")
        layout.addWidget(title)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("请输入网址，例如：https://example.com")
        self.url_input.setStyleSheet("font-size: 18px; padding: 12px;")
        layout.addWidget(self.url_input)

        self.btn = QPushButton("开始检测")
        self.btn.setStyleSheet("font-size: 20px; padding: 15px; background: #1e88e5; color: white;")
        self.btn.clicked.connect(self.start_detection)
        layout.addWidget(self.btn)

        self.result_label = QLabel("等待检测...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; margin: 40px 0;")
        layout.addWidget(self.result_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(320)
        layout.addWidget(self.image_label)

        central.setLayout(layout)

    def speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass

    def start_detection(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入URL")
            return
        if not url.startswith("http"):
            url = "https://" + url

        self.result_label.setText("检测中，请稍候...")
        self.result_label.setStyleSheet("font-size: 24px; color: orange;")
        QApplication.processEvents()

        if self.blacklist_manager.is_blacklisted(url):
            self.show_danger("本地黑名单直接拦截", url)
            return

        result = self.detector.detect(url)

        if result.get("is_phishing"):
            self.show_danger("多模态融合检测", url, result["final_score"])
            self.blacklist_manager.add_to_blacklist(url)
        else:
            self.show_safe(result["final_score"])

    def show_danger(self, source, url, score=0.0):
        self.setStyleSheet("background-color: #ff5252;")
        text = f"【⚠️ 危险！钓鱼网站】\n来源：{source}\n网址：{url}"
        if score > 0:
            text += f"\n综合置信度：{score:.1%}"
        self.result_label.setText(text)
        self.speak("警告！这是钓鱼网站，请立即关闭！")

    def show_safe(self, score):
        self.setStyleSheet("background-color: #4caf50;")
        self.result_label.setText(f"【✅ 安全网站】\n综合置信度：{score:.1%}\n可以放心访问")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhishingDetectorApp()
    window.show()
    sys.exit(app.exec_())