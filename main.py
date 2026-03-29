import sys
import os
import time
import hashlib
import requests
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
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

    def check_online_phishtank(self, url: str) -> bool:
        """在线PhishTank查询（必须执行）"""
        try:
            response = requests.post(
                "https://checkurl.phishtank.com/checkurl/",
                data={'url': url, 'format': 'json'},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('results', {}).get('in_database', False):
                    print(f"[在线黑名单] PhishTank命中: {url}")
                    return True
        except Exception as e:
            print(f"[在线黑名单查询失败]: {e}")
        return False

    def is_blacklisted(self, url: str) -> dict:
        """黑名单检查：先本地，后在线（两者都必须执行）"""
        url_lower = url.lower()

        if url_lower in self.blacklist:
            print(f"[本地黑名单] 命中: {url}")
            return {"is_blacklisted": True, "source": "本地黑名单"}

        if self.check_online_phishtank(url):
            return {"is_blacklisted": True, "source": "PhishTank在线黑名单"}

        return {"is_blacklisted": False, "source": None}

    def add_to_blacklist(self, url):
        lower_url = url.lower()
        if lower_url not in self.blacklist:
            self.blacklist.add(lower_url)
            self.save()
            print(f"[黑名单] 已自动添加: {url}")


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

    # ====================== 1. URL模态特征提取 ======================
    def url_feature_score(self, url: str) -> dict:
        """1. URL模态：提取URL词法和结构特征"""
        score = 0.0
        reasons = []
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if len(domain) > 35 or domain.count('.') > 4:
            score += 0.4
            reasons.append("域名过长或子域名过多")
        if any(kw in domain for kw in ['login', 'bank', 'pay', 'secure', 'verify', 'account', 'update']):
            score += 0.35
            reasons.append("包含可疑关键词")
        if domain.replace('.', '').isdigit():
            score += 0.45
            reasons.append("使用IP地址代替域名")

        return {"score": min(1.0, score), "reasons": reasons}

    # ====================== 2. HTML模态特征提取 ======================
    def html_feature_score(self, html: str) -> dict:
        """2. HTML模态：提取页面源码中的结构和行为特征"""
        if not html:
            return {"score": 0.0, "reasons": ["无法获取HTML"]}
        soup = BeautifulSoup(html, 'html.parser')
        score = 0.0
        reasons = []

        if len(soup.find_all('form')) >= 1:
            score += 0.25
            reasons.append(f"存在表单 ({len(soup.find_all('form'))}个)")
        if len(soup.find_all('input', {'type': 'password'})) >= 1:
            score += 0.3
            reasons.append(f"存在密码输入框")
        if len(soup.find_all('script')) > 20:
            score += 0.15
            reasons.append("脚本数量异常")

        title = soup.title.string if soup.title else ""
        if any(kw in title.lower() for kw in ['login', 'verify', 'account', 'bank', 'secure']):
            score += 0.2
            reasons.append("标题包含可疑关键词")

        return {"score": min(1.0, score), "reasons": reasons}

    # ====================== 3. 视觉模态特征提取 (YOLOv8) ======================
    def visual_feature_score(self, screenshot_path: str) -> float:
        """3. 视觉模态：使用YOLOv8对网页截图进行分类判断"""
        if not self.model:
            return 0.0
        try:
            results = self.model.predict(screenshot_path, conf=0.5, verbose=False)
            print(f"【视觉预测结果】: {results[0].probs.data.tolist()}")
            probs = results[0].probs.data.tolist()
            return max(probs) if probs else 0.0
        except Exception as e:
            print("视觉检测异常:", e)
            return 0.0

    def detect(self, url: str):
        try:
            options = Options()
            # options.add_argument('--headless')   # 已恢复弹出浏览器窗口
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36')

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(15)
            driver.get(url)
            time.sleep(6)

            screenshot_path = f"screenshots/{hashlib.md5(url.encode()).hexdigest()}.png"
            os.makedirs("screenshots", exist_ok=True)
            driver.save_screenshot(screenshot_path)

            html = driver.page_source
            driver.quit()

            url_result = self.url_feature_score(url)
            html_result = self.html_feature_score(html)
            visual_score = self.visual_feature_score(screenshot_path)

            final_score = (url_result["score"] * 0.3) + (html_result["score"] * 0.3) + (visual_score * 0.4)
            is_phishing = final_score > 0.65

            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)

            return {
                "is_phishing": is_phishing,
                "final_score": final_score,
                "url_result": url_result,
                "html_result": html_result,
                "visual_score": visual_score
            }

        except Exception as e:
            print("检测异常:", str(e))
            return {"is_phishing": False, "final_score": 0.0, "error": str(e)}


# ====================== 主界面 ======================
class PhishingDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv8的多模态钓鱼网站检测系统 v1.0")
        self.resize(1300, 900)
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

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("font-size: 16px; line-height: 1.8; padding: 10px;")
        layout.addWidget(self.result_area)

        central.setLayout(layout)

    def start_detection(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入URL")
            return
        if not url.startswith("http"):
            url = "https://" + url

        self.result_area.setText("检测中，请稍候...")
        QApplication.processEvents()

        # 黑名单检查
        blacklist_result = self.blacklist_manager.is_blacklisted(url)
        if blacklist_result["is_blacklisted"]:
            self.show_danger(blacklist_result["source"], url)
            return

        # 判断网站是否可以连接（弹出浏览器）
        can_connect = self.check_connectivity(url)
        if not can_connect:
            self.show_invalid_website(url)
            return

        # 多模态检测
        result = self.detector.detect(url)
        self.display_detailed_result(result, url)

    def check_connectivity(self, url: str) -> bool:
        """判断网站是否可以连接（弹出浏览器）"""
        try:
            options = Options()
            # options.add_argument('--headless')   # 不使用headless，弹出浏览器
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36')

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(15)
            driver.get(url)
            time.sleep(6)  # 等待页面加载
            driver.quit()
            return True
        except Exception as e:
            print("连接测试异常:", str(e))
            return False

    def show_invalid_website(self, url):
        self.setStyleSheet("background-color: #ff9800;")  # 橙色警告
        text = f"<h2 style='color:orange'>【⚠️ 无效网站】</h2>"
        text += f"<p><b>网址：</b>{url}</p>"
        text += "<p>无法连接该网站，可能不存在或已被屏蔽。</p>"
        self.result_area.setHtml(text)

    def show_danger(self, source, url):
        self.setStyleSheet("background-color: #ff5252;")
        text = f"<h2 style='color:red'>【⚠️ 危险！钓鱼网站】</h2>"
        text += f"<p>来源：{source}</p>"
        text += f"<p>网址：{url}</p>"
        self.result_area.setHtml(text)

    def display_detailed_result(self, result, url):
        if result.get("is_phishing"):
            color = "red"
            title = "【⚠️ 危险！钓鱼网站】"
            self.setStyleSheet("background-color: #ff5252;")
        else:
            color = "green"
            title = "【✅ 安全网站】"
            self.setStyleSheet("background-color: #4caf50;")

        text = f"<h2 style='color:{color}'>{title}</h2>"
        text += f"<p><b>网址：</b>{url}</p>"
        text += f"<p><b>最终综合分数：</b>{result['final_score']:.1%}</p><hr>"

        # ====================== 黑名单查询结果 ======================
        text += "<h3>黑名单查询结果</h3>"
        text += f"<p>本地黑名单：{'命中' if url in self.blacklist_manager.blacklist else '未命中'}</p>"
        text += f"<p>在线黑名单：已查询（未命中）</p><hr>"

        # ====================== 1. URL模态 ======================
        text += "<h3>1. URL特征分析</h3>"
        text += f"<p>可疑分数：{result['url_result']['score']:.1%}</p>"
        if result['url_result']['reasons']:
            text += "<p>原因：" + "、".join(result['url_result']['reasons']) + "</p>"

        # ====================== 2. HTML模态 ======================
        text += "<h3>2. HTML特征分析</h3>"
        text += f"<p>可疑分数：{result['html_result']['score']:.1%}</p>"
        if result['html_result']['reasons']:
            text += "<p>原因：" + "、".join(result['html_result']['reasons']) + "</p>"

        # ====================== 3. 视觉模态 ======================
        text += "<h3>3. 视觉特征分析 (YOLOv8)</h3>"
        text += f"<p>可疑分数：{result['visual_score']:.1%}</p>"

        self.result_area.setHtml(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhishingDetectorApp()
    window.show()
    sys.exit(app.exec_())