import os
import traceback

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from pre import get_base
from register_from_base import register_from_base
from evaluate_registration import evaluate
from video_demo import make_before_after_video


class PipelineWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, images_dir: str, results_dir: str):
        super().__init__()
        self.images_dir = images_dir
        self.results_dir = results_dir

    def run(self):
        try:
            if not self.images_dir or not os.path.isdir(self.images_dir):
                raise FileNotFoundError(f"输入图片文件夹不存在: {self.images_dir}")
            if not self.results_dir:
                raise ValueError("导出目录不能为空")
            os.makedirs(self.results_dir, exist_ok=True)

            self.progress.emit(5)
            json_path, filtered_dir, ref, n = get_base(
                self.images_dir, 
                results_folder=self.results_dir,
                remove_specular=False           # 禁用光斑去除
            )
            if not ref:
                raise RuntimeError("没有有效图像，无法继续配准。")

            self.progress.emit(35)

            def on_progress(done: int, total: int):
                if total <= 0:
                    self.progress.emit(100)
                    return
                # 35% -> 100% 线性映射
                pct = 35 + int((done / total) * 65)
                self.progress.emit(min(100, max(35, pct)))

            out_dir = register_from_base(results_dir=self.results_dir, on_progress=on_progress)
            
            # 运行预处理脚本，生成filtered_predictor_preprocessed文件夹
            self.progress.emit(95)
            import subprocess
            subprocess.run(["python", "preprocess_filtered.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
            
            self.progress.emit(100)
            self.finished.emit(out_dir)
        except Exception:
            msg = traceback.format_exc()
            self.failed.emit(msg)


def start_pipeline_in_thread(images_dir: str, results_dir: str):
    """
    创建 QThread + worker 的小工具函数，返回 (thread, worker) 给界面持有引用。
    """
    thread = QThread()
    worker = PipelineWorker(images_dir=images_dir, results_dir=results_dir)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker


class EvaluateWorker(QObject):
    progress = pyqtSignal(int)
    eval_row = pyqtSignal(dict)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, results_dir: str):
        super().__init__()
        self.results_dir = results_dir

    def run(self):
        try:
            self.progress.emit(5)
            out_csv, rows, _ = evaluate(
                self.results_dir,
                predictor_filtered_subdir="filtered_predictor_preprocessed",
                threshold=50  # 可以根据需要调整阈值
            )
            self.progress.emit(80)
            for r in rows:
                self.eval_row.emit(r)
            self.progress.emit(100)
            self.finished.emit(out_csv)
        except Exception:
            self.failed.emit(traceback.format_exc())


def start_evaluate_in_thread(results_dir: str):
    thread = QThread()
    worker = EvaluateWorker(results_dir=results_dir)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker


class VideoWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, results_dir: str, out_path: str, fps: int = 10):
        super().__init__()
        self.results_dir = results_dir
        self.out_path = out_path
        self.fps = fps

    def run(self):
        try:
            self.progress.emit(5)

            def on_progress(done: int, total: int):
                if total <= 0:
                    self.progress.emit(100)
                    return
                self.progress.emit(min(100, max(5, int((done / total) * 95) + 5)))

            out = make_before_after_video(
                self.results_dir, self.out_path, fps=self.fps, on_progress=on_progress
            )
            self.progress.emit(100)
            self.finished.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


def start_video_in_thread(results_dir: str, out_path: str, fps: int = 10):
    thread = QThread()
    worker = VideoWorker(results_dir=results_dir, out_path=out_path, fps=fps)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker

