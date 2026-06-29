import os
import sys

def _ensure_qt_platform_plugin_path():
    """
    Windows + 非 ASCII 路径下，Qt 有时无法自动定位 platforms 插件。
    这里显式指定插件路径，避免报错：
      Could not find the Qt platform plugin "windows" in ""
    """
    try:
        import PyQt5  # noqa: WPS433

        pyqt5_dir = os.path.dirname(PyQt5.__file__)
        qt_plugins_dir = os.path.join(pyqt5_dir, "Qt5", "plugins")
        qt_platforms_dir = os.path.join(qt_plugins_dir, "platforms")

        if os.path.isdir(qt_platforms_dir):
            os.environ.setdefault("QT_PLUGIN_PATH", qt_plugins_dir)
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", qt_platforms_dir)
    except Exception:
        # 让后续正常抛出 Qt 自身错误信息
        pass


_ensure_qt_platform_plugin_path()

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui_worker import start_evaluate_in_thread, start_pipeline_in_thread, start_video_in_thread


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("眼底图像配准工具")
        self.setMinimumSize(980, 620)

        self._thread = None
        self._worker = None

        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setPlaceholderText("请选择输入图片文件夹（包含待处理图像）")

        self.results_dir_edit = QLineEdit()
        self.results_dir_edit.setPlaceholderText("请选择导出目录（将生成 results/filtered、frame_info.json、registered_superretina）")

        self.btn_pick_images = QPushButton("选择输入文件夹")
        self.btn_pick_results = QPushButton("选择导出目录")
        self.btn_run = QPushButton("一键生成基准配准")
        self.btn_eval = QPushButton("评价配准效果")
        self.btn_video = QPushButton("生成配准视频")
        self.btn_open_out = QPushButton("打开导出目录")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.eval_table = QTableWidget(0, 5)
        self.eval_table.setHorizontalHeaderLabels(
            [
                "frame",
                "ncc_before",
                "ncc_after",
                "dsc_before",
                "dsc_after",
            ]
        )
        self.eval_table.horizontalHeader().setStretchLastSection(True)
        self.eval_table.setSelectionBehavior(self.eval_table.SelectRows)
        self.eval_table.setEditTriggers(self.eval_table.NoEditTriggers)

        self._build_layout()
        self._wire()

        # 默认导出目录：项目内 results
        this_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir_edit.setText(os.path.join(this_dir, "results"))

    def _build_layout(self):
        io_group = QGroupBox("输入 / 输出")
        io_layout = QGridLayout()
        io_layout.setColumnStretch(1, 1)

        io_layout.addWidget(QLabel("输入图片文件夹"), 0, 0)
        io_layout.addWidget(self.images_dir_edit, 0, 1)
        io_layout.addWidget(self.btn_pick_images, 0, 2)

        io_layout.addWidget(QLabel("导出目录"), 1, 0)
        io_layout.addWidget(self.results_dir_edit, 1, 1)
        io_layout.addWidget(self.btn_pick_results, 1, 2)

        io_group.setLayout(io_layout)

        action_group = QGroupBox("运行")
        action_layout = QHBoxLayout()
        action_layout.addWidget(self.btn_run, 2)
        action_layout.addWidget(self.btn_eval, 1)
        action_layout.addWidget(self.btn_video, 1)
        action_layout.addWidget(self.btn_open_out, 1)
        action_group.setLayout(action_layout)

        eval_group = QGroupBox("评价结果")
        eval_layout = QVBoxLayout()
        eval_layout.addWidget(self.eval_table)
        eval_group.setLayout(eval_layout)

        root = QVBoxLayout()
        root.addWidget(io_group)
        root.addWidget(action_group)
        root.addWidget(self.progress)
        root.addWidget(eval_group, 2)
        self.setLayout(root)

    def _wire(self):
        self.btn_pick_images.clicked.connect(self.pick_images_dir)
        self.btn_pick_results.clicked.connect(self.pick_results_dir)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_eval.clicked.connect(self.run_evaluate)
        self.btn_video.clicked.connect(self.run_video)
        self.btn_open_out.clicked.connect(self.open_results_dir)

    def pick_images_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入图片文件夹", self.images_dir_edit.text() or "")
        if path:
            self.images_dir_edit.setText(path)

    def pick_results_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择导出目录", self.results_dir_edit.text() or "")
        if path:
            self.results_dir_edit.setText(path)

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_eval.setEnabled(not running)
        self.btn_video.setEnabled(not running)
        self.btn_pick_images.setEnabled(not running)
        self.btn_pick_results.setEnabled(not running)
        self.btn_open_out.setEnabled(True)
        self.progress.setEnabled(True)

    def run_pipeline(self):
        images_dir = self.images_dir_edit.text().strip()
        results_dir = self.results_dir_edit.text().strip()

        if not images_dir:
            QMessageBox.warning(self, "缺少输入", "请先选择输入图片文件夹。")
            return
        if not results_dir:
            QMessageBox.warning(self, "缺少导出目录", "请先选择导出目录。")
            return

        self.progress.setValue(0)
        self.eval_table.setRowCount(0)
        self.set_running(True)

        thread, worker = start_pipeline_in_thread(images_dir=images_dir, results_dir=results_dir)
        self._thread, self._worker = thread, worker

        worker.progress.connect(self.progress.setValue)
        worker.finished.connect(self.on_finished)
        worker.failed.connect(self.on_failed)

        thread.start()

    def run_evaluate(self):
        results_dir = self.results_dir_edit.text().strip()
        if not results_dir:
            QMessageBox.warning(self, "缺少导出目录", "请先选择导出目录。")
            return

        self.progress.setValue(0)
        self.eval_table.setRowCount(0)
        self.set_running(True)

        thread, worker = start_evaluate_in_thread(results_dir=results_dir)
        self._thread, self._worker = thread, worker
        worker.progress.connect(self.progress.setValue)
        worker.eval_row.connect(self.add_eval_row)
        worker.finished.connect(self.on_eval_finished)
        worker.failed.connect(self.on_failed)
        thread.start()

    def run_video(self):
        results_dir = self.results_dir_edit.text().strip()
        if not results_dir:
            QMessageBox.warning(self, "缺少导出目录", "请先选择导出目录。")
            return

        out_path = os.path.join(results_dir, "registration_demo.mp4")
        self.progress.setValue(0)
        self.set_running(True)

        thread, worker = start_video_in_thread(results_dir=results_dir, out_path=out_path, fps=10)
        self._thread, self._worker = thread, worker
        worker.progress.connect(self.progress.setValue)
        worker.finished.connect(self.on_video_finished)
        worker.failed.connect(self.on_failed)
        thread.start()

    def on_finished(self, out_dir: str):
        self.set_running(False)
        QMessageBox.information(self, "完成", f"配准完成，输出目录：\n{out_dir}")

    def on_eval_finished(self, out_csv: str):
        self.set_running(False)
        QMessageBox.information(self, "评价完成", f"评价结果已保存：\n{out_csv}")

    def on_video_finished(self, out_path: str):
        self.set_running(False)
        QMessageBox.information(self, "视频已生成", f"对比视频已保存：\n{out_path}")

    def on_failed(self, trace: str):
        self.set_running(False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("失败")
        box.setText("运行失败。")
        box.setDetailedText(trace)
        box.exec_()

    def add_eval_row(self, row: dict):
        r = self.eval_table.rowCount()
        self.eval_table.insertRow(r)
        cols = [
            "frame",
            "ncc_before",
            "ncc_after",
            "dsc_before",
            "dsc_after",
        ]
        for c, key in enumerate(cols):
            v = row.get(key, "")
            if isinstance(v, float):
                self.eval_table.setItem(r, c, QTableWidgetItem(f"{v:.6f}"))
            else:
                self.eval_table.setItem(r, c, QTableWidgetItem(str(v)))

    def open_results_dir(self):
        path = self.results_dir_edit.text().strip()
        if not path:
            return
        if os.path.isdir(path):
            # Windows: 用资源管理器打开
            os.startfile(path)  # noqa: S606


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

