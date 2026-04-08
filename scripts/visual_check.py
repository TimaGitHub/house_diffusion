import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from PIL import Image

# --- 1. НАСТРОЙКА ПУТЕЙ ---
FOLDERS = {
    'gt_no_doors': 'outputs/gt_no_doors',
    'gt': 'outputs/gt',
    'pred_no_doors': 'outputs/pred_no_doors',
    'pred': 'outputs/pred'
}

LEGEND_DATA = [
    ("Kitchen", "#C67C7B"), ("Storage", "#1F849B"), ("Bedroom", "#FFD274"),
    ("Entrance", "#7BA779"), ("Bathroom", "#BEBEBE"), ("Study Room", "#FF8C69"),
    ("Balcony", "#BFE3E8"), ("Living Room", "#EE4D4D"), ("Dining Room", "#E87A90"),
    ("Outside", "#FFFFFF"), ("Front Door", "#727171"), ("Interior Door", "#D3A2C7"),
    ("Unknown", "#785A67")
]


def get_indices(folder_path):
    if not os.path.exists(folder_path):
        print(f"ОШИБКА: Путь не найден: {os.path.abspath(folder_path)}")
        return []
    indices = [
        int(re.findall(r'\d+', f)[0])
        for f in os.listdir(folder_path)
        if f.endswith(('.jpg', '.png')) and re.findall(r'\d+', f)
    ]
    return sorted(indices)


def get_path(key, n):
    """
    Формирует пути. Предполагается формат {n}c_... для png
    и просто {n}.png для обычных gt/pred (если у вас иначе — поправьте).
    """
    if key == 'gt_no_doors':
        return os.path.join(FOLDERS[key], f"{n}c_gt_no_doors.png")
    elif key == 'pred_no_doors':
        return os.path.join(FOLDERS[key], f"{n}c_pred_no_doors.png")
    elif key == 'gt':
        return os.path.join(FOLDERS[key], f"{n}c_gt.png")  # или f"{n}.png"
    elif key == 'pred':
        return os.path.join(FOLDERS[key], f"{n}c_pred.png")  # или f"{n}.png"
    return ""


# --- 2. КЛАСС ВИЗУАЛИЗАТОРА ---
class WideVisualizer:
    def __init__(self):
        # Используем gt_no_doors как эталон для списка индексов
        self.all_indices = get_indices(FOLDERS['gt_no_doors'])

        if not self.all_indices:
            print("Список файлов пуст. Проверьте папку 'outputs/gt_no_doors'.")
            return

        self.current_idx = 0

        # Создаем широкое окно (figsize 20x7) и сетку 1x4
        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 7))
        plt.subplots_adjust(bottom=0.2, right=0.85, top=0.85, wspace=0.2)

        # Кнопки навигации
        ax_prev = plt.axes([0.4, 0.05, 0.08, 0.05])
        ax_next = plt.axes([0.52, 0.05, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, '⬅ Назад')
        self.btn_next = Button(ax_next, 'Вперед ➡')

        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)

        # Горячие клавиши
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.draw_content()

    def draw_content(self):
        n = self.all_indices[self.current_idx]
        self.fig.suptitle(f'Образец №{n} ({self.current_idx + 1}/{len(self.all_indices)})',
                          fontsize=16, fontweight='bold')

        # Список того, что отображаем в ряд
        mapping = [
            ('gt_no_doors', 'GT (No Doors)'),
            ('gt', 'Ground Truth'),
            ('pred_no_doors', 'Pred (No Doors)'),
            ('pred', 'Prediction')
        ]

        for i, (key, title) in enumerate(mapping):
            ax = self.axes[i]
            ax.clear()
            path = get_path(key, n)

            if os.path.exists(path):
                img = Image.open(path)
                ax.imshow(img)
                ax.set_title(title, fontsize=12, pad=10)
            else:
                ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}",
                        ha='center', va='center', color='red', fontsize=10)
            ax.axis('off')

        # Легенда
        self.fig.legends = []  # Очистка старой легенды
        patches = [mpatches.Patch(color=color, label=label) for label, color in LEGEND_DATA]
        self.fig.legend(handles=patches, title="Legend", loc='center right',
                        bbox_to_anchor=(0.98, 0.5), fontsize=10, frameon=True)

        self.fig.canvas.draw_idle()

    def next(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.all_indices)
        self.draw_content()

    def prev(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.all_indices)
        self.draw_content()

    def on_key(self, event):
        if event.key == 'right':
            self.next(None)
        elif event.key == 'left':
            self.prev(None)


if __name__ == "__main__":
    vis = WideVisualizer()
    if hasattr(vis, 'all_indices') and vis.all_indices:
        plt.show()