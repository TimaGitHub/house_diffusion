# House Diffusion

## 🛠 Установка и настройка окружения

Инструкция ориентирована на операционную систему Windows.

### 1. Системные требования
Перед установкой Python-пакетов необходимо подготовить системные компоненты:

1.  **Python 3.10**: Убедитесь, что используете именно эту версию.
2.  **GTK for Windows**: Скачайте и запустите [установщик GTK](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).
3.  **Microsoft MPI**:
    * Установите `msmpisetup.exe` (основная библиотека).
    * Установите `msmpisdk.msi` (пакет разработчика).
4.  **Visual Studio Build Tools**: Необходимо установить компоненты для сборки C++.
5.  **Visual C++ Redistributable**: Скачайте актуальную версию [здесь](https://aka.ms/vs/17/release/vc_redist.x64.exe).
6.  **Graphviz**: Скачайте и установите для корректной визуализации графов [отсюда](https://graphviz.org/download/).

### 2. Клонирование и установка зависимостей

Выполните следующие команды в терминале:

```bash
# Клонирование репозитория
git clone [https://github.com/TimaGitHub/house_diffusion.git](https://github.com/TimaGitHub/house_diffusion.git)
cd house_diffusion

# Установка зависимостей с поддержкой CUDA
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# Установка проекта в режиме редактирования
pip install -e .
```

---

## 📂 Подготовка данных и весов

Для работы модели необходимо разместить файлы в соответствующих папках.

### Веса модели (Checkpoints)
1. Скачайте файл `model250000.pt` по [ссылке](https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view).
2. Поместите его в директорию: `ckpts/exp/`.

### Набор данных (Dataset)
1. Скачайте данные по [ссылке](https://drive.google.com/file/d/12lfJ8cxRs5gbjeDbPdk1AhQQdpa4GXlX/view?usp=drive_link).
2. Извлеките файлы: `rplan__8.npz`, `rplan_eval_8.npz`, `rplan_eval_8_syn.npz`, `rplan_train_8.npz`, `rplan_train_8_cndist.npz`.
3. Поместите эти файлы в директорию: `scripts/processed_rplan/`.

---

## 🚀 Использование

### Генерация планировок
Перейдите в папку со скриптами и запустите генерацию:

```bash
cd scripts
python image_sample.py --dataset rplan --batch_size 16 --model_path ../ckpts/exp/model250000.pt --num_samples 10000 --target_set 8 --save_svg True --set_name eval
```

### Визуальная проверка
Для сравнения и просмотра полученных результатов запустите:

```bash
python visual_check.py
```
