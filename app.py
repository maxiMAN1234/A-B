import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
import dash
from dash import dcc, html, Input, Output, ALL
import dash_table

# Инициализация Dash-приложения
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Загрузка и обработка файла"),
    
    # Часть загрузки файла
    html.Label("Введите путь к файлу (CSV или Excel):"),
    dcc.Input(id='file-path', type='text', placeholder='Пример: C:/Users/User/data.csv', style={'width': '80%'}),
    html.Button("📂 Загрузить файл", id='load-file', n_clicks=0),
    html.Div(id='file-status'),
    
    # Хранилище для сохранённого DataFrame (в формате JSON)
    dcc.Store(id='stored-df', storage_type='memory'),
    
    html.Hr(),
    
    # Динамически сгенерированные выпадающие списки для каждой колонки
    html.Div(id='dropdown-container'),
    
    html.Hr(),
    
    # Ввод пути к папке для сохранения результатов
    html.Label("Введите путь к папке для сохранения файлов:"),
    dcc.Input(id='save-folder', type='text', placeholder='Пример: C:/Users/User/Documents', style={'width': '80%'}),
    html.Button("💾 Сохранить файлы", id='save-button', n_clicks=0),
    html.Div(id='output')
])

# CALLBACK 1: Загрузка файла и сохранение в dcc.Store
@app.callback(
    [Output('file-status', 'children'),
     Output('stored-df', 'data')],
    Input('load-file', 'n_clicks'),
    Input('file-path', 'value')
)
def load_file_callback(n_clicks, file_path):
    if n_clicks == 0:
        return "", None
    if not file_path:
        return "Введите путь к файлу!", None
    if not os.path.exists(file_path):
        return f"❌ Файл '{file_path}' не найден!", None
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0)
        else:
            df = pd.read_excel(file_path, index_col=0)
        msg = f"✅ Файл загружен: {file_path} (строк: {df.shape[0]}, колонок: {df.shape[1]})"
        # Сохраняем DataFrame в формате JSON (orient='split')
        return msg, df.to_json(date_format='iso', orient='split')
    except Exception as e:
        return f"❌ Ошибка загрузки файла: {str(e)}", None

# CALLBACK 2: Генерация выпадающих списков для выбора типа колонок (на основе загруженного DataFrame)
@app.callback(
    Output('dropdown-container', 'children'),
    Input('stored-df', 'data')
)
def generate_dropdowns(data):
    if data is None:
        return ""
    df = pd.read_json(data, orient='split')
    dropdowns = []
    for i, col in enumerate(df.columns):
        dropdowns.append(
            html.Div([
                html.Label(col[:40] + ('...' if len(col) > 40 else '')),
                dcc.Dropdown(
                    id={'type': 'column-dropdown', 'index': i},
                    options=[
                        {'label': '1️⃣ Числовая', 'value': 1},
                        {'label': '2️⃣ Бинарная', 'value': 2},
                        {'label': '3️⃣ Категориальная', 'value': 3},
                        {'label': '4️⃣ Оставить, но не добавлять', 'value': 4},
                        {'label': '5️⃣ Удалить', 'value': 5},
                    ],
                    value=1,  # значение по умолчанию
                )
            ], style={'margin-bottom': '10px'})
        )
    return dropdowns

# CALLBACK 3: Обработка данных (с учётом выбранных типов колонок) и сохранение результатов в указанный путь
@app.callback(
    Output('output', 'children'),
    Input('save-button', 'n_clicks'),
    Input('save-folder', 'value'),
    Input({'type': 'column-dropdown', 'index': ALL}, 'value'),
    Input('stored-df', 'data')
)
def process_and_save(n_clicks, save_folder, dropdown_values, data):
    if n_clicks == 0:
        return ""
    if data is None:
        return "Сначала загрузите файл!"
    if not save_folder or not os.path.exists(save_folder):
        return "❌ Укажите корректный путь для сохранения!"
    
    # Восстанавливаем DataFrame из JSON
    df = pd.read_json(data, orient='split')
    
    # Сбор выбранных категорий по колонкам
    numeric_columns = []
    binary_columns = []
    categorial_columns = []
    keep_columns = []
    drop_columns = []
    
    for idx, col in enumerate(df.columns):
        col_type = dropdown_values[idx]
        if col_type == 1:
            numeric_columns.append(col)
        elif col_type == 2:
            binary_columns.append(col)
        elif col_type == 3:
            categorial_columns.append(col)
        elif col_type == 4:
            keep_columns.append(col)
        elif col_type == 5:
            drop_columns.append(col)
    
    # Применение LabelEncoder для категориальных колонок
    le = LabelEncoder()
    for col in categorial_columns:
        df[col] = le.fit_transform(df[col].fillna('missing'))
    
    # Заменяем пропуски медианой для числовых колонок
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Деление DataFrame на две группы с проверкой однородности с помощью KS-теста
    max_attempts = 50
    attempt = 1
    is_homogeneous = False
    df1, df2 = None, None
    results = {}
    
    while attempt < max_attempts and not is_homogeneous:
        df1, df2 = train_test_split(df, test_size=0.3, random_state=attempt)
        is_homogeneous = True  # предполагаем, что группы однородны
        results = {}
        for col in numeric_columns:
            stat, p_value = ks_2samp(df1[col], df2[col])
            results[col] = {'KS Statistic': stat, 'p-value': p_value}
            if p_value < 0.5:
                is_homogeneous = False
        attempt += 1

    if is_homogeneous:
        file_A = os.path.join(save_folder, 'A.xlsx')
        file_B = os.path.join(save_folder, 'B.xlsx')
        df1.to_excel(file_A, index=False)
        df2.to_excel(file_B, index=False)
        output = f"✅ Группы однородны! Количество попыток: {attempt}\n"
        output += '\n'.join([f"{col}: KS Statistic = {res['KS Statistic']}, p-value = {res['p-value']}" 
                              for col, res in results.items()])
        output += f"\nФайлы сохранены:\n📂 {file_A}\n📂 {file_B}"
    else:
        output = "❌ Группы не однородны. Попробуйте снова."
    
    return output

if __name__ == '__main__':
    app.run_server(debug=True)