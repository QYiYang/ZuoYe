import tkinter as tk
from tkinter import ttk
import torch
import requests
import yfinance as yf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import date
import akshare as ak
import LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

matplotlib.use('TkAgg')

# 输入的维度为1，只有Close收盘价
input_dim = 1
# 隐藏层特征的维度
hidden_dim = 32
# 循环的layers
num_layers = 2
# 预测后一天的收盘价
output_dim = 1
model = LSTM.LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)


def plot_stock():
    ticker = stock_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    data_type = data_type_var.get()
    data = LSTM.load_data(ticker, start_date, end_date)

    if data is not None:
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(data['Date'], data[data_type], label=data_type)
        ax.set_title(f'{ticker} {data_type} from {start_date} to {end_date}')
        ax.set_xlabel('Date')
        ax.set_ylabel(data_type)
        ax.legend()

        # 清除之前的图像
        for widget in plot_frame.winfo_children():
            widget.destroy()

        # 绘制新的图像
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    else:
        print("Failed to fetch data or data is empty.")


def fetch_stock_codes():
    try:
        stock_codes_df = ak.stock_info_a_code_name()
        stock_codes_text.delete(1.0, tk.END)
        for index, row in stock_codes_df.iterrows():
            market = "SH" if row['code'].startswith('6') else "SZ"
            stock_codes_text.insert(tk.END, f"{row['code']} - {row['name']} ({market})\n")
    except requests.exceptions.Timeout:
        stock_codes_text.delete(1.0, tk.END)
        stock_codes_text.insert(tk.END, "请求超时，请稍后再试。")
    except Exception as e:
        stock_codes_text.delete(1.0, tk.END)
        stock_codes_text.insert(tk.END, f"发生错误: {e}")


def on_test():
    # 加载数据
    scaler = MinMaxScaler(feature_range=(-1, 1))
    tricker = stock_entry.get()
    start_time = start_date_entry.get()
    end_time = end_date_entry.get()
    data = LSTM.load_data(tricker, start_time, end_time)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(LR_entry.get()))
    lookback = 20
    price = data[['Close']]
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))
    x_train, y_train, x_test, y_test = LSTM.preprocess_data(price, lookback)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)

    # 真实的数据
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

    hist = np.zeros(int(num_epoch_entry.get()))
    # 训练模型
    LSTM.train_model(model, x_train, y_train_lstm, hist, criterion, optimizer, num_epochs=int(num_epoch_entry.get()))

    # 预测
    y_train_pred = model(x_train)
    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    print(predict)  # 预测值
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
    print(original)  # 真实值

    # 绘图
    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 确保 original 和 predict 的索引是日期
    original.index = date_range[:len(original)]
    predict.index = date_range[:len(predict)]

    sns.set_style("darkgrid")

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x=predict.index, y=predict[0], label="Training Prediction (LSTM)", color='tomato')
    print(predict.index)
    print("aaaa")
    print(predict[0])

    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD)", size=14)

    # 设置日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 每一个月显示一个日期
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("Training Loss", size=14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.show()


def on_predict():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    tricker = stock_entry.get()
    start_time = start_date_entry.get()
    end_time = end_date_entry.get()
    lookback = 20
    data = LSTM.load_data(tricker, start_time, end_time)
    price = data[['Close']]
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

    data_raw = price.to_numpy()
    x = []
    # you can free play（seq_length）
    # 将data按lookback分组，data为长度为lookback的list
    for index in range(len(data_raw) - lookback):
        x.append(data_raw[index: index + lookback])

    x = np.array(x)
    print(type(x))
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    y_train = x[:train_set_size, -1, :]

    x_test = x[:data.shape[0], :-1, :]

    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_train_pred = model(x_test)

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
    # 绘图
    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 确保 original 和 predict 的索引是日期
    original.index = date_range[:len(original)]
    predict.index = date_range[:len(predict)]

    sns.set_style("darkgrid")

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x=predict.index, y=predict[0], label="Training Prediction (LSTM)", color='tomato')

    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD)", size=14)
    ax.set_xticklabels('', size=10)

    # 设置日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 每一个月显示一个日期
    plt.xticks(rotation=45)

    fig.set_figheight(6)
    fig.set_figwidth(16)

    plt.show()


def get_months(period_str):
    if period_str == '1mo':
        return 1
    if period_str == '3mo':
        return 3
    if period_str == '6mo':
        return 6
    if period_str == '1y':
        return 12
    if period_str == '5y':
        return 60


window = tk.Tk()
window.title("Stock Price Viewer")

frame = ttk.Frame(window, padding="10")
frame.pack()

ttk.Label(frame, text="Enter Stock Ticker:").pack()
stock_entry = ttk.Entry(frame)
stock_entry.insert(0, "000001.SZ")
stock_entry.pack()

ttk.Label(frame, text="Enter Start Date (YYYY-MM-DD):").pack()
start_date_entry = ttk.Entry(frame)
start_date_entry.insert(0, "2024-01-01")  # 默认值
start_date_entry.pack()

end_date_entry = ttk.Entry(frame)
end_date_entry.insert(0, date.today().strftime("%Y-%m-%d"))  # 默认值
ttk.Label(frame, text="Enter End Date (YYYY-MM-DD):").pack()
end_date_entry.pack()

ttk.Label(frame, text="Enter lr:").pack()
LR_entry = ttk.Entry(frame)
LR_entry.insert(0, "0.01")  # 默认值
LR_entry.pack()

ttk.Label(frame, text="Enter num_epoch:").pack()
num_epoch_entry = ttk.Entry(frame)
num_epoch_entry.insert(0, "100")  # 默认值
num_epoch_entry.pack()

ttk.Label(frame, text="Select Data Type:").pack()
data_type_var = tk.StringVar()
data_type_menu = ttk.Combobox(frame, textvariable=data_type_var)
data_type_menu['values'] = ('Close', 'Open', 'High', 'Low', 'Volume')
data_type_menu.current(0)  # 默认选择 'Close'
data_type_menu.pack()

ttk.Button(frame, text="Plot", command=plot_stock).pack()

ttk.Button(frame, text="Fetch Stock Codes", command=fetch_stock_codes).pack()

ttk.Button(frame, text="test", command=on_test).pack()

ttk.Button(frame, text="predict", command=on_predict).pack()

stock_codes_text = tk.Text(frame, height=10, width=50)
stock_codes_text.pack()

plot_frame = ttk.Frame(window, padding="10")
plot_frame.pack()

window.mainloop()