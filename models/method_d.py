import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class LotteryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LotteryLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        logits = self.fc(out)
        return logits

def train_predict_lstm(df, model_config, lottery_config):
    try:
        torch.set_num_threads(1)

        TOTAL_NUMBERS = lottery_config['total_numbers']
        WINDOW_SIZE = lottery_config['window_size']
        RED_COLS = lottery_config['red_cols']
        SEPARATE_POOL = lottery_config['separate_pool']

        if SEPARATE_POOL:
            RED_NUM_LIST = lottery_config['red_num_list']
            BLUE_NUM_LIST = lottery_config['blue_num_list']
            BLUE_COLS = lottery_config['blue_cols']
            TOTAL_RED = lottery_config['total_red']
        else:
            RED_NUM_LIST = lottery_config['red_num_list']

        logging.info(f"Method D: 训练 LSTM 走势捕捉模型 (内部ID映射, 词表大小={TOTAL_NUMBERS})...")

        n_rows = len(df)
        ball_seqs = []
        for i in range(n_rows):
            ids = []
            row_red = [int(n) for n in df.iloc[i][RED_COLS].values if pd.notna(n)]
            for val in row_red:
                if val in RED_NUM_LIST:
                    ids.append(RED_NUM_LIST.index(val))
            if SEPARATE_POOL:
                row_blue = [int(n) for n in df.iloc[i][BLUE_COLS].values if pd.notna(n)]
                for val in row_blue:
                    if val in BLUE_NUM_LIST:
                        ids.append(TOTAL_RED + BLUE_NUM_LIST.index(val))
            ball_seqs.append(sorted(ids))

        X_list, y_list = [], []
        for i in range(WINDOW_SIZE, n_rows):
            x_seq = []
            for j in range(i - WINDOW_SIZE, i):
                x_seq.extend(ball_seqs[j])

            for target_id in ball_seqs[i]:
                X_list.append(x_seq)
                y_list.append(target_id)

        if not X_list:
            return np.zeros(TOTAL_NUMBERS)

        X_train = torch.LongTensor(X_list)
        y_train = torch.LongTensor(y_list)

        vocab_size = TOTAL_NUMBERS
        conf_d = model_config
        model = LotteryLSTM(
            vocab_size,
            conf_d['embedding_dim'],
            conf_d['hidden_dim'],
            conf_d['num_layers'],
            conf_d['dropout']
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=conf_d['lr'])

        epochs = conf_d['epochs']
        model.train()
        pbar = tqdm(range(epochs), desc="LSTM 训练", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            if torch.isnan(loss): break
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        last_x = []
        for j in range(n_rows - WINDOW_SIZE, n_rows):
            last_x.extend(ball_seqs[j])

        last_x_tensor = torch.LongTensor([last_x])
        with torch.no_grad():
            logits = model(last_x_tensor)
            final_probs = torch.softmax(logits, dim=1).numpy()[0]

        return final_probs
    except Exception as e:
        logging.error(f"Method D (LSTM) Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return np.zeros(TOTAL_NUMBERS)