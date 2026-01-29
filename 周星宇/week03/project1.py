import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# --- 通用的RNN分类器 (支持RNN/LSTM/GRU) ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='LSTM'):
        super(RNNClassifier, self).__init__()
        
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 根据类型选择不同的RNN层
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        
        if self.rnn_type == 'LSTM':
            rnn_out, (hidden_state, cell_state) = self.rnn(embedded)
        else:  # RNN 和 GRU
            rnn_out, hidden_state = self.rnn(embedded)
        
        out = self.fc(hidden_state.squeeze(0))
        return out


# --- 训练函数 ---
def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name):
    print(f"\n========== 开始训练 {model_name} 模型 ==========")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"[{model_name}] Batch {idx}, 当前Batch Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"[{model_name}] Epoch [{epoch + 1}/{num_epochs}], 平均Loss: {avg_loss:.4f}")
    print(f"========== {model_name} 模型训练完成 ==========\n")
    return avg_loss


# --- 评估函数 ---
def evaluate_model(model, dataloader, model_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"[{model_name}] 准确率: {accuracy:.2f}%")
    return accuracy


# --- 预测函数 ---
def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# --- 准备数据 ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_epochs = 4
index_to_label = {i: label for label, i in label_to_index.items()}

# 存储实验结果
results = {}

# --- 实验1: RNN ---
print("\n" + "="*60)
print("实验1: 使用 RNN 进行文本分类")
print("="*60)
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='RNN')
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
rnn_loss = train_model(rnn_model, dataloader, rnn_criterion, rnn_optimizer, num_epochs, "RNN")
rnn_accuracy = evaluate_model(rnn_model, dataloader, "RNN")
results['RNN'] = {'loss': rnn_loss, 'accuracy': rnn_accuracy}

# --- 实验2: LSTM ---
print("\n" + "="*60)
print("实验2: 使用 LSTM 进行文本分类")
print("="*60)
lstm_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='LSTM')
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
lstm_loss = train_model(lstm_model, dataloader, lstm_criterion, lstm_optimizer, num_epochs, "LSTM")
lstm_accuracy = evaluate_model(lstm_model, dataloader, "LSTM")
results['LSTM'] = {'loss': lstm_loss, 'accuracy': lstm_accuracy}

# --- 实验3: GRU ---
print("\n" + "="*60)
print("实验3: 使用 GRU 进行文本分类")
print("="*60)
gru_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='GRU')
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
gru_loss = train_model(gru_model, dataloader, gru_criterion, gru_optimizer, num_epochs, "GRU")
gru_accuracy = evaluate_model(gru_model, dataloader, "GRU")
results['GRU'] = {'loss': gru_loss, 'accuracy': gru_accuracy}

# --- 结果对比 ---
print("\n" + "="*60)
print("实验结果对比")
print("="*60)
print(f"{'模型类型':<10} {'最终Loss':<15} {'准确率':<15}")
print("-"*60)
for model_type, metrics in results.items():
    print(f"{model_type:<10} {metrics['loss']:<15.4f} {metrics['accuracy']:<15.2f}%")
print("="*60)

# 找出最佳模型
best_model_type = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n最佳模型: {best_model_type}, 准确率: {results[best_model_type]['accuracy']:.2f}%")

# --- 使用最佳模型进行预测 ---
if best_model_type == 'RNN':
    best_model = rnn_model
elif best_model_type == 'LSTM':
    best_model = lstm_model
else:
    best_model = gru_model

print(f"\n使用 {best_model_type} 模型进行预测:")
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
