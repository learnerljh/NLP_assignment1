from torch.utils.data import Dataset, DataLoader
import jieba
import torch.nn as nn
import torch
import torch.nn.functional as F
epoch_num=4
batch_size=4
embed_size=256
num_filters=32
filter_sizes=[2, 2, 2, 3, 3, 3, 4, 4, 5, 6]
hidden_size=64
num_classes=4
dropout=0.5
vocab_size=58559
max_len=16
def build_vocab(file_path, tokenizer):
    all_tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line[:-1]  # Remove the newline character at the end
            if line:
                text = line.strip()  # Remove any extra whitespaces from the line
                tokens = tokenizer(text)
                all_tokens.extend(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for token in all_tokens:
        vocab[token] = len(vocab)
    return vocab

class TextNumberDataset(Dataset):
    def __init__(self, file_path, vocab, max_len=16):
        self.data = []
        self.vocab = vocab
        self.max_len = max_len

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # Remove any extra whitespaces from the line
                if line: # Split into text and number
                    text = line[:-1]  # The text part
                    text = jieba.cut(text, cut_all=False)
                    text = [self.vocab.get(token, self.vocab['<UNK>']) for token in text]
                    # Pad the sequence if it is less than max_len
                    if len(text) < self.max_len:
                        text += [self.vocab['<PAD>']] * (self.max_len - len(text))
                    else:
                        text = text[:self.max_len]
                    # one_hot encoding
                    token_tensor = torch.tensor(text, dtype=torch.long)
                    number = int(line[-1])  # Convert the number part to a float
                    self.data.append((token_tensor, number))  # Store as a tuple (text, number)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, number = self.data[idx]
        return text, number

def create_dataloader(file_path, vocab, batch_size=batch_size, shuffle=True):
    dataset = TextNumberDataset(file_path, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class CNNmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, hidden_size, num_classes, dropout, max_len):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes])
        self.poolings = nn.ModuleList([
            nn.MaxPool2d((self.max_len - fs + 1, 1)) for fs in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  
        self.output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch_size, max_len, embed_size)
        x = x.unsqueeze(1) 
        x = [conv(x) for conv in self.convs]  # Apply convolution: list of (batch_size, num_filters, H_out, 1)
        x = [pool(conv) for conv, pool in zip(x, self.poolings)]  # Apply pooling: list of (batch_size, num_filters, 1, 1)
        x = torch.cat(x, dim=1)  # Concatenate along the num_filters dimension: (batch_size, num_filters * len(filter_sizes), 1, 1)
        x = x.squeeze(3).squeeze(2)  # Remove the extra dimensions: (batch_size, num_filters * len(filter_sizes))
        x = self.fc(x)  # Fully connected layer: (batch_size, hidden_size)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)  # Output layer: (batch_size, num_classes)
        return x
    
    
if __name__ == "__main__":
    vocab = build_vocab('train.txt', tokenizer=jieba.cut)
    train_file_path = 'train.txt'
    train_dataloader = create_dataloader(train_file_path, vocab, batch_size=2, shuffle=True)
    valid_file_path = 'dev.txt'
    valid_dataloader = create_dataloader(valid_file_path, vocab, batch_size=2, shuffle=False)
    model = CNNmodel(vocab_size=vocab_size, embed_size=embed_size, num_filters=num_filters, filter_sizes=filter_sizes, hidden_size=hidden_size, num_classes=num_classes, dropout=dropout, max_len=max_len).to('cuda')   
    optimizer = torch.optim.Adam(model.parameters())
    torch.cuda.empty_cache()

    best_loss = float('inf')
    for epoch in range(epoch_num):
        model.train()
        for batch in train_dataloader:
            texts, numbers = batch
            texts = texts.to('cuda')
            numbers = numbers.to('cuda')    
            optimizer.zero_grad()
            output = model(texts)
            loss = F.cross_entropy(output, numbers)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epoch_num}, Training Loss: {loss.item()}")
        model.eval()
        valid_loss = 0
        valid_batch_num = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                texts, numbers = batch
                texts = texts.to('cuda')
                numbers = numbers.long().to('cuda')  # Ensure labels are LongTensor for cross_entropy
                
                valid_output = model(texts)
                loss = F.cross_entropy(valid_output, numbers)
                valid_loss += loss.item()
                valid_batch_num += 1
            valid_loss /= valid_batch_num
            print(f"Epoch {epoch + 1}/{epoch_num}, Validation Loss: {valid_loss}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Best model saved at epoch {epoch + 1} with validation loss {best_loss}")

    test_file_path = 'test.txt'
    test_dataloader = create_dataloader(test_file_path, vocab, batch_size=2, shuffle=False)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        for batch in test_dataloader:
            texts, numbers = batch
            texts = texts.to('cuda')
            numbers = numbers.to('cuda')
            output = model(texts)
            predictions = output.argmax(dim=1)
            for pred, truth in zip(predictions, numbers):
                if pred == truth:
                    correct_num += 1
                total_num += 1
        accuracy = correct_num / total_num
        print(f"Test accuracy: {accuracy}")
                    

            