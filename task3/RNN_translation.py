from __future__ import unicode_literals, print_function, division
import time
import math
from janome.tokenizer import Tokenizer
from torch.utils.data import Dataset
from io import open
import unicodedata
import re
import random
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from janome.tokenizer import Tokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

PAD_token = 0
EOS_token = 1
SOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": 0, "EOS": 1, "SOS": 2}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "EOS", 2: "SOS"}
        self.n_words = 3  # Count PAD and EOS

    def addSentence(self, sentence, tokenizer=None):
        if self.name == 'jpn':
            for word in tokenizer.tokenize(sentence):
                self.addWord(word)
        else:
            for word in sentence.split(' '):
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s_%s.txt' % (lang2, lang1), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]
    for pair in pairs:
        pair[1] = normalizeString(pair[1])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 13

def filterPair(p, tokenizer):
    return len([word for word in tokenizer.tokenize(p[0])]) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    tokenizer = Tokenizer(wakati=True)
    return [pair for pair in pairs if filterPair(pair, tokenizer)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    tokenizer = Tokenizer(wakati=True)
    for pair in pairs:
        input_lang.addSentence(pair[0], tokenizer)
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(random.choice(pairs))
    return input_lang, output_lang, pairs

def create_embedding_matrix(word2vec_model, lang, hidden_size):
    embedding_matrix = torch.zeros((lang.n_words, hidden_size))
    for word, idx in lang.word2index.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = torch.tensor(word2vec_model.wv[word])
        else:
            embedding_matrix[idx] = torch.randn(hidden_size)
    return embedding_matrix

class EncoderRNN(nn.Module):
    def __init__(self, input_embedding, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(input_embedding, freeze=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden= self.gru(embedded)
        return output, hidden

class EncoderLSTM(nn.Module):
    def __init__(self, input_embedding, hidden_size, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(input_embedding, freeze=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        embedded = self.layernorm(embedded)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)

class DecoderRNN(nn.Module):
    def __init__(self, output_embedding, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(output_embedding, freeze=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(PAD_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, output_embedding, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(output_embedding, freeze=True)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(PAD_token)
        decoder_hidden  = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = torch.tensor(hidden).permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, output_embedding, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(output_embedding, freeze=True)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        (decoder_hidden, decoder_cell) = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, (decoder_hidden, decoder_cell), attn_weights = self.forward_step(
                decoder_input, (decoder_hidden, decoder_cell), encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        
        return decoder_outputs, (decoder_hidden, decoder_cell), attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = torch.tensor(hidden[0]).permute(1, 0, 2)  # Use hidden state from LSTM
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(input_lstm, hidden)
        output = self.layernorm(output)
        output = self.out(output)

        return output, (hidden, cell), attn_weights

def indexesFromSentence(lang, sentence, tokenizer=None):
    if tokenizer:
        return [lang.word2index[word] for word in tokenizer.tokenize(sentence)]
    return [lang.word2index[word] for word in sentence.strip().split(' ')]

def tensorFromSentence(lang, sentence, tokenizer=None):
    indexes = indexesFromSentence(lang, sentence, tokenizer)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader_and_word2vec(batch_size, vector_size):
    import nltk
    nltk.download('punkt_tab')
    from nltk.tokenize import word_tokenize
    input_lang, output_lang, pairs = prepareData('jpn', 'eng')
    input_sentences = [pair[0] for pair in pairs]
    output_sentences = [word_tokenize(pair[1]) for pair in pairs]
    input_word2vec_model = Word2Vec(input_sentences, vector_size=vector_size, window=5, min_count=1, sg=1)
    output_word2vec_model = Word2Vec(output_sentences, vector_size=vector_size, window=5, min_count=1, sg=1)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    tokenizer = Tokenizer(wakati=True)
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp, tokenizer)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    num_ids = len(input_ids)
    print(input_ids.shape)
    train_ids = num_ids * 0.8
    valid_ids = num_ids * 0.1
    all_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                torch.LongTensor(target_ids).to(device))
    train_data, valid_data, test_data = torch.utils.data.random_split(all_data, [int(train_ids), int(valid_ids), int(num_ids - train_ids - valid_ids)])

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, valid_dataloader, test_dataloader, input_word2vec_model, output_word2vec_model

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, pad_token=PAD_token):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def calulate_validation_loss(dataloader, encoder, decoder, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, valid_dataloader, encoder, decoder, n_epochs, learning_rate,
            print_every=1, plot_every=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    encoder_sceduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=10, eta_min=0.0002)
    decoder_sceduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=10, eta_min=0.0002)
    criterion = nn.NLLLoss()
    best_valid_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        valid_loss = calulate_validation_loss(valid_dataloader, encoder, decoder, criterion)
        print('Validation Loss: %.4f' % valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(encoder.state_dict(), 'encoder_best.pth')
            torch.save(decoder.state_dict(), 'decoder_best.pth')
        
        encoder_sceduler.step()
        decoder_sceduler.step()

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        tokenizer = Tokenizer(wakati=True)
        input_tensor = tensorFromSentence(input_lang, sentence, tokenizer)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, input_lang, output_lang):
    case_1 = "私の名前は愛です"
    case_2 = "昨日はお肉を食べません"
    case_3 = "いただきますよう"
    case_4 = "秋は好きです"
    case_5 = "おはようございます"
    cases = [case_1, case_2, case_3, case_4, case_5]
    for case in cases:
        print('>', case)
        output_words, _ = evaluate(encoder, decoder, case, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate_word2vec(word2vec_model, dataset_path):
    from sklearn.metrics.pairwise import cosine_similarity
    results = {}
    import csv

    with open(dataset_path, 'r', newline='\n', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        similarities = []
        human_scores = []
        reader = list(reader)
        for row in reader[4:]:
            word1, word2, human_score = row[0].split('\t')
            try:
                vector1 = word2vec_model.wv[word1].reshape(1, -1)
            except:
                continue
            try:
                vector2 = word2vec_model.wv[word2].reshape(1, -1)
            except:
                continue
            similarity = cosine_similarity(vector1, vector2)[0][0]
            human_score = float(human_score)
            similarities.append(similarity)
            human_scores.append(human_score)
            
        if similarities:
            correlation = np.corrcoef(similarities, human_scores)[0, 1]
            results['word_similarity'] = correlation
        else:
            results['word_similarity'] = None
    
    print(results)

def calculate_bleu_score_and_perplexity(encoder, decoder, dataloader, input_lang, output_lang):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    total_bleu_score = 0.0
    num_samples = 0
    total_log_prob = 0.0
    total_words = 0
    for batch in dataloader:
        input_tensor, target_tensor = batch
        for i in range(len(input_tensor)):
            input_sentence = ''
            for idx in input_tensor[i]:
                if input_lang.index2word[idx.item()] == 'EOS':
                    break
                input_sentence += input_lang.index2word[idx.item()]
            target_sentence = ''
            for idx in target_tensor[i]:
                if output_lang.index2word[idx.item()] == 'EOS':
                    break
                target_sentence += output_lang.index2word[idx.item()] + ' '
            new_input_tensor = input_tensor[i].unsqueeze(0)
            encoder_outputs, encoder_hidden = encoder(new_input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if output_lang.index2word[idx.item()] == 'EOS':
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
            output_sentence = ' '.join(decoded_words)

            reference = target_sentence.split()
            hypothesis = output_sentence.split()
            bleu_score = sentence_bleu([reference], hypothesis, weights=[0.5, 0.5, 0, 0], smoothing_function=SmoothingFunction().method1)
            if i < 5 and num_samples < 5:
                print(f"Input: {input_sentence}")
                print(f"Target: {target_sentence}")
                print(f"Output: {output_sentence}")
                print(f"BLEU Score: {bleu_score:.4f}")

            total_bleu_score += bleu_score
            num_samples += 1

            new_target_tensor = target_tensor[i]

            shift_logits = decoder_outputs.squeeze(0)
            shift_labels = new_target_tensor
        
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(shift_logits, shift_labels.view(-1))
            log_prob = -loss.sum().item()
        
            total_log_prob += log_prob
            total_words += len(new_target_tensor)
    
    perplexity = np.exp(-total_log_prob / total_words)

    average_bleu_score = total_bleu_score / num_samples if num_samples > 0 else 0.0
    return average_bleu_score, perplexity
            
def calculate_perplexity(encoder, decoder, dataloader):
    total_log_prob = 0.0
    total_words = 0
    for batch in dataloader:
        input_tensor, target_tensor = batch
        for i in range(len(input_tensor)):
            new_input_tensor = input_tensor[i].unsqueeze(0)
            new_target_tensor = target_tensor[i]
            encoder_outputs, encoder_hidden = encoder(new_input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            shift_logits = decoder_outputs.squeeze(0)
            shift_labels = new_target_tensor
        
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(shift_logits, shift_labels.view(-1))
            log_prob = -loss.sum().item()
        
            total_log_prob += log_prob
            total_words += len(new_target_tensor)
    
    perplexity = np.exp(-total_log_prob / total_words)
    return perplexity

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset.")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        required=True,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=6,
        required=True,
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.4,
        required=True,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    dropout_p = args.dropout_p
    input_lang, output_lang, train_dataloader, valid_dataloader, test_dataloader, \
        input_word2vec, output_word2vec = get_dataloader_and_word2vec(64, hidden_size)
    evaluate_word2vec(input_word2vec, r'.\wordsimjap.csv')
    evaluate_word2vec(output_word2vec, r'.\wordsimeng.csv')
    input_embedding = create_embedding_matrix(input_word2vec, input_lang, hidden_size)
    output_embedding = create_embedding_matrix(output_word2vec, output_lang, hidden_size)
    encoder = EncoderLSTM(input_embedding, hidden_size, dropout_p=dropout_p).to(device)
    decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words, output_embedding, dropout_p=dropout_p).to(device)
    train(train_dataloader, valid_dataloader, encoder, decoder, n_epochs, learning_rate, print_every=1, plot_every=1)
    encoder.load_state_dict(torch.load('encoder_best.pth'))
    decoder.load_state_dict(torch.load('decoder_best.pth'))
    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder, input_lang, output_lang)

    train_bleu_score, train_perplexity = calculate_bleu_score_and_perplexity(encoder, decoder, train_dataloader, input_lang, output_lang)
    valid_bleu_score, valid_perplexity = calculate_bleu_score_and_perplexity(encoder, decoder, valid_dataloader, input_lang, output_lang)
    test_bleu_score, test_perplexity = calculate_bleu_score_and_perplexity(encoder, decoder, test_dataloader, input_lang, output_lang)
    print(f"Train BLEU Score: {train_bleu_score:.4f}")
    print(f"Valid BLEU Score: {valid_bleu_score:.4f}")
    print(f"Test BLEU Score: {test_bleu_score:.4f}")
    print(f"Train Perplexity: {train_perplexity:.4f}")
    print(f"Valid Perplexity: {valid_perplexity:.4f}")
    print(f"Test Perplexity: {test_perplexity:.4f}")

