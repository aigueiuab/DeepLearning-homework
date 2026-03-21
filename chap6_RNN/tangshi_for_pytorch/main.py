import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn as rnn_lstm   # 修正：将模块别名统一为 rnn_lstm

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """
    :param file_name:
    :return: poems_vector  have two dimensions, first is the poem, the second is the word_index
    """
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError:
                pass

    poems = sorted(poems, key=lambda line: len(line))

    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def process_poems2(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' ', '').replace('，', '').replace('。', '')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    content = start_token + content + end_token
                    poems.append(content)
            except ValueError:
                pass

    poems = sorted(poems, key=lambda line: len(line))
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y = row[1:]
            y.append(row[-1])
            y_data.append(y)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    print("finish loading data, vocab size:", len(word_to_int))

    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_emb = rnn_lstm.word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=BATCH_SIZE,
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_emb,
        embedding_dim=100,
        lstm_hidden_dim=128
    )

    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss()

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype=np.int64)
                y = np.array(batch_y[index], dtype=np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x, axis=1)))
                y = Variable(torch.from_numpy(y))
                pre = rnn_model(x)
                loss += loss_fun(pre, y)
                if index == 0:
                    _, pre_idx = torch.max(pre, dim=1)
                    print('prediction', pre_idx.data.tolist())
                    print('b_y       ', y.data.tolist())
                    print('*' * 30)

            loss = loss / BATCH_SIZE
            print("epoch", epoch, 'batch number', batch, "loss is:", loss.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish save model")


def to_word(predict, vocabs):
    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):
    # 去掉起始符和终止符后打印
    result = []
    for w in poem:
        if w == start_token:
            continue
        if w == end_token:
            break
        result.append(w)
    poem_str = ''.join(result)
    poem_sentences = poem_str.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 3:
            print(s + '。')


def gen_poem(begin_word):
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_emb = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=64,
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_emb,
        embedding_dim=100,
        lstm_hidden_dim=128
    )
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn', map_location='cpu'))
    rnn_model.eval()

    poem = begin_word
    word = begin_word
    with torch.no_grad():
        while word != end_token:
            # 将当前诗句转换为索引序列，逐字输入
            input_indices = []
            for w in poem:
                if w in word_int_map:
                    input_indices.append(word_int_map[w])
                else:
                    input_indices.append(len(word_int_map))  # UNK
            x = np.array(input_indices, dtype=np.int64)
            x = Variable(torch.from_numpy(x))
            output = rnn_model(x, is_test=True)
            word = to_word(output.data.tolist()[0], vocabularies)
            poem += word
            if len(poem) > 50:
                break
    return poem


if __name__ == '__main__':
    # 训练模型（首次运行）
    # run_training()

    # 生成诗歌
    print("\n===== 生成诗歌 =====")
    for begin in ['日', '红', '山', '夜', '湖', '海', '月']:
        print(f"\n--- 以「{begin}」开头 ---")
        pretty_print_poem(gen_poem(begin))