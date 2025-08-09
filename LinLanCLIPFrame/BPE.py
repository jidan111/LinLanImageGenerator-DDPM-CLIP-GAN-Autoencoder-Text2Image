from .globals import *
from .utils import *


class BPE(object):
    def __init__(self, target_size=100, max_epoch=100):
        self.config = get_config(target_size=target_size, max_epoch=max_epoch)
        self.vocab = defaultdict(int)
        self.target_size = target_size
        self.max_epoch = max_epoch
        self.encode_embedding = {}
        self.decode_embedding = {}
        self.un_know = "</unk>"
        self.pad = "</null>"
        self.sentence_start = "</sos>"
        self.sentence_end = "</eos>"
        self.word_end = "</w>"

    def length(self):
        assert len(self.encode_embedding) > 0, "词表为空"
        return len(self.encode_embedding)

    def set_vocab(self, text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # 处理连续空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  # 更合理的字符过滤
        text = re.sub(r'[0-9]', '', text)
        words = []
        for char in text:
            if char.isspace():
                if words:  # 将连续单词保存
                    self.vocab[" ".join(words) + self.word_end] += 1
                    words = []
            elif '\u4e00' <= char <= '\u9fff':  # 中文字符
                if words:  # 处理中文前的单词
                    self.vocab[" ".join(words) + self.word_end] += 1
                    words = []
                self.vocab[char + self.word_end] += 1  # 每个中文字符单独处理
            else:
                words.append(char)
        if words:  # 处理最后一个单词
            self.vocab[" ".join(words) + self.word_end] += 1

    def get_best_pair(self):
        pair_counts = defaultdict(int)
        for token_seq, freq in self.vocab.items():
            tokens = token_seq.split()
            for i in range(len(tokens) - 1):
                if tokens[i].endswith(self.word_end):
                    continue
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
        if not pair_counts:
            return None
        best_pair = max(pair_counts, key=pair_counts.get)
        return best_pair, pair_counts[best_pair]

    def update_vocab(self, best_pair):
        if not best_pair:
            return
        new_vocab = defaultdict(int)
        new_token = f"{best_pair[0]}{best_pair[1]}"
        for token_seq, freq in self.vocab.items():
            tokens = token_seq.split()
            new_seq = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(tokens[i])
                    i += 1
            new_vocab[" ".join(new_seq)] += freq
        self.vocab = new_vocab

    def set_embedding_map(self, max_length=8, min_length=1, add_char=True):
        assert self.vocab, "词表为空, vocab is None"
        if len(self.encode_embedding) != 0:
            self.encode_embedding = {}
        all_tokens = set()
        for token_seq in self.vocab:
            seq = [i for i in token_seq.split() if min_length < len(i) <= max_length]
            all_tokens.update(seq)
        if add_char:
            all_tokens = [self.pad, self.un_know, self.sentence_start, self.sentence_end, self.word_end] + list(
                "0123456789abcdefghijklmnopqrstuvwxyz~!@#$%^&*()_+=-[]{};':\"<>,./?\\") + sorted(list(all_tokens))
        else:
            all_tokens = [self.pad, self.un_know, self.sentence_start, self.sentence_end, self.word_end] + sorted(
                list(all_tokens))
        for idx, token in enumerate(all_tokens):  # 排序保证确定性
            self.encode_embedding[token] = idx
            self.decode_embedding[idx] = token

    def fit(self, text, continue_train=False, min_freq=4, max_length=8, min_length=1, add_char=True, stop_cnt=50):
        if not continue_train:
            self.set_vocab(text)
        stop_pair = [None, 0]
        for epoch in tqdm(range(self.max_epoch)):
            pair_item = self.get_best_pair()
            if not pair_item or len(self.encode_embedding) >= self.target_size:
                break
            if pair_item[1] < min_freq:
                break
            self.update_vocab(pair_item[0])
            if stop_pair[0] == pair_item[0]:
                if stop_pair[1] == stop_cnt:
                    break
                else:
                    stop_pair[1] += 1
            else:
                stop_pair[0] = pair_item[0]
                stop_pair[1] = 1
        self.set_embedding_map(max_length=max_length, min_length=min_length, add_char=add_char)
        print(f"Final vocab size: {len(self.encode_embedding)}")

    def encode_word(self, x):
        assert len(self.encode_embedding) != 0, "编码本为空"
        out = []
        if not x.endswith(self.word_end):
            x = x + self.word_end
        left = 0
        length = len(x)
        while left < length:
            right = length
            while True:
                s = x[left:right]
                if s in self.encode_embedding:
                    out.append(self.encode_embedding[s])
                    left = right
                    break
                else:
                    if right == left + 1:
                        out.append(self.encode_embedding[self.un_know])
                        left += 1
                        break
                    else:
                        right -= 1
        return out

    def encode_sentence(self, txt, dim=64, to_tensor=False, dim_judge=False):
        dim -= 1
        out = [self.encode_embedding[self.sentence_start]]
        for i in txt.lower().split():
            out += self.encode_word(i)
        if len(out) < dim:
            out = out + [self.encode_embedding[self.pad]] * (dim - len(out))
        else:
            if dim_judge:
                print("出现截断token现象")
            out = out[:dim]
        out.append(self.encode_embedding[self.sentence_end])
        if to_tensor:
            return torch.Tensor(out).to(torch.long)
        return out

    def encode_sentences_to_tensor(self, texts, dim=64, dim_judge=False):
        arr = [self.encode_sentence(i, dim=dim, dim_judge=dim_judge) for i in texts]
        arr = torch.Tensor(arr).to(torch.long)
        return arr

    def decode(self, arr):
        assert len(self.decode_embedding) != 0, "解码本为空"
        out = ""
        for i in arr:
            i = int(i)
            out += self.decode_embedding[i]
        return out.replace(self.word_end, " ").replace(self.sentence_start, "").replace(self.sentence_end, "").replace(
            self.pad, "")

    def decode_sentences(self, arr):
        arr = [self.decode(i) for i in arr]
        return arr

    def save(self, path, save_vocab=False):
        with open(path, 'w', encoding="utf-8") as f:
            if save_vocab:
                json.dump({"vocab": self.vocab, "embed": self.encode_embedding}, f)
            else:
                json.dump(self.encode_embedding, f)

    def load(self, file_name):
        with open(file_name, 'r', encoding="utf-8") as f:
            content = json.load(f)
        if "vocab" in content:
            self.encode_embedding = content["embed"]
            self.vocab = content["vocab"]
        else:
            self.encode_embedding = content
        for k, v in self.encode_embedding.items():
            self.decode_embedding[v] = k
