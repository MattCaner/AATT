from base64 import encode
from hmac import trans_36
import io
from typing import List
from numpy import number
from sqlalchemy import false
from sympy import numer
import torch
import torch.nn.functional as f
from torch import inner, nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader

# custom util transformer entity
# CUTE

class Utils():
    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokenizer = get_tokenizer("moses")
        return tokenizer(text)

    @staticmethod
    def yield_tokens(file_path):
        with io.open(file_path, encoding = 'utf-8') as f:
            for line in f:
                yield map(str.lower, get_tokenizer("moses")(line))
    
    @staticmethod
    def encode_position(seq_len: int, dim_model: int, highValue = 1e4, device: torch.device = torch.device("cpu")) -> Tensor:
        pos = torch.arange(seq_len,dtype=torch.float,device=device).reshape(1,-1,1)
        dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1,1,-1)
        phase = pos / highValue ** (torch.div(dim, dim_model, rounding_mode='floor'))
        return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

class ParameterProvider():
    def __init__(self, configname):
        self.config = configparser.ConfigParser()
        self.config.read(configname)
        self.dictionary = {
            "d_model": int(self.config['DIMENSIONS AND SIZES']['d_model']),
            "d_qk": int(self.config['DIMENSIONS AND SIZES']['d_qk']),
            "d_v": int(self.config['DIMENSIONS AND SIZES']['d_v']),
            "d_ff": int(self.config['DIMENSIONS AND SIZES']['d_ff']),
            "n_encoders": int(self.config['DIMENSIONS AND SIZES']['n_encoders']),
            "n_decoders": int(self.config['DIMENSIONS AND SIZES']['n_decoders']),
            "n_heads": int(self.config['DIMENSIONS AND SIZES']['n_heads']),
            "language_in_file": self.config['VOCAB PARAMETERS']['language_in_file'],
            "language_out_file": self.config['VOCAB PARAMETERS']['language_out_file'],
            "vocab_in_size": 0,
            "vocab_out_size": 0,
        }

    def provide(self, key: str):
        return self.dictionary[key]
        
    def change(self, key: str, value: number):
        self.dictionary[key] = value




class VocabProvider():
    def __init__(self, config: ParameterProvider, vocabSourceFile: str):
        self.vocab = build_vocab_from_iterator(Utils.yield_tokens(config.provide(vocabSourceFile)), specials=["<unk>"])
        self.vocab.append_token("<eos>")
        self.vocab.set_default_index(0)
    
    def getVocabLength(self) -> int:
        return len(self.vocab)

    def getValues(self, text):
        return torch.tensor(self.vocab(text))

class Embedder(nn.Module):
    def __init__(self, config: ParameterProvider, vocab: VocabProvider):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.embedding = nn.Embedding(vocab.getVocabLength(), self.d_model)
        self.vocab = vocab
    
    def forward(self, text: str) -> Tensor:
        tokenized_str = Utils.tokenize(text)
        if len(tokenized_str) == 0:
            return torch.empty(self.dimension)
        else:
            return self.embedding(self.vocab.getValues(tokenized_str))
'''
class Embedding(nn.Module):
    def __init__(self, vocab: Vocab, dE: int):
        super().__init__()
        self.vocab = vocab
        self.dimension = dE
        self.embedding = nn.Embedding(len(vocab), dE)
    
    def forward(self, tokenized_str) -> Tensor:
        if len(tokenized_str) == 0:
            return torch.empty(self.dimension)
        else:
            return self.embedding(torch.tensor(self.vocab(tokenized_str)))
'''
class AttentionHead(nn.Module):
    def __init__(self, config: ParameterProvider, masked = false):
        super().__init__()
        self.l = nn.Linear(100,200)
        self.masked = masked
        self.d_model = config.provide("d_model")
        self.d_qk = config.provide("d_qk")
        self.d_v = config.provide("d_v")
        self.WQ = nn.Linear(self.d_model,self.d_qk)
        self.WK = nn.Linear(self.d_model,self.d_qk)
        self.WV = nn.Linear(self.d_model,self.d_v)
    
    def forward(self,input_q: Tensor, input_k: Tensor, input_v: Tensor) -> Tensor:
        Q = self.WQ(input_q)
        K = self.WK(input_k)
        V = self.WV(input_v)

        if self.masked:
            if Q.size(0) != K.size(0):
                raise TypeError('Masking can be only performed when Querry and Key Matrices have the same sizes (i.e. their product is square)')
            mask = torch.triu(torch.full((Q.size(0),Q.size(0)),-1*torch.inf),diagonal=1)
            return torch.matmul(torch.softmax(torch.add(torch.matmul(Q,torch.transpose(K,0,1)),mask)/math.sqrt(self.d_model),dim=-1),V)
        else:
            return torch.matmul(torch.softmax(torch.matmul(Q,torch.transpose(K,0,1))/math.sqrt(self.d_model),dim=-1),V)


class MultiHeadedAttention(nn.Module):
    def __init__(self,config: ParameterProvider, masked = False):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_v = config.provide("d_v")
        self.n_heads = config.provide("n_heads")
        self.heads = [AttentionHead(config,masked=masked) for _ in range(self.n_heads)]
        self.linear = nn.Linear(self.d_v*self.n_heads,self.d_model)
    
    def forward(self, input_q, input_k, input_v):
        concatResult = torch.cat([h(input_q, input_k, input_v) for h in self.heads], dim = -1)
        return self.linear(concatResult)


class EncoderLayer(nn.Module):
    def __init__(self,config: ParameterProvider):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_ff = config.provide("d_ff")
        self.feed_forward = nn.Sequential(nn.Linear(self.d_model,self.d_ff),nn.ReLU(),nn.Linear(self.d_ff,self.d_model))
        self.mha = MultiHeadedAttention(config)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self,input_data: Tensor) -> Tensor:
        mha_output = self.mha(input_data,input_data,input_data)
        intermediate = self.norm(torch.add(mha_output,input_data))
        return self.norm(torch.add(intermediate,self.feed_forward(intermediate)))

class DecoderLayer(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_ff = config.provide("d_ff")
        self.feed_forward = nn.Sequential(nn.Linear(self.d_model,self.d_ff),nn.ReLU(),nn.Linear(self.d_ff,self.d_model))
        self.self_mha = MultiHeadedAttention(config, masked = True)
        self.ed_mha = MultiHeadedAttention(config)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self,input_data: Tensor, encoder_data: Tensor) -> Tensor:
        mha_output = self.self_mha(input_data,input_data,input_data)
        intermediate = self.norm(torch.add(mha_output,input_data))
        mha_output = self.ed_mha(input_data,encoder_data,encoder_data)
        intermediate = self.norm(torch.add(mha_output,intermediate))
        return self.norm(torch.add(intermediate,self.feed_forward(intermediate)))

class EncoderStack(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.n_encoders = config.provide("n_encoders")
        self.encoders = [EncoderLayer(config) for _ in range(self.n_encoders)]
    
    def forward(self, input_data: Tensor) -> Tensor:
        for l in self.encoders:
            input_data = l(input_data)
        return input_data

class NumericalOut(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("d_model"),config.provide("vocab_out_size"))
    
    def forward(self, input_data: Tensor) -> Tensor:
        return torch.softmax(self.linear(input_data), dim = -1)

class LexicalOut(nn.Module):
    def __init__(self, config: ParameterProvider, vocab: VocabProvider):
        super().__init__()
        self.vocab = vocab
    
    def forward(self, input_data: Tensor) -> List[str]:
        result = []
        for i in input_data:
            last_result = self.vocab.vocab.lookup_token(torch.argmax(i))
            result.append(last_result)
        return result

class DecoderStack(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.n_decoders = config.provide("n_decoders")
        self.decoders = [DecoderLayer(config) for _ in range(self.n_decoders)]
    
    def forward(self, input_data: Tensor, ed_data: Tensor) -> Tensor:
        for l in self.decoders:
            input_data = l(input_data,ed_data)
        return input_data

class Transformer(nn.Module):
    def __init__(self, config: ParameterProvider, vocab_in: VocabProvider, vocab_out: VocabProvider):
        super().__init__()
        self.vocab_in = vocab_in 
        config.change("vocab_in_size",self.vocab_in.getVocabLength())        
        self.embedding_in = Embedder(config,self.vocab_in)

        self.vocab_out = vocab_out
        config.change("vocab_out_size",self.vocab_out.getVocabLength())
        self.embedding_out = Embedder(config,self.vocab_out)

        self.encoder_stack = EncoderStack(config)
        self.decoder_stack = DecoderStack(config)

        self.numerical_out = NumericalOut(config)
        self.lexical_out = LexicalOut(config, self.vocab_out)

    def forward(self, encoder_input: str, decoder_input: str):
        in_embedded = self.embedding_in(encoder_input)

        encoder_out = self.encoder_stack(in_embedded)

        out_embedded = self.embedding_out(decoder_input)
        decoder_out = self.decoder_stack(out_embedded,encoder_out)
        numerical = self.numerical_out(decoder_out)
        return numerical
        #return self.lexical_out(numerical)

class CustomDataSet(Dataset):
    def __init__(self, infile: str, outfile: str, vocab: VocabProvider):

        self.vocab = vocab

        fin = open(infile, "r",encoding = 'utf-8')
        fout = open(outfile,"r",encoding = 'utf-8')
        self.X = list(map(str.lower,fin.readlines()))
        self.Y = list(map(str.lower,fout.readlines()))
        if len(self.X) != len(self.Y):
            raise Exception('Sets are of different sizes')
        fin.close()
        fout.close()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        tokenized_str = Utils.tokenize(_y)
        _z = torch.zeros((len(tokenized_str),self.vocab.getVocabLength()))
        values = self.vocab.getValues(tokenized_str)
        for i in range(len(tokenized_str)):
            _z[i][int(values[i])] = 1.0
        
        return _x, _y, _z



params = ParameterProvider("params.config")


v_in = VocabProvider(params,"language_in_file")
v_out = VocabProvider(params,"language_out_file")
t = Transformer(params, v_in, v_out)

cd = CustomDataSet('simplepl.txt', 'simpleen.txt',v_out)


train_size = int(0.8 * len(cd))
test_size = len(cd) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cd, [train_size, test_size])

loader = iter(DataLoader(train_dataset, batch_size=1, shuffle=True))



out = t("skończmy to jak najszybciej zamiast to przeciągać","let's get it over with as soon as possible rather than drag it out")

criterion = torch.nn.CrossEntropyLoss()

lr = 1.0
optimizer = torch.optim.SGD(t.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs = 100

def train(model: nn.Module) -> None:

    total_loss = 0.
    for i, value in enumerate(train_dataset):
        data_in = value[0]
        data_out = value[1]
        data_out_numeric = value[2]

        output = model(data_in,data_out)
        loss = criterion(output,data_out_numeric)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


def evaluate(model: nn.Module) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i, value in enumerate(test_dataset):
            data_in = value[0]
            data_out = value[1]
            data_out_numeric = value[2]
            output = model(data_in, data_out)
            total_loss += criterion(output, data_out_numeric).item()
    return total_loss / (len(train_dataset) - 1)


for i in range(0, 100):

    train(t)
    ev = evaluate(t)
    print(f' Epoch: {i}, eval: {ev:f}')


