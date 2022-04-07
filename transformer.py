import io
from typing import List
from xmlrpc.client import Boolean
from numpy import number
import torch
from torch import nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader
import copy

# custom util transformer entity
# CUTE

class Utils():
    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokenizer = get_tokenizer("moses")
        result = tokenizer(text)
        result.append("<eos>")
        return result

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

# empty class so that pylance works
class ParameterProvider():
    pass

class ParameterProvider():
    def __init__(self, configname):
        if configname:
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
                "learning_rate": float(self.config['TRAINING PARAMETERS']['learning_rate']),
                "epochs": int(self.config['TRAINING PARAMETERS']['epochs'])
            }
        else:
            self.dictionary = {
                "d_model": 0,
                "d_qk": 0,
                "d_v": 0,
                "d_ff": 0,
                "n_encoders": 0,
                "n_decoders": 0,
                "n_heads": 0,
                "language_in_file": "",
                "language_out_file": "",
                "vocab_in_size": 0,
                "vocab_out_size": 0,
                "learning_rate": 0.0,
                "epochs": 0
            }
    
    def modifyWithArray(self, arr: List) -> None:
        self.dictionary = {
            "d_model": arr[0],
            "d_qk": arr[1],
            "d_v": arr[2],
            "d_ff": arr[3],
            "n_encoders": arr[4],
            "n_decoders": arr[5],
            "n_heads": arr[6],
            "epochs": arr[7],
            "learning_rate": self.dictionary["learning_rate"],
            "language_in_file": self.dictionary["language_in_file"],
            "language_out_file": self.dictionary["language_out_file"],
            "vocab_in_size": self.dictionary["vocab_in_size"],
            "vocab_out_size": self.dictionary["vocab_out_size"],
        }

    def getArray(self) -> List:
        return [
            self.dictionary["d_model"],
            self.dictionary["d_qk"],
            self.dictionary["d_v"],
            self.dictionary["d_ff"],
            self.dictionary["n_encoders"],
            self.dictionary["n_decoders"],
            self.dictionary["n_heads"],
            self.dictionary["epochs"],
        ]

    def provide(self, key: str) -> any:
        return self.dictionary[key]
        
    def change(self, key: str, value: number) -> None:
        self.dictionary[key] = value

    def getChangedCopy(self,key: str, value:number) -> ParameterProvider:
        pp = copy.deepcopy(self)
        pp.change(key,value)
        return pp


class VocabProvider():
    def __init__(self, config: ParameterProvider, vocabSourceFile: str):
        self.vocab = build_vocab_from_iterator(Utils.yield_tokens(vocabSourceFile), specials=["<unk>"])
        self.vocab.append_token("<eos>")
        self.vocab.set_default_index(0)
    
    def getVocabLength(self) -> int:
        return len(self.vocab)

    def getValues(self, text):
        return torch.tensor(self.vocab(text))

class Embedder(nn.Module):
    def __init__(self, config: ParameterProvider, vocab: VocabProvider, use_string_input: Boolean = False):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.embedding = nn.Embedding(vocab.getVocabLength(), self.d_model)
        self.vocab = vocab
        self.use_string_input = use_string_input
    
    def forward(self, text: any) -> Tensor:
        if self.use_string_input:
            tokenized_str = Utils.tokenize(text)
            if len(tokenized_str) == 0:
                return torch.empty(self.dimension)
            else:
                return self.embedding(self.vocab.getValues(tokenized_str))
        else:
            return self.embedding(text)

class AttentionHead(nn.Module):
    def __init__(self, config: ParameterProvider, masked: bool = False, d_v_override: int = None, d_qk_override: int = None):
        super().__init__()
        self.l = nn.Linear(100,200)
        self.masked = masked
        self.d_model = config.provide("d_model")
        self.d_qk = config.provide("d_qk") if d_qk_override is None else d_qk_override
        self.d_v = config.provide("d_v") if d_v_override is None else d_v_override
        self.WQ = nn.Linear(self.d_model,self.d_qk)
        self.WK = nn.Linear(self.d_model,self.d_qk)
        self.WV = nn.Linear(self.d_model,self.d_v)
    
    def forward(self,input_q: Tensor, input_k: Tensor, input_v: Tensor) -> Tensor:
        Q = self.WQ(input_q)
        K = self.WK(input_k)
        V = self.WV(input_v)

        if self.masked:
            if Q.size(1) != K.size(1):
                raise TypeError('Masking can be only performed when Querry and Key Matrices have the same sizes (i.e. their product is square)')
            mask = torch.stack([torch.triu(torch.full((Q.size(1),Q.size(1)),-1*torch.inf),diagonal=1) for _ in range(Q.size(0))])
            mask = mask.to(device=Q.device)
            return torch.bmm(torch.softmax(torch.add(torch.bmm(Q,torch.transpose(K,1,2)),mask)/math.sqrt(self.d_model),dim=-1),V)
        else:
            return torch.bmm(torch.softmax(torch.bmm(Q,torch.transpose(K,1,2))/math.sqrt(self.d_model),dim=-1),V)


class MultiHeadedAttention(nn.Module):
    def __init__(self,config: ParameterProvider, masked = False, d_v_override = None, d_qk_ovveride: int = None, n_heads_override = None):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_v = config.provide("d_v") if d_v_override is None else d_v_override
        self.d_qk = config.provide("d_qk") if d_qk_ovveride is None else d_qk_ovveride
        self.n_heads = config.provide("n_heads") if n_heads_override is None else n_heads_override
        self.heads = nn.ModuleList([AttentionHead(config,masked=masked,d_qk_override=d_qk_ovveride,d_v_override=d_v_override) for _ in range(self.n_heads)])
        self.linear = nn.Linear(self.d_v*self.n_heads,self.d_model)
        self.masked = masked
    
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
        self.encoders = nn.ModuleList([EncoderLayer(config) for _ in range(self.n_encoders)])
    
    def forward(self, input_data: Tensor) -> Tensor:
        for l in self.encoders:
            input_data = l(input_data)
        return input_data

class NumericalOut(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("d_model"),config.provide("vocab_out_size"))
    
    def forward(self, input_data: Tensor) -> Tensor:
        #return torch.softmax(self.linear(input_data), dim = -1)
        return self.linear(input_data)

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
        self.decoders = nn.ModuleList([DecoderLayer(config) for _ in range(self.n_decoders)])
    
    def forward(self, input_data: Tensor, ed_data: Tensor) -> Tensor:
        for l in self.decoders:
            input_data = l(input_data,ed_data)
        return input_data

class Transformer(nn.Module):
    def __init__(self, config: ParameterProvider, vocab_in: VocabProvider, vocab_out: VocabProvider, use_string_input = False):
        super().__init__()
        self.config = config
        self.d_model = config.provide("d_model")
        self.vocab_in = vocab_in
        config.change("vocab_in_size",self.vocab_in.getVocabLength())        
        self.embedding_in = Embedder(config,self.vocab_in,use_string_input)
        self.use_string_input = use_string_input

        self.vocab_out = vocab_out
        config.change("vocab_out_size",self.vocab_out.getVocabLength())
        self.embedding_out = Embedder(config,self.vocab_out,use_string_input)

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
    def __init__(self, infile: str, outfile: str, vocab_in: VocabProvider, vocab_out: VocabProvider):

        self.vocab_in = vocab_in
        self.vocab_out = vocab_out

        fin = open(infile, "r",encoding = 'utf-8')
        fout = open(outfile,"r",encoding = 'utf-8')
        Xlines = list(map(str.lower,fin.readlines()))
        Ylines = list(map(str.lower,fout.readlines()))

        self.X = [self.vocab_in.getValues(Utils.tokenize(l[:-1])) for l in Xlines]
        self.Y = [self.vocab_out.getValues(Utils.tokenize(l[:-1])) for l in Ylines]
        self.LX = [len(l) for l in self.X]
        self.LY = [len(l) for l in self.Y]
        if len(self.X) != len(self.Y):
            raise Exception('Sets are of different sizes')
        self.Z = []
        for y in self.Y:
            z = torch.zeros((len(y),vocab_out.getVocabLength()))
            for i in range(len(y)):
                z[i][int(y[i])] = 1.0
            self.Z.append(z)        

        self.X = torch.nn.utils.rnn.pad_sequence(self.X,batch_first=True)
        self.Y = torch.nn.utils.rnn.pad_sequence(self.Y,batch_first=True)
        self.Z = torch.nn.utils.rnn.pad_sequence(self.Z,batch_first=True)

        fin.close()
        fout.close()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        _z = self.Z[index]
        _lx = self.LX[index]
        _ly = self.LY[index]
        return _x, _y, _z, _lx, _ly

    def getSets(self):
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])



def train_cuda(model: nn.Module, train_dataset: CustomDataSet, device: int, batch_size = 32, lr: float = 0.1, epochs: int = 1) -> None:
    
    model.cuda(device=device)
    criterion = nn.CrossEntropyLoss(reduction="sum").cuda(device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,)

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    last_loss = 0.
    for epoch in range(epochs):

        for i, (data_in, data_out, data_out_numeric, _, len_out) in enumerate(data_loader):
            data_in = data_in.cuda(device)
            data_out = data_out.cuda(device)
            data_out_numeric = data_out_numeric.cuda(device)
            optimizer.zero_grad()
            output = model(data_in, data_out)
            loss = criterion(output,data_out_numeric)
            loss.backward()
            optimizer.step()

            last_loss = loss.item() / sum(len_out)

    return last_loss



def evaluate(model: nn.Module, test_dataset: CustomDataSet, use_cuda: Boolean = False, device: int = 0, batch_size = 32) -> float:

    criterion = nn.CrossEntropyLoss(reduction='sum')
    if use_cuda:
        criterion = nn.CrossEntropyLoss(reduction='sum').cuda(device)
        model.cuda(device)    
    
    model.eval()
    total_loss = 0.

    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    

    with torch.no_grad():
        for i, (data_in, data_out, data_out_numeric, _, len_out) in enumerate(data_loader):
            if use_cuda:
                data_in = data_in.cuda(device)
                data_out = data_out.cuda(device)
                data_out_numeric = data_out_numeric.cuda(device)
            output = model(data_in, data_out)
            total_loss += criterion(output, data_out_numeric).item() / sum(len_out)
    return total_loss / (len(test_dataset))

def train(model: nn.Module, train_dataset: CustomDataSet, lr: float = 0.1, epochs: int = 1) -> None:

    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.
    for _ in range(epochs):
        for value in train_dataset:
            data_in = value[0]
            data_out = value[1]
            data_out_numeric = value[2]


            optimizer.zero_grad()
            output = model(data_in,data_out)
            loss = criterion(output,data_out_numeric)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / epochs
    
    return total_loss

def train_until_difference(model: nn.Module, train_dataset: CustomDataSet, min_difference = 0.001, lr: float = 0.1, max_epochs: int = 50, criterion = torch.nn.CrossEntropyLoss()) -> float:
    
    new_result = evaluate(model,train_dataset)
    for i in range(0,max_epochs):
        old_result = new_result
        train(model,train_dataset,lr,1)
        new_result = evaluate(model,train_dataset)
        difference = (old_result - new_result) / old_result
        if abs(difference) < min_difference:
            return new_result
    return new_result

def train_until_difference_cuda(model: nn.Module, train_dataset: CustomDataSet, min_difference = 0.001, device: int = 0, batch_size = 32,  lr: float = 0.1, max_epochs: int = 50, criterion = torch.nn.CrossEntropyLoss(reduction='sum')) -> float:
    result_epochs = 0
    new_result = evaluate(model,train_dataset, use_cuda=True, device=device, batch_size=batch_size)
    for i in range(0,max_epochs):
        result_epochs += 1
        old_result = new_result
        new_result = train_cuda(model,train_dataset,lr=lr,epochs=1,batch_size=batch_size,device=device)
        #new_result = evaluate(model,train_dataset,use_cuda=True,device=device,batch_size=batch_size)
        difference = (old_result - new_result) / old_result
        if abs(difference) < min_difference:
            return new_result, result_epochs

    return new_result, result_epochs