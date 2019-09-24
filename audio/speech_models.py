import torch
from torch import nn
import torch.nn.functional as F
from fastai import *
from fastai.vision import *
from fastai.layers import *


MAX_LENGTH = 1000


class SpeechModel(Module):
    "Contains an Encoder and a Decoder"
    def __init__(self,
                 encoder_output_size,
                 lang,
                 device=None,
                 teacher_forcing_pct=0.5):
        
        # select device
        self.device = device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                
        self.lang = lang
        self.vocab_size = len(lang)
        self.teacher_forcing_pct = teacher_forcing_pct
        
        self.encoder = CNNEncoder(encoder_output_size)
        self.embedding = nn.Embedding(self.vocab_size, encoder_output_size)
        self.pos_enc = PositionalEncoding(encoder_output_size, max_len=MAX_LENGTH)
        
        self.decoder = TransformerDecoder(encoder_output_size)
        self.ffn = nn.Linear(encoder_output_size, self.vocab_size)
        
        # put model on device
        if self.device is not "cpu":
            self.cuda()
    
    def forward(self, spectrogram, l=MAX_LENGTH):
        """
        spectrogram shape: [B x C x W x H]
        """
    
        spectrogram = spectrogram.to(self.device)
        batch_size = spectrogram.size(0)
        
        # run encoder to transform audio
        # to a vector in latent space
        context = self.encoder(spectrogram)
        print("context:", context.shape)
        
        # prepare input tokens for decoder
        sos = self.lang.t2i["<SOS>"]
        pad = self.lang.t2i["<PAD>"]
        
        # TODO: the input is all empty...?
        input_tokens = torch.LongTensor([[sos] + [pad] * (l-1)] * batch_size).to(self.device)
        print("input_tokens:", input_tokens.shape)
        
        # embed input token
        embedded = self.embedding(input_tokens)
            
        # apply positional encoding
        embedded_pos_enc = self.pos_enc(embedded)
        embedded_pos_enc = embedded_pos_enc.permute(1, 0, 2)
        print("embedded_pos_enc:", embedded_pos_enc.shape)
        
        # feed into decoder
        output_decoder = self.decoder(embedded_pos_enc, context)
        print("output_decoder:", output_decoder.shape)
        
        # ffn net
        output_tokens = self.ffn(output_decoder)
        
        # reshape output to [B x T x D]
        output_tokens = output_tokens.permute(1, 0, 2)
        print("output_tokens:", output_tokens.shape)
        return output_tokens
        
    def decode_greedy(self, spectrogram, **kwargs):
        output_tokens = self.forward(spectrogram, **kwargs)
        output_tokens = torch.log_softmax(output_tokens, dim=-1)
        
        
        with torch.no_grad():
            output_tokens = output_tokens.cpu().numpy()
            print(output_tokens)
            
            out = []
            for batch in output_tokens:

                sentence = ""
                for t in batch:
                    idx = t.argmax()
                    sentence += self.lang.i2t[idx]

                out.append(sentence)
            return out

    def predict(self):
        steps = 0
        while not tokens[-1] == eos and not steps > MAX_LENGTH:
            
            # NO teacher forcing here
            input_token = tokens[-1]
            
            # embed input token
            embedded = self.embedding(input_token)
            
            # apply positional encoding
            embedded = self.pos_enc(embedded)
            
            # feed to decoder
            decoder_output = self.decoder(embedded, context)
            decoder_output = self.linear(decoder_output)
            decoder_output_sm = F.log_softmax(decoder_output, dim=-1)
            
            print("decoder_output_sm:", decoder_output_sm.shape)
            _, decoder_output_token = decoder_output_sm.max(0)
            
            # append to already created tokens
            print("decoder_output_token:", decoder_output_token.shape)
            tokens.append(decoder_output_token)

        return tokens


class CNNEncoder(Module):
    "Encodes an audio spectrogram into a latent state representation"
    def __init__(self, output_size, base_model=models.resnet34):
        self.output_size = output_size
        self.model = create_cnn_model(base_model, output_size)
        
    def forward(self, spectrogram):
        context = self.model(spectrogram)
        return context
    
    
class TransformerDecoder(Module):
    "TransformerDecoder Module"
    def __init__(self, input_size, n_heads=2, num_layers=2):
        self.decoder_layer = nn.TransformerDecoderLayer(input_size, n_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.forward = self.decoder.forward

        
class PositionalEncoding(Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
