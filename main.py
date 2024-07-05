#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

# In[14]:


MAX_SEQ_LENGTH = 10
BATCH_SIZE = 64
num_epochs = 15
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 64
ENGLISH_VOCAB_SIZE = 25
FRENCH_VOCAB_SIZE = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(device)


# In[15]:


from data_utils import get_data, decode
dataloader = get_data(MAX_SEQ_LENGTH, False, BATCH_SIZE)


# In[16]:


class Encoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1):
		super(Encoder, self).__init__()
		self.embedding = torch.nn.Embedding(ENGLISH_VOCAB_SIZE, EMBEDDING_SIZE, dtype=torch.float32)
		self.rnn = torch.nn.RNN(EMBEDDING_SIZE, hidden_size, num_layers, bidirectional=True, batch_first=True)

	def forward(self, x):
		embedding = self.embedding(x)
		output, hn = self.rnn(embedding)
		return output, hn


# #### Alignment Model
# Although it can be done by concatenating both the annotation and prev hidden state, and making a single layered neural network, keeping them seperate helps understand why "weighting" said inputs are done.

# In[17]:


class AlignmentModel(torch.nn.Module):
	def __init__(self, hidden_size):
		super(AlignmentModel, self).__init__()
		self.W = torch.nn.Linear(hidden_size, hidden_size)
		self.U = torch.nn.Linear(hidden_size*2, hidden_size)
		self.V = torch.nn.Linear(hidden_size, 1)

	def forward(self, encoder_annotations, decoder_prev_hidden): # query, keys
		decoder_prev_hidden = decoder_prev_hidden
		output = self.W(decoder_prev_hidden.permute(1,0,2)) + self.U(encoder_annotations) # check notes 
		output = torch.tanh(output)
		scores = self.V(output) # batch, sequence, score (1)
		scores = scores.permute(0,2,1) # batch, score, sequence bc annotations is batch, sequence (1), hidden*2
		weights = torch.softmax(scores, dim=2)
		context = torch.bmm(weights, encoder_annotations)
		return context


# In[18]:


class Decoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1):
		super(Decoder, self).__init__()
		self.embedding = torch.nn.Embedding(FRENCH_VOCAB_SIZE, EMBEDDING_SIZE, dtype=torch.float32)
		self.rnn = torch.nn.RNN(EMBEDDING_SIZE+(2*hidden_size), hidden_size, num_layers, bidirectional=False, batch_first=True)
		self.linear = torch.nn.Linear(hidden_size, FRENCH_VOCAB_SIZE)
		self.alignment_model = AlignmentModel(hidden_size)

	def forward(self, encoder_outputs, encoder_hiddens):
		decoder_outputs = []
		decoder_hidden = encoder_hiddens[1].unsqueeze(0) # final hidden state of backwards
		decoder_input = torch.empty(BATCH_SIZE, 1, 1, dtype=torch.long).fill_(1).to(device) # batch size, seq len = 1, <EOS> TOKEN (using embedding layer so input is an integer)
		for t in range(MAX_SEQ_LENGTH):
			decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
			decoder_outputs.append(decoder_output)
			_, decoder_output = decoder_output.detach().topk(1, dim=2)
			decoder_input = decoder_output
		
		decoder_outputs = torch.concat(decoder_outputs, dim=1).squeeze(-1) # batch, seq len, token

		return decoder_outputs, decoder_hidden
	
	def forward_step(self, x, hn, encoder_annotations): # write this code
		embedding = self.embedding(x).squeeze(1)
		context = self.alignment_model(encoder_annotations, hn)
		decoder_input = torch.concat((embedding, context), dim=2) # batch, seq len = 1, both embedding and context
		decoder_output, decoder_hn = self.rnn(decoder_input, hn)
		decoder_output = self.linear(decoder_output)
		decoder_output = torch.softmax(decoder_output, dim=2) # batch, seq, pred
		return decoder_output, decoder_hn


# In[19]:


class EncoderDecoder(torch.nn.Module):
	def __init__(self, encoder, decoder):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
	
	def forward(self, x):
		encoder_outputs, encoder_hiddens = self.encoder(x)
		decoder_outputs, decoder_hiddens = self.decoder(encoder_outputs, encoder_hiddens)
		return decoder_outputs, decoder_hiddens


# In[20]:


model = EncoderDecoder(Encoder(1,128), Decoder(1,128)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[21]:


for epoch in range(num_epochs):

	running_loss = 0.0
	with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', ncols=100) as pbar:
		for i, data in enumerate(dataloader):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			outputs, hn = model(inputs)
			outputs=outputs.permute(0,2,1)
			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			pbar.set_postfix(loss=f'{running_loss/len(dataloader):.4f}')
			pbar.update(1)
	
	#print(f'\r {running_loss/len(dataloader)}', end='', flush=True)

print('\nFinished Training')


# In[22]:

num_samples=3 # > 1
with torch.no_grad():
	for i, data in enumerate(dataloader):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		outputs, hn = model(inputs)
		print(decode("english", inputs[:num_samples].squeeze().tolist()))
		print(decode("vietnamese", labels[:num_samples].squeeze().tolist()))
		_, outputs = outputs.detach().topk(1, dim=2)
		print(decode("vietnamese", outputs[:num_samples].squeeze().tolist()))
		print()
		break

# In[23]:


# need to account for init input for decoder
# have to control the index value for special tokens
# need to reverse labels


# In[24]:


# notes for cross entropy loss and data that has idx as target
# output should be the softmax output vector


# In[ ]:




