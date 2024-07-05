from torch.utils.data import Dataset, DataLoader
import torch

# using pytorch dataset and dataloader can be helpful
class EN_VN_dataset(Dataset):
	def __init__(self, lang1, lang2):
		self.lang1 = lang1
		self.lang2 = lang2

	def __len__(self):
		return self.lang1.size(0)

	def __getitem__(self, index):
		sentence1 = self.lang1[index]
		sentence2 = self.lang2[index]
		return sentence1, sentence2
	
def get_data(MAX_SEQ_LEN, shuffle, BATCH_SIZE):
	# to keep thing simple lets use one hot encoding representations for our sequential data
	# we are going to split the data X, Y
	data = [["i like to eat fish", "toi thit an ca"],
			["have you ate yet", "co an com chua"],
			["we are going to church tomorrow", "ngay may minh di le"]]

	# most tokenizers add special tokens such as <SOS>, <EOS>, <SEP>, <MASK>, etc.
	data = [["<SOS> "+sentence+" <EOS>" for sentence in entry] for entry in data]

	# padding done as at this level as well
	MAX_SEQ_LEN = 10
	def pad(text):
		while len(text.split(" ")) < MAX_SEQ_LEN:
			text += " <PAD>"
		return text
	data = [[pad(sentence) for sentence in entry] for entry in data]

	english_dict = {}
	idx = 1
	for entry in data:
		tokens = entry[0].split(" ")
		for token in tokens:

			if token not in english_dict:
				english_dict[token] = idx
				idx += 1

	idx = 1
	vietnamese_dict = {}
	for entry in data:
		tokens = entry[1].split(" ")
		for token in tokens:

			if token not in vietnamese_dict:
				vietnamese_dict[token] = idx
				idx += 1

	# convert tokens to integer index
	english = []
	for entry in data:
		tokens = entry[0].split(" ")
		idxs = []
		for token in tokens:
			idxs.append(english_dict[token])
		english.append(idxs)

	vietnamese = []
	for entry in data:
		tokens = entry[1].split(" ")
		idxs = []
		for token in tokens:
			idxs.append(vietnamese_dict[token])
		vietnamese.append(idxs)

	reps = 1000
	vietnamese = torch.tensor(vietnamese*reps)
	english = torch.tensor(english*reps)

	return DataLoader(EN_VN_dataset(english, vietnamese), shuffle=shuffle, batch_size=BATCH_SIZE, drop_last=True)

def decode(lang, input):
	# decode input to lang
	# input is a single dimensional list
	data = [["i like to eat fish", "toi thit an ca"],
			["have you ate yet", "co an com chua"],
			["we are going to church tomorrow", "ngay may minh di le"]]

	data = [["<SOS> "+sentence+" <EOS>" for sentence in entry] for entry in data]

	MAX_SEQ_LEN = 10
	def pad(text):
		while len(text.split(" ")) < MAX_SEQ_LEN:
			text += " <PAD>"
		return text
	data = [[pad(sentence) for sentence in entry] for entry in data]

	english_dict = {}
	idx = 1
	for entry in data:
		tokens = entry[0].split(" ")
		for token in tokens:

			if token not in english_dict:
				english_dict[token] = idx
				idx += 1

	idx = 1
	vietnamese_dict = {}
	for entry in data:
		tokens = entry[1].split(" ")
		for token in tokens:

			if token not in vietnamese_dict:
				vietnamese_dict[token] = idx
				idx += 1

	english_dict = {v: k for k, v in english_dict.items()}
	vietnamese_dict = {v: k for k, v in vietnamese_dict.items()}

	batch = []
	for entry in input:
		output = []
		for idx in entry:
			if lang == "english":
				lang_dict = english_dict
			elif lang == "vietnamese":
				lang_dict = vietnamese_dict
		
			output.append(lang_dict[idx])
		batch.append(" ".join(output))
	return batch
