from torch.utils.data import Dataset


class TextClassificationDataSet(Dataset):
    """
    Load dataset with List[Dict{"text": text, "label", label}]
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
                 
        return text, label
        
        
        