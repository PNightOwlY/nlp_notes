from torch.utils.data import Dataset


class TextClassificationDataSet(Dataset):
    """
    
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
        
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return {
            "input": inputs,
            "label": label
        }
        
        
        