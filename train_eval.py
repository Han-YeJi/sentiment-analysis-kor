
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class Trainer:
    def __init__(self, args, loaders, model):
        self.args = args
        self.model = model
        self.loaders = loaders
        
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def train(self, data_loader): 
        epoch_loss, acc, total = 0, 0, 0
        self.model.train() # train mode

        for i, data in enumerate(data_loader): # index, data # 위에 출력한거 확인! i=train loader의 길이-1
            text = data.text
            label = data.label

            out = self.model(text).squeeze(1) #뒤에 1차원이 붙어서 그걸 없애기!
            loss = self.criterion(out.float(), label.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            acc += (torch.round(out)==label).sum()
            epoch_loss += loss.item()

        return epoch_loss/(i+1), acc / len(data_loader.dataset) * 100

    def evaluate(self, data_loader):
        epoch_loss, acc = 0, 0
        self.model.eval() # evaluation mode

        with torch.no_grad(): #backprop 기능 끄기
            for data in data_loader: # enumerate안해도 가능! 
                text = data.text
                label = data.label

                out = self.model(text).squeeze(1)
                loss = self.criterion(out.float(), label.float())

                acc += (torch.round(out)==label).sum()
                epoch_loss += loss.item()

        return epoch_loss/len(data_loader), acc / len(data_loader.dataset) * 100

    def predict(self, data_loader,root,predict_path):
        df = pd.DataFrame(columns=["Id", "Predicted"])
        with torch.no_grad(): #backprop 기능 끄기
            
            for data in data_loader: # enumerate안해도 가능!
                id = data.id 
                text = data.text
                # predict
                out = self.model(text).squeeze(1)
                out=torch.round(out)
                out=out.int()
                
                batch_df = pd.DataFrame({"Id" : id.cpu(), "Predicted" : out.cpu()})
                df = pd.concat([df, batch_df])
        
        df.to_csv('predicted_ko_data.csv', index=False)
            