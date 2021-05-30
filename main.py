import os
import argparse
import torch
import mymodel
from preprocess import CustomDataset
from train_eval import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    # Hyperparameters
    parser.add_argument("--num_epochs", default=20, type=int, help="The num of epochs for training")
    parser.add_argument("--embedd_dim", default=128, type=int, help="embedding dim size")
    parser.add_argument("--hidden_dim", default=256, type=int, help="hidden dim size")
    parser.add_argument("--num_layers", default=3, type=int, help="The num of layers")
    parser.add_argument("--model_type", default="RNN", type=str, help="model type")
    parser.add_argument("--out_node", default=1, type=int, help="Binary Classification : 1, Multiclass classification : 2 or more")
    parser.add_argument("--root", default="C:\\Users\\hanyeji\\Desktop\\workspace\\sentiment analysis\\mycode", type=str, help="root directory")
    parser.add_argument("--train_path", default='.\\ratings_train.txt', type=str, help="train path")
    parser.add_argument("--test_path", default='.\\ratings_test.txt', type=str, help="test path")
    parser.add_argument("--predict_path", default='.\\ko_data.csv', type=str, help="predict path")
    parser.add_argument("--batch_size", default=64, type=int, help="The num of batch size")
    args = parser.parse_args()
  

    loaders = CustomDataset(args.root, args.train_path, args.test_path, args.predict_path)
    vocab_size = len(loaders.text.vocab)
    if args.model_type=='LSTM':
    
        model = mymodel.NSMC_LSTM(vocab_size, args.embedd_dim, args.hidden_dim, args.num_layers, args.out_node)
    elif args.model_type=='GRU':
        model = mymodel.NSMC_GRU(vocab_size, args.embedd_dim, args.hidden_dim, args.num_layers, args.out_node)
    elif args.model_type=='RNN':
        model = mymodel.NSMC_RNN(vocab_size, args.embedd_dim, args.hidden_dim, args.num_layers, args.out_node)
    elif args.model_type=='Attention':
        model=mymodel.LSTM_attention(vocab_size,args.hidden_dim,args.num_layers,args.out_node,args.embedd_dim)
    
    trainer = Trainer(args, loaders, model)

    print("-------Train Start------")
    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in range(1,args.num_epochs+1):
        train_loss, train_acc = trainer.train(loaders.train_loader)
        valid_loss, valid_acc = trainer.evaluate(loaders.valid_loader)
        print("Epoch[{}/{}], Train Loss : {:.4f}, Train Acc : {:.2f}%, Valid Loss : {:.4f}, Valid Acc : {:.2f}%".format(epoch, args.num_epochs, train_loss, train_acc, valid_loss, valid_acc))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "{}_epoch_{}.pth".format(args.model_type, epoch))

    print("-------Train Ended------")

    # load best epoch model and evaluate on test set
    model.load_state_dict(torch.load('./{}_epoch_{}.pth'.format(args.model_type, best_epoch)))
    RNN_loss, RNN_acc = trainer.evaluate(loaders.test_loader)
    print("\n[Using Epoch {}'s model, evaluate on Test set]".format(best_epoch))
    print("{}'s Accuracy:{:.2f}% (Loss:{:.4f})".format(args.model_type, RNN_acc, RNN_loss))

    # predict Kaggle data (ko_data.csv)
    print("\n[Predicting on Kaggle Trainset]")
    model.load_state_dict(torch.load('./{}_epoch_{}.pth'.format(args.model_type, best_epoch)))
    trainer.predict(loaders.predict_loader,args.root,args.predict_path)


