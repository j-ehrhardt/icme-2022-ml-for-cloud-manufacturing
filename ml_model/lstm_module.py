import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from create_experiments import *
from data_module import *

class LSTM(nn.Module):
    """
    standard torch LSTM Module linear layer added as output
    """
    def __init__(self, hparam: dict, input_size):
        super(LSTM, self).__init__()

        self.num_classes = input_size
        self.num_layers = hparam["NUM_LAYERS"]
        self.input_size = input_size
        self.hidden_size = input_size * hparam["XZ_RATIO"]
        self.seq_len = hparam["SEQ_LEN"]

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)

        out = self.linear(h_out)
        return out

def train_lstm(hparam: dict, train_X, train_y, input_size):
    """
    training an lstm
    """
    torch.manual_seed(hparam["MAN_SEED"])

    lstm = LSTM(hparam, input_size)

    loss_fct = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=hparam["LEARNING_RATE"])

    logger = SummaryWriter(hparam["CHECKPOINT_PATH"] + "/logs_exp" + str(hparam["EXPERIMENT_ID"]))

    # Train lstm
    for epoch in range(hparam["NUM_EPOCHS"]):
        outputs = lstm(train_X)
        optimizer.zero_grad()
        loss = loss_fct(outputs, train_y)
        loss.backward()
        optimizer.step()
        logger.add_scalar("Loss/train", loss, epoch)
    torch.save(lstm.state_dict(), hparam["CHECKPOINT_PATH"] + "/logs_exp" + str(hparam["EXPERIMENT_ID"]) + "/model.pth")
    return lstm

if __name__ == "__main__":
    hparam = load_experiments()
    hparam = hparam["0"]

    comp_X, comp_y, train_X, train_y, _, _,input_size, _ = create_ds(hparam=hparam)
    lstm = train_lstm(hparam, train_X, train_y, input_size)

