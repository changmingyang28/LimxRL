from .encoder import Encoder
import torch.nn as nn
from limx_rl.utils import unpad_trajectories

class RnnEncoder(Encoder):
    is_recurrent = True
    def __init__(self,
                 input_dim,
                 rnn_type="lstm",
                 rnn_hidden_size=256,
                 rnn_num_layers=1,
                 **kwargs):
        if kwargs:
            print(
                "RnnEncoder.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()
        self.memory = Memory(input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.output_dim = rnn_hidden_size
    def reset(self, dones=None):
        self.memory.reset(dones)
    def forward(self, input, *args, masks=None, hidden_states=None, **kwargs):
        return self.memory(input, masks, hidden_states).squeeze(0)

    def get_hidden_states(self):
        return self.memory.hidden_states


class Memory(nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0