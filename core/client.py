import torch 
from torch.utils.data import DataLoader 
import math
import random
from copy import deepcopy

class Client:
    def __init__(self, 
            id,                                                     # id
            pos,                                                    # position
            model,                                                  # local model
            data_subset,                                            # local dataset
            localE, bs, lr,                                         # local training parameters
            q,                                                      # activation probability
            select_config,                                          # selection strategy config
            num_clients,                                            # total number of clients
            vq,                                                     # vq_handler
            device,                                                 # torch.device('cpu') or torch.device('cuda')
            seed,                                                   # seed for reproducibility
        ):

        # ---------- Seed for reproducibility ----------
        self.device = device
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # ---------- Local id, position, model ----------
        self.id = id
        self.pos = pos
        self.model = deepcopy(model).to(device)

        # ---------- Learning parameters ----------
        self.localE = localE
        self.bs = bs
        self.lr = lr
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)

        # ---------- Local dataset ----------
        self.datasubset = data_subset

        # ---------- State info ----------
        model_state = self.model.state_dict()
        self.state_keys = list(model_state.keys())
        self.state_info_dict = {}
        self.e_state = {}
        W = 0
        for key in self.state_keys:
            tensor = model_state[key]
            self.e_state[key] = torch.zeros_like(tensor, device=self.device)
            self.state_info_dict[key] = {
                "numel": tensor.numel(),
                "shape": tensor.shape
            }
            W += tensor.numel()
        self.W = W

        # ---------- VQ parameters ----------
        self.vq = vq
        if self.vq != 0:
            self.Q = vq.Q
            self.D = math.ceil(self.W / self.Q)
            self.pad_0_nums = self.D * self.Q - self.W

        # ---------- Selection parameters ----------
        self.selection_strategy = select_config["strategy"]
        self.num_clients = num_clients
        self.Ktar = select_config["Ktar"]
        if self.selection_strategy == "self":
            self.utility_scores = []
            self.prob_selections = []
            self.ths = []
            self.th = select_config["th_0"]
            self.d = select_config["d"]
            self.a = select_config["a"]
        self.selecteds = []
        self.activities = []
        self.total_rounds = 0
        self.round_number = 0

        # ---------- Set active and selected boolean False ----------
        self.q = q
        self.active = False
        self.selected = False

    # ---------------------------------------------------------
    def __str__(self):
        return f"Client {self.id+1}"

    # ---------------------------------------------------------
    def _flattened_to_dict(self, x):
        """Convert flattened parameter vector -> state dict."""
        res = {}
        i = 0
        for key in self.state_keys:
            numel = self.state_info_dict[key]["numel"]
            shape = self.state_info_dict[key]["shape"]
            res[key] = x[i:i + numel].reshape(shape)
            i += numel
        return res

    def _dict_to_flattened(self, x):
        """Convert state dict -> flattened parameter vector."""
        dtype = x[list(self.state_keys)[0]].dtype
        res = torch.empty(0, dtype=dtype, device=self.device)
        for key in self.state_keys:
            temp = x[key].reshape(-1)
            res = torch.hstack((res, temp))
        res = torch.hstack((res, torch.zeros(self.pad_0_nums, dtype=dtype, device=self.device)))
        return res

    # ---------------------------------------------------------
    def choose_self_activity(self):
        """Client chooses itself as active or not. For each round, a client is active with probability q."""
        self.active = random.random() < self.q
        self.selected = False
        if self.active:
            self.activities.append(1)
        else:
            self.selecteds.append(0.0)
            self.activities.append(0.0)
            if self.selection_strategy=="self":
                self.utility_scores.append(0.0)
                self.prob_selections.append(0.0)
                self.ths.append(0.0)
        self.round_number += 1

    # ---------------------------------------------------------
    def receive(self, global_parameters):
        """Client receives the global model from the server."""
        self.set_params(global_parameters)

    def receive_vq_codebook(self, vq_codebook):
        """
        Client receives vq codebook from the server.
        """
        self.vq_codebook = vq_codebook.to(self.device)

    def receive_th(self, th):
        self.th = th

    # ---------------------------------------------------------
    def set_params(self, parameters):
        """ set parameters of global model """
        with torch.no_grad():
            for model_param, param in zip(self.model.parameters(), parameters):
                model_param.copy_(param)

    def get_params(self):
        """ get parameters of global model """
        parameters = []
        with torch.no_grad():
            for model_param in self.model.parameters():
                parameters.append(model_param.detach().clone().to(self.device))
        return parameters

    # ---------------------------------------------------------
    def self_select(self):
        """Client self-selection."""
        self.selected = False
        if self.selection_strategy == "self":
            # ---------- Strategy: self-selection ----------
            self.total_rounds +=1
            self.selected_candidate = random.random() < self.d/(self.q * self.num_clients)
            if self.selected_candidate:
                dataloader = DataLoader(self.datasubset,
                    batch_size=len(self.datasubset),
                    shuffle=True,
                    pin_memory=True
                )
                U, _ = self.eval(self.model, dataloader)
                p = 1.0 / (1.0 + math.exp(-self.a * (U - self.th)))
                self.selected = random.random() < p
        elif self.selection_strategy=="rand":
            # ---------- Strategy: uniform selection ----------
            p = self.Ktar/(self.q * self.num_clients)
            self.selected = random.random() < p
        elif self.selection_strategy=="full":
            # ---------- Strategy: full selection ----------
            self.selected = True
        else:
            raise Exception("Invalid selection strategy:", self.selection_strategy)
        if self.selected:
            self.selecteds.append(1.0)

    # ---------------------------------------------------------
    def eval(self, in_model, dataloader):
        """ compute loss, acc for client `i` on train data """
        model = deepcopy(in_model)
        model.eval()
        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for _, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                # forward pass
                y_hat = model(X)
                # compute loss
                loss_tmp = self.criterion(y_hat, y)
                loss += loss_tmp.item() * X.size(0)
                # prediction
                _, pred_labels = torch.max(y_hat,1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum((pred_labels == y).float()).item()
                total += X.size(0)
        loss /= total
        acc = correct/total
        return loss, acc

    # ---------------------------------------------------------
    def local_update(self):
        previous_model_state = deepcopy(self.model.state_dict())
        loss = 0
        for _ in range(self.localE):
            tmp_loss, _ = self.local_train()
            loss += tmp_loss
        loss = loss/self.localE
        self.local_params = self.get_params()

        trained_model_state = deepcopy(self.model.state_dict())
        self.model_update_state = {}
        self.s_bar_state = {}
        for key in self.state_keys:
            self.model_update_state[key] = trained_model_state[key] - previous_model_state[key]
            self.s_bar_state[key] = self.model_update_state[key] + self.e_state[key]
        return self.local_params, loss

    def local_train(self):
        """compute loss, acc for client `i` on train data and run optimizer step """
        self.model.train()

        dataloader = DataLoader(self.datasubset,
                                batch_size=min(self.bs, len(self.datasubset)),
                                shuffle=True,
                                pin_memory=True)
        
        X, y = next(iter(dataloader))
        X, y = X.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        y_hat = self.model(X)
        loss = self.criterion(y_hat, y)
        _, pred_labels = torch.max(y_hat,1)
        pred_labels = pred_labels.view(-1)
        acc = torch.mean((pred_labels == y).float())
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()

    # ---------------------------------------------------------
    def add_cf_data(self, zone, zone_center, side):
        self.zone = zone
        self.zone_center = zone_center
        self.side = side

    # ---------------------------------------------------------
    def quantize(self):
        """Client applies quantization to the model update and computes the error. """
        s_bar_flattened = self._dict_to_flattened(self.s_bar_state)
        s_bar_matrix = s_bar_flattened.reshape(self.D, self.Q)
        s_flattened, self.quant_indices = self.quantize_func(s_bar_matrix)
        s_flattened = s_flattened.reshape(-1)
        e_flattened = s_flattened - s_bar_flattened
        self.s_state = self._flattened_to_dict(s_flattened)
        self.e_state = self._flattened_to_dict(e_flattened)
        return self.s_state, self.quant_indices

    def quantize_func(self, X):
        """Quantization function."""
        dists = torch.cdist(X, self.vq_codebook, p=2)
        labels = torch.argmin(dists, dim=1)
        centroids = self.vq_codebook[labels]
        return centroids, labels
