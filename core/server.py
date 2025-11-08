import torch
from torch.utils.data import DataLoader
import math
from copy import deepcopy

class Server:
    def __init__(self, 
            model,             # global model
            lr,                # aggregation learning rate
            test_dataset,      # dataset for testing
            select_config,     # selection strategy config
            vq,                # vq_handler
            data_subset,       # dataset for VQ codebook learning
            train_lr,          # training learning rate
            localE,            # local epochs
            bs,                # batch size
            device,            # torch.device('cpu') or torch.device('cuda')
            seed               # seed for reproducibility
        ):
        
        # ---------- Seed for reproducibility ----------
        self.device = device
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # ---------- Global model ----------
        self.model = deepcopy(model).to(self.device)
        self.global_parameters = [
            p.detach().clone() for p in self.model.parameters()
        ]

        # ---------- VQ codebook learning parameters ----------
        self.test_dataset = test_dataset
        self.datasubset = data_subset
        self.bs = bs
        self.localE = localE
        self.train_lr = train_lr
        self.momentum = 0.0
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.train_lr, weight_decay=1e-3
        )

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


        # ---------- Aggregation parameters ----------
        self.lr = lr
        self.round = 0

        # ---------- Selection parameters ----------
        self.selection_strategy = select_config["strategy"]
        self.Ktar = select_config["Ktar"]
        if self.selection_strategy == "self":
            self.gamma = select_config["gamma"]
            self.th = select_config["th_0"]
            self.ths = [self.th]
            self.thU = select_config["thU"]
            self.thL = select_config["thL"]
        elif self.selection_strategy == "pow-d":
            self.d = select_config["d"]

        # ---------- Learn VQ codebook ----------
        if vq!=0:
            self.learn_VQ_codebook()  

    # ---------------------------------------------------------
    def learn_VQ_codebook(self):
        # Perform local training to get model updates
        self.train()

        # Flatten trained update dictionary
        s_bar_flattened = self._dict_to_flattened(self.s_bar_state)

        # Learn vector quantization codebook
        s_bar_matrix = s_bar_flattened.reshape(self.D, self.Q).to(self.device)
        self.train_data_for_VQ = s_bar_matrix
        self.vq.fit(s_bar_matrix) 
        self.vq_codebook = self.vq.centroids.to(self.device)

        # Quantize s̄ -> s
        s_matrix, _ = self.vq.evaluate(s_bar_matrix)
        s_flattened = s_matrix.reshape(-1)

        # Compute quantization error e = s - s̄
        e_flattened = s_flattened - s_bar_flattened

        # Unflatten to parameter dictionaries
        self.s_state = self._flattened_to_dict(s_flattened)
        self.e_state = self._flattened_to_dict(e_flattened)

    # ---------------------------------------------------------
    def train(self):
        previous_model_state = deepcopy(self.model.state_dict())
        # ---- train on a COPY, not self.model ----
        model_copy = deepcopy(self.model).to(self.device)
        model_copy.train()
        optimizer_copy = torch.optim.SGD(
            model_copy.parameters(), lr=self.train_lr, weight_decay=1e-3
        )
        criterion = self.criterion

        # Run local training
        for _ in range(self.localE):
            dataloader = DataLoader(
                self.datasubset,
                batch_size=min(self.bs, len(self.datasubset)),
                shuffle=True,
                pin_memory=True
            )
            X, y = next(iter(dataloader))
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer_copy.zero_grad()
            y_hat = model_copy(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer_copy.step()

        # Compute model updates
        trained_model_state = deepcopy(model_copy.state_dict())
        self.model_update_state = {}
        self.s_bar_state = {}
        for key in self.state_keys:
            update = trained_model_state[key] - previous_model_state[key]
            self.model_update_state[key] = update
            self.s_bar_state[key] = update + self.e_state[key]

        # ---- restore original global model & parameters ----
        self.model.load_state_dict(previous_model_state)    # ensure NO drift
        self._refresh_global_parameters()                   # keep clients in sync

    # ---------------------------------------------------------
    def receive_tuma_outputs(self, rx_quant_inds, rx_mults, est_num_clients):
        self.num_clients = est_num_clients
        self.aggregate_locally(rx_quant_inds, rx_mults)

    def receive_perfect_comm_no_vq(self, rx_model_updates):
        self.clients_model_updates = rx_model_updates
        self.aggregate()    

    # ---------------------------------------------------------
    def aggregate_locally(self, local_quant_inds, mults):
        model_state = deepcopy(self.model).state_dict()
        model_state_flattened = self._dict_to_flattened(model_state)
        for sub_rnd, (local_quant_ind, mult) in enumerate(zip(local_quant_inds, mults)):
            if len(local_quant_ind)!=0:
                local_update_state = torch.sum(self.vq_codebook[local_quant_ind] * mult.reshape(-1,1),axis=0) / mult.sum()
                model_state_flattened[sub_rnd*self.Q:(sub_rnd+1)*self.Q] = model_state_flattened[sub_rnd*self.Q:(sub_rnd+1)*self.Q] + self.lr * local_update_state
        new_model_state = self._flattened_to_dict(model_state_flattened)
        self.model.load_state_dict(new_model_state)
        self._refresh_global_parameters()
    
    def aggregate(self):
        """ aggregation strategy in FedAvg """
        self.num_clients = len(self.clients_model_updates)
        model_state = deepcopy(self.model).state_dict()
        for key in self.state_keys:
            for client_model_updates in self.clients_model_updates:
                model_state[key] += self.lr * client_model_updates[key] / self.num_clients
        self.model.load_state_dict(model_state)
        self._refresh_global_parameters()

    # ---------------------------------------------------------
    def update_th(self):
        # Convert all scalars to torch tensors on the correct device
        th = torch.tensor(self.th, dtype=torch.float64, device=self.device)
        thL = torch.tensor(self.thL, dtype=torch.float64, device=self.device)
        thU = torch.tensor(self.thU, dtype=torch.float64, device=self.device)
        gamma = torch.tensor(self.gamma, dtype=torch.float64, device=self.device)
        err = torch.tensor(self.num_clients - self.Ktar, dtype=torch.float64, device=self.device)

        # Deadband ±1
        if torch.abs(err) > 1:
            delta = gamma * err
            delta = torch.clamp(delta, -0.1, 0.1)
            th = th + delta

        # Clamp to allowed range
        th = torch.clamp(th, thL, thU)

        # Update attributes (keep float for compatibility)
        self.th = th.item()
        self.ths.append(self.th)

    # ---------------------------------------------------------
    def set_params(self, parameters):
        """ set parameters of global model """
        with torch.no_grad():
            for model_param, param in zip(self.model.parameters(), parameters):
                model_param.copy_(param)

    # ---------------------------------------------------------
    def eval(self, in_model, dataloader, device):
        """ compute loss, acc for client i on train data """
        model = deepcopy(in_model)
        model.eval()

        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for _, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                
                # foward pass
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
    def evaluate_approx(self):
        self.model.eval()
        # fetch full data
        dataset = self.test_dataset
        dataloader = DataLoader(dataset,
                                batch_size=min(self.bs, len(dataset)),
                                shuffle=True,
                                pin_memory=True)
        loss, acc = self.eval(self.model, dataloader, self.device)
        return loss, acc

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

    def _refresh_global_parameters(self):
        self.global_parameters = [p.detach().clone() for p in self.model.parameters()]
