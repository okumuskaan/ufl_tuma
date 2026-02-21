# Type-Based Unsourced Federated Learning with Client Self Selection

This repository provides research code for the unsourced federated learning framework proposed in our paper, accepted to IEEE International Conference on Communication (ICC) 2026. The framework integrates a client self-selection strategy into federated learning while addressing challenges of noisy communication over wireless fading channels. To model and manage the communication, we employ the type-based unsourced multiple access (TUMA) scheme over a distributed MIMO network.


## Environment
This project was developed and tested with Python 3.11.8. To set up the environment, it’s recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then, install the required libraries:
- **Torch:** For machine learning training and tensor operations. Install with `pip install torch`.
- **Torchvision:** For dataset handling. Install with  `pip install torchvision`.
- **Numpy:** For matrix and vector operations. Install with `pip install numpy`.
- **Scipy:** For probability distributions and pmf calculations. Install with `pip install scipy`.
- **Pandas:** For data management in the logger. Install with `pip install pandas`.
- **Matplotlib:**  For data visualization, required if using the Jupyter Notebook option (detailed below). Install with `pip install matplotlib`.
- **Plotly**: For enhanced data visualization, required if using the GUI option (detailed below). Install with `pip install plotly`.
- **Streamlit:** Required if using the GUI option. Install with `pip install streamlit`.




## Project Structure 

The project structure is as follows:
```bash
./
├── core/  
│  ├── client.py                    # Client class
│  ├── server.py                    # Server class
│  └── federated_learning.py        # Federated learning simulation function
├── communication/  
│  ├── comm_manager.py              # CommunicationManager class
│  ├── amp_da.py                    # MD-AirComp AMP-DA function
│  ├── topology.py                  # Network Topology class
│  ├── tuma_bayesian_decoder.py     # Bayesian denoiser functions
│  ├── tuma_centralized_decoder.py  # TUMA centralized decoder function 
│  ├── tuma_distributed_decoder.py  # TUMA distributed decoder function 
│  └── tuma.py                      # TUMAEnvironment class
├── data/ 
│  └── data_handler.py              # DataHandler class
├── model/ 
│  └── model_handler.py             # modelHandler function
├── quant/
│  └── vq_handler.py                # vqHandler function
├── setup/
│  └── initialize.py                # initialize_network_and_data function
├── utils/
│  ├── config_loader.py             # ConfigLoader class
│  └── logger.py                    # Logger class
├── configs/
│  └── sample_config.json           # Example configuration file
├── run.py                          # Run simulation directly via command line
├── run_GUI.py                      # Streamlit GUI for interactive runs
└── README.md
```


## Usage

You can run the project in two ways:

1. **Using command line (`run.py`):** This provides a direct run for simulation.
    * Open a terminal in the main directory.
    * Run the following command:
        ```
        python run.py
        ```
        To customize system parameters, modify `configs/sample_config.json` as desired.

2. **Using the GUI (`run_GUI.py`):** This provides a user-friendly way to explore the setup. To run the GUI:
    * Open a terminal in the main directory.
    * Run the following command:
        ```
        python -m streamlit run run_GUI.py
        ```

    This will open your default browser and launch the GUI, allowing you to adjust parameters and view results in real-time. 

### Notes

- On the first run, `initialize_network_and_data` automatically downloads and prepares the dataset in the specified data path (default is `./data/`).  
- For noisy communication with **TUMA**, priors for multiplicities are computed and stored in `./prior_data/`. These are reused in later runs.  
- Simulation results (e.g., test accuracy, number of selected clients) are saved as `.csv` files in `./results/`.  
- The folders `./prior_data/` and `./results/` are created automatically during the first execution.


## Citation

This work is based on our paper accepted to IEEE ICC 2026, which has not been published yet. You can cite the arXiv version of that paper:

```bibtex
@misc{okumus2026ufltuma,
      title={Type-Based Unsourced Federated Learning With Client Self-Selection}, 
      author={Kaan Okumus and Khac-Hoang Ngo and Unnikrishnan Kunnath Ganesan and Giuseppe Durisi and Erik G. Ström and Shashi Raj Pandey},
      year={2026},
      month={Feb.},
      eprint={2602.06601},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2602.06601}, 
}
```

 
## References

- **TUMA**:
  K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Ström, "Type-Based Unsourced Multiple Access", *IEEE Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, 2024.  [https://ieeexplore.ieee.org/document/10694658](https://ieeexplore.ieee.org/document/10694658)

- **TUMA over fading channels with CF massive MIMO**:
  K. Okumus, K.-H. Ngo, G. Durisi, and E. G. Ström, "Type-Based Unsourced Multiple Access Over Fading Channels with Cell-Free Massive MIMO", *IEEE International Symposium on Information Theory (ISIT)*, 2025.  [https://ieeexplore.ieee.org/document/11195493](https://ieeexplore.ieee.org/document/11195493)

- **Multisource AMP Algorithm**:
  B. Cakmak, E. Gkiouzepi, M. Opper, and G. Caire, "Joint Message Detection and Channel Estimation for Unsourced Random Access in Cell-Free User-Centric Wireless Networks," *IEEE Transactions on Information Theory*, 2025. [https://ieeexplore.ieee.org/document/10884602](https://ieeexplore.ieee.org/document/10884602)

- **MD-AirComp's AMP-DA Algorithm**:
  L. Qiao, Z. Gao, M. B. Mashadi, and D. Gunduz "Digital Over-the-Air Aggregation for Federated Edge Learning," *IEEE Journal on Selected Areas in Communications*, 2024. [https://ieeexplore.ieee.org/document/10648926](https://ieeexplore.ieee.org/document/10648926)

- **Code References**:
  + Code-base influenced by [pringlesinghal/GreedyFed](https://github.com/pringlesinghal/GreedyFed).
  + TUMA scheme codes from [okumuskaan/tuma_fading_cf](https://github.com/okumuskaan/tuma_fading_cf).
  + AMP-DA codes based on [liqiao19/MD-AirComp](https://github.com/liqiao19/MD-AirComp).
