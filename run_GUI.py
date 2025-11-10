import streamlit as st

import torch
import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.colors import qualitative 
from scipy.stats import gaussian_kde

from utils.config_loader import ConfigLoader
from setup.initialize import initialize_network_and_data
from core.federated_learning import simulate_ufl_for_GUI


# Center-align Streamlit UI elements
st.markdown(
    """
    <style>
    .stButton > button {
        display: flex;
        margin: auto;
    }
    h1, h2, p {
        text-align: center;
    }
    """,
    unsafe_allow_html=True
)


# 1. Display App Title and Description
def display_header():
    st.title("Type-Based Unsourced Federated Learning Simulation")
    st.write("Simulate federated learning rounds with type-based unsourced multiple access")


# 2. Configuration settings layout
def configure_settings():
    config_loader = ConfigLoader()
    config = {}
    with st.container(border=True):
        st.subheader("Configuration settings:")
        for row in range(3):
            maincol1, maincol2 = st.columns(2)
            with maincol1:
                if row==0:
                    config['federated_learning'] = display_federated_learning_settings()
                elif row==1:
                    config['model'] = display_model_settings()
                else:
                    config['data'] = display_data_settings()
            with maincol2:
                if row==0:
                    config['selection'] = display_selection_settings()
                elif row==1:
                    config['communication'] = display_communication_settings()
                else:
                    config['quantization'] = display_quantization_settings()

        config_loader.set_config(config)
        st.session_state["config_loader"] = config_loader

def display_federated_learning_settings():
    with st.expander("Federated Learning Settings"):
        for row in range(3):
            col1, col2, col3 = st.columns(3)
            with col1:
                if row==0:
                    num_clients = st.number_input("Num. Clients", min_value=1, max_value=2000, value=1000)
                elif row==1:
                    num_rounds = st.number_input("Rounds", min_value=1, max_value=1000, value=500)
                else:
                    q = st.slider("Activ. prob.", min_value=0.0, max_value=1.0, value=0.8)
            with col2:
                if row==0:
                    client_lr = st.number_input("Client lr", min_value=0.0001, max_value=1.0, value=0.001, format="%.4f")
                elif row==1:
                    server_lr = st.number_input("Server lr", min_value=0.001, max_value=10.0, value=1.0)
            with col3:
                if row==0:
                    localE = st.number_input("Epochs", min_value=1, max_value=100, value=30)
                elif row==1:
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=64)          
        return {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'q': q,
            'client_lr': client_lr,
            'server_lr': server_lr,
            'localE': localE,
            'bs': batch_size,
            'seed': 12345
        }

def display_model_settings():
    with st.expander("Model Settings"):
        col1, _, _ = st.columns(3)
        with col1:
            architecture = st.selectbox("Architecture", ["MLP", "cnn"], index=0)
        return {
            'architecture': architecture,
            'input_shape': [28, 28],
            'output_dim': 10
        }

def display_data_settings():
    with st.expander("Data Settings"):
        for row in range(2):
            col1, col2 = st.columns(2) 
            with col1:
                if row==0:
                    dataset = st.selectbox("Dataset", ["fmnist", "cifar10", "mnist"], index=0)                
                else:
                    alpha = st.number_input("Alpha", min_value=0.0001, max_value=100.0, value=2.0, format="%0.2f")
            with col2:
                if row==0:
                    data_dir = st.text_input("Data Dir", value="./data")

        return {
            'dataset': dataset,
            'alpha': alpha,
            'data_dir': data_dir
        }

def display_selection_settings():
    with st.expander("Selection Settings"):
        th_0 = 2.32
        thU = 3.0
        thL = -1.0
        a = 10
        Ktar = 100
        d = 200
        strategy = "self"
        gamma = 0.004

        partial_selection = st.checkbox("Partial Client Participation", value=True, key="partial_client_participation")
        if partial_selection:
            col1, _ = st.columns(2)
            with col1:
                strategy = st.selectbox("Strategy", ["rand", "powd", "self"], index=2)

            if strategy in ["rand"] :
                col1, _ = st.columns(2)
                with col1:
                    Ktar = st.number_input("Ktar", min_value=1, max_value=1000, value=100)
            elif strategy == "powd":
                col1, col2 = st.columns(2)
                with col1:
                    Ktar = st.number_input("Ktar", min_value=1, max_value=1000, value=100)
                with col2:
                    d = st.number_input("d", min_value=10, max_value=1000, value=200)
            else:
                for row in range(3):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if row==0:
                            Ktar = st.number_input("Ktar", min_value=1, max_value=1000, value=100)
                        elif row==1:
                            a = st.number_input("a", min_value=1, max_value=100, value=2)
                        elif row==2:
                            th_0 = st.number_input("Th0", min_value=0.0, max_value=4.0, value=2.32)
                    with col2:
                        if row==0:
                            d = st.number_input("d", min_value=10, max_value=1000, value=200)
                        elif row==1:
                            gamma = st.number_input("Gamma", min_value=0.001, max_value=0.01, value=0.004, format="%.3f")
                        elif row==2:
                            thL = st.number_input("Th L", min_value=-3.0, max_value=1.0, value=-1.0)
                    with col3:
                        if row==2:
                            thU = st.number_input("Th U", min_value=1.0, max_value=3.0, value=3.0)
        else:
            strategy = "full"
            st.write("Full client participation.")

        return {
            'strategy': strategy,
            'th_0': th_0,
            'thU': thU,
            'thL': thL,
            'a': a,
            'd': d,
            'Ktar': Ktar,
            'gamma': gamma,
        }

def display_communication_settings():
    decoder_type = "centralized"
    SNR_rx_dB = 10.0
    zone_side = 0.1
    N = 100
    A = 4
    rho = 3.67
    d0 = 0.01357
    nAMPiter = 2
    N_MC = 50
    Kmax = 8
    perfectCSI = 1
    phase_max_pi_div = 6
    with st.expander("Communication Settings"):
        perfect_comm = st.checkbox("Perfect Communication", value=True, key="perfect_comm")
        if not perfect_comm:
            coding_scheme = st.selectbox("Comm. Scheme", ["TUMA", "MD-AirComp"])
            if coding_scheme=="TUMA":
                col1, col2 = st.columns(2)
                with col1:
                    decoder_type = st.selectbox("Comm. Scheme", ["centralized", "distributed"])
            else:
                decoder_type = "AMP-DA"
            for row in range(3):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if row==0:
                        SNR_rx_dB = st.number_input("SNR_rx (dB)", min_value=-50.0, max_value=20.0, value=10.0)  
                    if row==1:
                        A = st.number_input("A", min_value=1, max_value=10, value=4)                        
                with col2:
                    if row==0:
                        zone_side = st.number_input("Zone side", min_value=0.01, max_value=10.0, value=0.1)
                    if row==1:
                        rho = st.number_input("rho", min_value=1.0, max_value=10.0, value=3.67)                        
                with col3:
                    if row==0:
                        N = st.number_input("N", min_value=1, max_value=1000, value=100)       
                    if row==1:
                        d0 = st.number_input("d0", min_value=0.001, max_value=0.1, value=0.01357, format="%.5f")  
            if decoder_type=="AMP-DA":
                col1, _ = st.columns(2)
                with col1:
                    perfectCSI = st.checkbox("Perfect CSI", value=True, key=f"perfectCSI")
                if not perfectCSI:
                    col1, _ = st.columns(2)
                    with col1:
                        phase_max_pi_div = st.number_input("Phase_max pi div", min_value=1, max_value=20, value=6)
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    nAMPiter = st.number_input("nAMPiter", min_value=1, max_value=10, value=2)
                with col2:
                    N_MC = st.number_input("N_MC", min_value=10, max_value=1000, value=50)
                with col3:
                        Kmax = st.number_input("Kmax", min_value=5, max_value=40, value=8)    
        return {
            "perfect_comm": perfect_comm,
            "decoder_type": decoder_type,
            "SNR_rx_dB": SNR_rx_dB,
            "zone_side": zone_side,
            "blocklength": N,
            "prior_type": "default",
            "A": A,
            "rho": rho,
            "d0": d0,
            "nAMPiter": nAMPiter, 
            "N_MC": N_MC, 
            "N_MC_smaller": 1, 
            "Kmax": Kmax,
            "perfectCSI": perfectCSI,
            "phase_max_pi_div": phase_max_pi_div
        }

def display_quantization_settings():
    with st.expander("Quantization Settings"):
        apply = st.checkbox("Apply Quantization", value=True, key="applyVQ")
        if apply:
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox("Method", ["kmeans++"])
            col1, col2 = st.columns(2)
            with col1:
                J = st.number_input("J", min_value=1, max_value=100, value=7)
            with col2:
                Q = st.number_input("Q", min_value=1, max_value=10000, value=30)
        else:
            method="kmeans++"
            Q = 30
            J = 7

        return {
            'apply': int(apply),
            'method': method,
            'Q': Q,
            'J': J
        }


# 3. Initialization button action
def initialize_network():
    with st.container():
        if st.button("Initialize Network", type="secondary"):
            config_loader = st.session_state["config_loader"]
            server, clients, topology, data_handler, logger = initialize_network_and_data(config_loader)
            st.session_state["server"] = server
            st.session_state["clients"] = clients
            st.session_state["topology"] = topology
            st.session_state["data_handler"] = data_handler
            st.session_state["logger"] = logger
            st.success(f"{len(clients)} Clients and 1 Server are generated. :white_check_mark:")
            init_figs()
            generate_visuals_data_heterogeneity(data_handler)
            generate_visuals_sample_counts_histogram(data_handler)
            generate_visuals_sample_distribution_with_kde(data_handler)
            with st.expander("See How Data is Distributed:"):   
                tab1, tab2 = st.tabs(["Data Heterogeneity", "Number of Samples"]) 
                with tab1:
                    tab11, tab12 = st.tabs(["Clients Train", "Clients Test"])
                    with tab11:     
                        st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_data_het"][0], use_contained_width=True)       
                    with tab12:
                        st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_data_het"][1], use_contained_width=True)  
                    if data_handler.include_server:
                        tab121, tab122 = st.tabs(["Server Train", "Server Test"])
                        with tab121:
                            st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_data_het"][2], use_contained_width=True)       
                        with tab122:
                            st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_data_het"][3], use_contained_width=True)

                with tab2:
                    tab21, tab22 = st.tabs(["Sample Counts per Object", "Histogram and KDE"])
                    with tab21:
                        tab211, tab212 = st.tabs(["Train", "Test"])
                        with tab211:
                            st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_samp_count"][0], use_contained_width=True)
                        with tab212:
                            st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_samp_count"][1], use_contained_width=True)
                    with tab22:
                        tab221, tab222 = st.tabs(["Train", "Test"])
                        with tab221:
                            st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_hist_kde"][0], use_contained_width=True)
                        with tab222:
                            st.plotly_chart(st.session_state["figs"]["data-heterogeneity"]["fig_hist_kde"][1], use_contained_width=True)

            if not config_loader.get_comm_config()["perfect_comm"]:
                with st.expander("See Topology:"):
                    zone_centers = torch.tensor([zone.center for zone in topology.zones])
                    cliet_positions = torch.tensor([client.pos for client in clients])
                    st.plotly_chart(generate_visual_topology(topology.ap_positions, topology.side_length, zone_centers, cliet_positions), use_contained_width=True)

    st.divider()

def generate_visuals_data_heterogeneity(data_handler):
    num_targets = len(data_handler.train_dataset.targets.unique())
    epsilon = 1e-10  # Small value to avoid division by zero

    alpha = data_handler.alpha
    num_clients = data_handler.num_clients
        
    # Distribute data across clients (and server if applicable)
    # Check if server data is included and handle accordingly
    if data_handler.include_server:
        server_index_train = data_handler.server_index_train
        server_index_test = data_handler.server_index_test

    client_indices_train = data_handler.client_indices_train
    client_indices_test = data_handler.client_indices_test

    def plot_distribution(indices, dataset, title):
        for_server = (len(indices)!=num_clients)

        label_distributions = []
        for i in range(len(indices)):
            labels = [dataset.targets[idx] for idx in indices[i]]
            label_counts = [labels.count(label) for label in range(num_targets)]
            label_distributions.append(label_counts)

        # Normalize to get proportions
        label_distributions = np.array(label_distributions)
        label_distributions = label_distributions / (label_distributions.sum(axis=1, keepdims=True) + epsilon)

        # Generate colors from Plotly's default qualitative color sequence
        colors = qualitative.Plotly * (num_targets // len(qualitative.Plotly) + 1)
        colors = colors[:num_targets]  # Trim or extend the list to match num_targets

        # Plot using Plotly
        fig = go.Figure()
        for target in range(num_targets):
            fig.add_trace(go.Bar(
                x= ["Server"] if for_server else [f'Client {i+1}' for i in range(num_clients)],
                y=label_distributions[:, target],
                name=f'Label {target}',
                hoverinfo="x+y",
                marker=dict(color=colors[target])
            ))

        fig.update_layout(
            title=title,  
            yaxis=dict(
                title="Proportion of Labels",
                range=[0,1]
            ),
            barmode="stack",
            legend=dict(x=1.02, y=1, orientation="v"),
        )
        if for_server:
            fig.update_layout(
                xaxis=dict(
                    title="", 
                    tickmode='array', 
                    tickvals = [0],
                    ticktext = ["Server"],
                    tickangle=0
                ),
            )
        else:
            xlabels = np.array(
                [f'Client {i+1}' for i in range(len(client_indices_train))]
            )
            xtickvals = np.linspace(0,data_handler.num_clients-1, 7).astype(int)
            xticktexts = xlabels[xtickvals]
            fig.update_layout(
                xaxis=dict(
                    title="", 
                    tickmode='array', 
                    tickvals = xtickvals,
                    ticktext = xticktexts,
                    tickangle=45  # Sloped labels
                ),
            )
        return fig

    # Plot train and test distributions for clients
    fig1 = plot_distribution(client_indices_train, data_handler.train_dataset, title=f"Train Label Distribution per Client (Alpha={alpha})")
    fig2 = plot_distribution(client_indices_test, data_handler.test_dataset, title=f"Test Label Distribution per Client (Alpha={alpha})")
    st.session_state["figs"]["data-heterogeneity"]["fig_data_het"].append(fig1)
    st.session_state["figs"]["data-heterogeneity"]["fig_data_het"].append(fig2)

    # Plot server data distributions if server data is included
    if data_handler.include_server:
        fig3 = plot_distribution([server_index_train], data_handler.train_dataset, title=f"Server Train Label Distribution (Alpha={alpha})")
        fig4 = plot_distribution([server_index_test], data_handler.test_dataset, title=f"Server Test Label Distribution (Alpha={alpha})")
        st.session_state["figs"]["data-heterogeneity"]["fig_data_het"].append(fig3)
        st.session_state["figs"]["data-heterogeneity"]["fig_data_het"].append(fig4)

def init_figs():
    st.session_state["figs"] = {
        "data-heterogeneity": {
            "fig_data_het": [],
            "fig_samp_count": [],
            "fig_hist_kde": []
        }
    }

def generate_visuals_sample_counts_histogram(data_handler):
    client_indices_train = data_handler.client_indices_train
    client_indices_test = data_handler.client_indices_test
    server_index_train = None #data_handler.server_index_train
    server_index_test = None #data_handler.server_index_test
    
    # Calculate the number of samples per client for train and test
    train_counts = [len(client_indices_train[key]) for key in client_indices_train.keys()]
    test_counts = [len(client_indices_test[key]) for key in client_indices_test.keys()]

    include_server = (server_index_train is not None and server_index_test is not None)

    # If server data is included, add server counts
    if include_server:
        train_counts.append(len(server_index_train))
        test_counts.append(len(server_index_test))
        xlabels = np.array(
            [f'Client {i+1}' for i in range(len(client_indices_train))] + ['Server']
        )
        title_train = "Number of Training Samples per Client, with Server Added at End"
        title_test = "Number of Testing Samples per Client, with Server Added at End"
    else:
        xlabels = np.array(
            [f'Client {i+1}' for i in range(len(client_indices_train))]
        )
        title_train = "Number of Training Samples per Client"
        title_test = "Number of Testing Samples per Client"

    xtickvals = np.linspace(0,data_handler.num_clients-1, 7).astype(int)
    xticktexts = xlabels[xtickvals]

    # Plot training sample counts
    fig_train = go.Figure()
    fig_train.add_trace(go.Bar(
        x=xlabels,
        y=train_counts,
        name="Train Samples",
        marker_color="blue"
    ))
    fig_train.update_layout(
        title=title_train,
        xaxis=dict(
            tickmode='array', 
            tickvals=xtickvals,
            ticktext=xticktexts,
            tickangle=45
        ),
        yaxis=dict(title="Number of Samples")
    )

    # Plot testing sample counts
    fig_test = go.Figure()
    fig_test.add_trace(go.Bar(
        x=xlabels,
        y=test_counts,
        name="Test Samples",
        marker_color="green"
    ))
    fig_test.update_layout(
        title=title_test,
        xaxis=dict(
            tickmode='array', 
            tickvals=xtickvals,
            ticktext=xticktexts,
            tickangle=45
        ),
        yaxis=dict(title="Number of Samples")
    )
    
    st.session_state["figs"]["data-heterogeneity"]["fig_samp_count"].append(fig_train)
    st.session_state["figs"]["data-heterogeneity"]["fig_samp_count"].append(fig_test)
    
def generate_visuals_sample_distribution_with_kde(data_handler, bin_size_train=12, bin_size_test=2):

    client_train_counts = np.array([len(indices) for indices in data_handler.client_indices_train.values()])
    client_test_counts = np.array([len(indices) for indices in data_handler.client_indices_test.values()])
    server_train_count = len(data_handler.server_index_train) if data_handler.include_server else 0
    server_test_count = len(data_handler.server_index_test) if data_handler.include_server else 0

    train_counts = np.concatenate((client_train_counts, [server_train_count])) if server_train_count > 0 else client_train_counts
    test_counts = np.concatenate((client_test_counts, [server_test_count])) if server_test_count > 0 else client_test_counts

    def plot_sample_distribution_with_kde(counts, title, bin_size, sample_color="#636EFA", kde_color="#EF553B"):
        fig = go.Figure()
        # Plot histogram with smaller bin size for narrower bars
        fig.add_trace(go.Histogram(
            x=counts,
            histnorm='probability',
            name='Sample Counts',
            xbins=dict(size=bin_size),
            marker=dict(color=sample_color),
            opacity=0.6
        ))

        # Calculate and plot KDE
        kde = gaussian_kde(counts, bw_method=0.3)
        x_range = np.linspace(min(counts), max(counts), 100)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode='lines',
            name='KDE',
            yaxis="y2",
            line=dict(color=kde_color, width=2)
        ))

        # Set layout with dual y-axis
        fig.update_layout(
            title=title,
            xaxis_title="Number of Samples",
            yaxis=dict(title="Probability Density", showgrid=False),
            yaxis2=dict(
                overlaying="y", side="right", showgrid=False
            ),
            legend=dict(x=0.01, y=1, orientation="v"),
            barmode="overlay"
        )

        # Add rotated "KDE" title as an annotation
        fig.add_annotation(
            text="KDE",
            xref="paper", yref="paper",
            x=1, xshift=60,
            y=0.5,
            showarrow=False,
            textangle=90,  # Rotate text clockwise
            font=dict(size=12)
        )
        return fig

    fig_train = plot_sample_distribution_with_kde(train_counts, title="Training Sample Count Distribution", bin_size=bin_size_train)
    fig_test = plot_sample_distribution_with_kde(test_counts, title="Testing Sample Count Distribution", bin_size=bin_size_test)

    st.session_state["figs"]["data-heterogeneity"]["fig_hist_kde"].append(fig_train)
    st.session_state["figs"]["data-heterogeneity"]["fig_hist_kde"].append(fig_test)

def generate_visual_topology(nus, side, zone_centers, user_positions=None):
    fig = go.Figure()

    # Scatter plot for RU positions
    fig.add_trace(go.Scatter(
        x=nus.real,
        y=nus.imag,
        mode='markers+text',
        marker=dict(color='orange', size=12),
        textposition="top right",
        name='accest points'
    ))

    # Scatter plot for zone centers
    fig.add_trace(go.Scatter(
        x=zone_centers.real,
        y=zone_centers.imag,
        mode='markers',
        marker=dict(color='green', size=12),
        name='zone centers'
    ))

    for u, center in enumerate(zone_centers):

        # Scatter plot for users in the zone
        if user_positions is not None:
            fig.add_trace(go.Scatter(
                x=user_positions.real.flatten(),
                y=user_positions.imag.flatten(),
                mode='markers',
                marker=dict(color='blue', size=4),
                name=f'clients',
                showlegend=u==0
            ))

        # Draw zone borders
        zone_x = [
            -0.5 * side + center.real,  0.5 * side + center.real,
            0.5 * side + center.real, -0.5 * side + center.real, -0.5 * side + center.real
        ]
        zone_y = [
            -0.5 * side + center.imag, -0.5 * side + center.imag,
             0.5 * side + center.imag,  0.5 * side + center.imag, -0.5 * side + center.imag
        ]
        fig.add_trace(go.Scatter(
            x=zone_x,
            y=zone_y,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

    # Draw system boundaries
    boundary_x = [
        -0.5 * side,  0.5 * side, 
         0.5 * side, -0.5 * side, -0.5 * side
    ]
    boundary_y = [
        -0.5 * side, -0.5 * side,
         0.5 * side,  0.5 * side, -0.5 * side
    ]
    fig.add_trace(go.Scatter(
        x=boundary_x,
        y=boundary_y,
        mode='lines',
        line=dict(color='black', width=2),
        name='System Boundary',
        showlegend=False
    ))

    # Adjust layout for better display
    fig.update_layout(
        title="System Topology",
        showlegend=True,
        legend=dict(
            x=1.05,  # Move the legend to the right of the graph
            y=0.5,     # Align it with the top
            orientation="v",  # Vertical legend
        ),
        autosize=False,
        width = 680,
        height = 680
    )

    # Set the axis limits
    fig.update_xaxes(range=[-side/2 - side - side*0.2, side/2 + side + side*0.2])
    fig.update_yaxes(range=[-side/2 - side - side*0.2, side/2 + side + side*0.2])

    # Enforce square aspect ratio and turn off axes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


# 4. Federated Learning Simulation
def run_federated_learning():

    progress_container_fl_bar = st.empty()
    progress_container_comm_bar = st.empty()
    plot_container1 = st.empty()
    plot_container2 = st.empty()

    fl_bar = progress_container_fl_bar.progress(0, text="FL rounds")
    config_loader = st.session_state["config_loader"]
    logger = st.session_state["logger"]
    fl_config = config_loader.get_fl_config()
    perfect_comm = config_loader.get_comm_config()["perfect_comm"]
    comm_bar = None if perfect_comm else progress_container_comm_bar.progress(0, text="Sub Comm rounds") 

    display_round_performance_plots_func = lambda df: display_round_performance_plots(df, plot_container1)
    display_round_number_of_selected_clients_func = lambda df: display_round_number_of_selected_clients(df, plot_container2, fl_config["num_clients"])

    simulate_ufl_for_GUI(
        st.session_state["clients"],
        st.session_state["server"], 
        st.session_state["topology"],
        st.session_state["data_handler"],
        config_loader,
        logger,
        fl_bar, 
        comm_bar,
        return_df_from_csv,
        display_round_performance_plots_func,
        display_round_number_of_selected_clients_func
    )    
    st.success("Federated learning completed.")

def return_df_from_csv(outfname):    
    return pd.read_csv(outfname)

def display_round_performance_plots(df, plot_container):
    # Update Accuracy Plot
    fig = px.line(df, x='Round', y='test_acc', markers=True, title="Accuracy vs FL Rounds")    
    fig.update_layout(title="Performance Results:", legend=dict(title="Legend"), xaxis_title="fl rounds", yaxis_title="test accuracy")
    plot_container.plotly_chart(fig, use_container_width=True)

def display_round_number_of_selected_clients(df, plot_container, num_clients):
    fig = px.line(
        df,
        x="Round", y="n_selected_clients",
        markers=True, title="Number of Selected Clients:",
        range_y = [0, num_clients]
    )
    fig.update_layout(
        xaxis_title="fl rounds", yaxis_title="number of selected clients"
    )
    plot_container.plotly_chart(fig, use_container_width=True)


# Streamlit App Execution
if __name__ == "__main__":
    display_header()
    configure_settings()
    initialize_network()
    if st.button("Run Federated Learning"):
        run_federated_learning()
