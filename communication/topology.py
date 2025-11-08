import torch


class AccessPoint:
    # Represents an Access Point (AP) positioned at a zone boundary.
    def __init__(self, ap_id, position):
        self.id = ap_id             # Unique ID
        self.position = position    # Complex coordinate (x + 1j*y)


class Zone:
    # Represents a square-shaped zone in the system.
    def __init__(self, zone_id, center_position, side_length, nAPs):
        self.id = zone_id
        self.center = center_position
        self.side_length = side_length
        self.U = nAPs
        
        # Will be set later inside TUMAEnvironment
        self.blocklength = None
        self.codebook_size = None
        self.s = None 

    def Cx(self, X):
        # Applies encoding (Cx) operation. 
        return torch.matmul(self.C, X)
    
    def Cy(self, Y):
        # Applies decoding (Cy) operation. 
        return torch.matmul(self.C.conj().T, Y)
    

class NetworkTopology:
    # Manages the system topology, including zones and APs. 
    def __init__(self, side_length, device, jitter=0):
        self.side_length = side_length
        self.area_side = side_length*3
        self.rows = 3
        self.cols = 3
        self.U = self.rows * self.cols  # Number of zones
        self.B = None                   # Number of APs (set after generating APs)
        self.jitter = jitter            # Random perturbation for AP positions
        self.device = device

        self.zones = self.generate_zones()
        self.aps = self.generate_aps()
        
        self.ap_positions = torch.tensor([ap.position for ap in self.aps], device=device)

    # -------------------------------------------------------------
    def generate_zones(self):
        # Generate zone centroids for a 3x3 grid.
        zone_centers_x = torch.arange(self.cols, device=self.device, dtype=torch.float32) - (self.cols - 1) / 2
        zone_centers_y = torch.arange(self.rows, device=self.device, dtype=torch.float32) - (self.rows - 1) / 2
        xx, yy = torch.meshgrid(zone_centers_x, zone_centers_y, indexing='xy')
        zone_centers = (xx + 1j * yy) * self.side_length

        # Create Zone objects
        zones = [Zone(zone_id=i, center_position=center, side_length=self.side_length, nAPs=self.U) 
                 for i, center in enumerate(zone_centers.flatten())]
        return zones

    # -------------------------------------------------------------
    def generate_aps(self):
        """Generate APs positioned on zone boundaries, with extra APs between each adjacent AP."""
        # Horizontal AP positions
        y_rows = torch.arange(self.rows + 1, device=self.device, dtype=torch.float32)
        y_positions = (y_rows - self.rows / 2) * self.side_length
        x_positions = torch.linspace(-self.cols / 2 * self.side_length,
                                     self.cols / 2 * self.side_length,
                                     steps=2 * self.cols + 1,
                                     device=self.device, dtype=torch.float32)
        xh, yh = torch.meshgrid(x_positions, y_positions, indexing="xy")
        horizontal = torch.stack([xh.flatten(), yh.flatten()], dim=1)

        # Vertical AP positions
        x_cols = torch.arange(self.cols + 1, device=self.device, dtype=torch.float32)
        x_positions_v = (x_cols - self.cols / 2) * self.side_length
        y_positions_v = torch.linspace(-self.rows / 2 * self.side_length,
                                       self.rows / 2 * self.side_length,
                                       steps=2 * self.rows + 1,
                                       device=self.device, dtype=torch.float32)
        xv, yv = torch.meshgrid(x_positions_v, y_positions_v, indexing="xy")
        vertical = torch.stack([xv.flatten(), yv.flatten()], dim=1)

        # Combine and remove duplicates
        aps_all = torch.cat([horizontal, vertical], dim=0)
        aps_all = torch.round(aps_all * 1e6) / 1e6
        aps_unique = torch.unique(aps_all, dim=0)

        # Lexicographic sort by (x, y)
        sort_keys = aps_unique[:, 0] * 1e6 + aps_unique[:, 1]
        sort_idx = torch.argsort(sort_keys)
        aps_unique = aps_unique[sort_idx]

        # Optionally add jitter
        if self.jitter > 0:
            aps_unique += self.jitter * torch.randn_like(aps_unique)

        # Convert to AccessPoint objects
        aps = [
            AccessPoint(i, complex(float(x), float(y)))
            for i, (x, y) in enumerate(aps_unique.cpu())
        ]
        self.B = len(aps)        
        return aps
    
    # -------------------------------------------------------------
    def setup_zone_codebooks(self, blocklength, codebook_size):
        for zone in self.zones:
            zone.blocklength = blocklength
            zone.codebook_size = codebook_size
            zone.C = self.generate_codebook(blocklength, codebook_size)
        self.C_blocks = torch.stack([zone.C for zone in self.zones], dim=0)
        self.CH_blocks = torch.stack([zone.C.conj().T for zone in self.zones], dim=0)

    def generate_codebook(self, blocklength, codebook_size):
        """ Generates a complex-valued normalized codebook matrix C. """    
        C = torch.randn(blocklength, codebook_size, device=self.device) + 1j * torch.randn(blocklength, codebook_size, device=self.device)
        return C / torch.linalg.norm(C, dim=0, keepdim=True)

    # -------------------------------------------------------------
    def get_zone_by_position(self, position):
        # Determines which zone a given position belongs to. 
        for zone in self.zones:
            x_min = zone.center.real - zone.side_length / 2
            x_max = zone.center.real + zone.side_length / 2
            y_min = zone.center.imag - zone.side_length / 2
            y_max = zone.center.imag + zone.side_length / 2

            if x_min <= position.real < x_max and y_min <= position.imag < y_max:
                return zone.id
        return None  # Out of bounds

    # -------------------------------------------------------------
    def assign_zones_to_clients(self, clients):
        for client in clients:
            zone_id = self.get_zone_by_position(client.pos)
            client.assigned_zone = self.zones[zone_id]

    # -------------------------------------------------------------
    def assign_LSFCs_to_clients(self, clients, ref_distance, path_loss_exp, num_antennas):
        for client in clients:
            client.LSFCs = (self.compute_path_loss(client.pos, ref_distance, path_loss_exp) * torch.ones((1, num_antennas), device=self.device)).reshape(-1)

    def compute_path_loss(self, pos, ref_distance, path_loss_exp):
        """ Computes path loss based on sensor position and AP locations. """
        distances = torch.abs(pos - self.ap_positions.reshape(-1, 1))  
        return 1 / (1 + (distances / ref_distance) ** path_loss_exp) # Path loss model

    # -------------------------------------------------------------
    def visualize(self):
        # Visualizes zones and APs.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))

        # Plot Zones
        zone_positions = torch.tensor([zone.center for zone in self.zones])
        plt.scatter(zone_positions.real, zone_positions.imag, c='blue', marker='s', s=100, label="Zones (Centers)")

        # Plot APs
        ap_positions = torch.tensor([ap.position for ap in self.aps])
        plt.scatter(ap_positions.real, ap_positions.imag, c='orange', marker='o', label="Access Points (APs)", s=100)

        # Draw zone boundaries
        for zone in self.zones:
            rect = plt.Rectangle((zone.center.real - zone.side_length / 2, zone.center.imag - zone.side_length / 2),
                                 zone.side_length, zone.side_length, linewidth=1, edgecolor='gray', facecolor='none')
            plt.gca().add_patch(rect)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title("Network Topology (Zones and APs)")
        plt.ylim([-self.rows*self.side_length/2 - self.rows*self.side_length/20, self.rows*self.side_length/2 + self.rows*self.side_length/20])
        plt.xlim([-self.cols*self.side_length/2 - self.cols*self.side_length/20, self.cols*self.side_length/2 + self.cols*self.side_length/20])
        
        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])
        
        plt.grid(True)
        plt.show()
