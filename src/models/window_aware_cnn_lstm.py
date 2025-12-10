import torch


class WindowAware_DeepCNN_LSTM(nn.Module):
    """
    Window-aware Deep CNN-LSTM for pain threshold detection.
    - Deep CNN for spatial feature extraction from EEG
    - Window embedding to encode temporal context (baseline/erp/post)
    - Bidirectional LSTM for temporal modeling
    - Fusion of EEG features and window embeddings
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128], 
                 lstm_hidden: int = 192, 
                 lstm_layers: int = 2,
                 window_embed_dim: int = 16,
                 dropout: float = 0.4):
        super().__init__()
        
        # Window embedding
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)
        
        # Deep CNN for hierarchical spatial features
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        
        # Calculate output time dimension after pooling
        time_reduced = n_time // 8
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        
        # LSTM input: CNN features + window embedding broadcasted
        lstm_input_dim = cnn_filters[2] + window_embed_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden, batch_first=True, 
                           num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0, 
                           bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier combines LSTM output and window embedding
        self.fc = nn.Linear(lstm_hidden * 2 + window_embed_dim, n_classes)
    
    def forward(self, x, window_idx):
        # x: (batch, channels, time)
        # window_idx: (batch,) - integer indices for window type
        batch_size = x.size(0)
        
        # Get window embeddings
        window_emb = self.window_embedding(window_idx)  # (batch, window_embed_dim)
        
        # Deep CNN feature extraction
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)  # (batch, cnn_filters[2], 1, time_reduced)
        
        # Reshape for LSTM: (batch, time_reduced, cnn_filters[2])
        x = x.squeeze(2).transpose(1, 2)
        seq_len = x.size(1)
        
        # Concatenate window embedding to each time step
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, window_embed_dim)
        x = torch.cat([x, window_emb_expanded], dim=2)  # (batch, seq_len, cnn_filters[2] + window_embed_dim)
        
        # LSTM temporal modeling
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, lstm_hidden*2)
        
        # Concatenate LSTM output with window embedding for final classification
        h = torch.cat([h, window_emb], dim=1)  # (batch, lstm_hidden*2 + window_embed_dim)
        
        # Classification
        h = self.dropout(h)
        return self.fc(h)

class WindowAware_CNN_Transformer(nn.Module):
    """
    Window-aware CNN-Transformer for pain threshold detection.
    - Deep CNN for spatial feature extraction from EEG
    - Window embedding to encode temporal context (baseline/erp/post)
    - Transformer encoder for temporal attention
    - Fusion of EEG features and window embeddings
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128],
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 window_embed_dim: int = 16,
                 dropout: float = 0.3,
                 time_downsample: int = 4):
        super().__init__()
        
        # Window embedding
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)
        
        # Deep CNN for hierarchical spatial features
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        
        # Calculate output time dimension after pooling
        time_reduced = n_time // 8
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        
        # Additional time downsampling for transformer
        self.time_downsample = time_downsample
        time_for_transformer = time_reduced // time_downsample
        
        # Project CNN features to transformer dimension
        self.cnn_to_transformer = nn.Linear(cnn_filters[2] + window_embed_dim, d_model)
        
        # Positional encoding
        max_len = time_for_transformer + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier combines Transformer output and window embedding
        self.fc = nn.Linear(d_model + window_embed_dim, n_classes)
    
    def forward(self, x, window_idx):
        # x: (batch, channels, time)
        # window_idx: (batch,) - integer indices for window type
        batch_size = x.size(0)
        
        # Get window embeddings
        window_emb = self.window_embedding(window_idx)  # (batch, window_embed_dim)
        
        # Deep CNN feature extraction
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)  # (batch, cnn_filters[2], 1, time_reduced)
        
        # Reshape: (batch, time_reduced, cnn_filters[2])
        x = x.squeeze(2).transpose(1, 2)
        
        # Downsample time dimension
        T = x.size(1)
        T_down = (T // self.time_downsample) * self.time_downsample
        x = x[:, :T_down, :]
        x = x.reshape(batch_size, T_down // self.time_downsample, -1, self.time_downsample)
        x = x.mean(dim=3)  # Average pool over downsample factor
        
        # Concatenate window embedding to each time step
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, window_emb_expanded], dim=2)  # (batch, seq_len, cnn_filters[2] + window_embed_dim)
        
        # Project to transformer dimension
        x = self.cnn_to_transformer(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer temporal attention
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Concatenate with window embedding for final classification
        x = torch.cat([x, window_emb], dim=1)  # (batch, d_model + window_embed_dim)
        
        # Classification
        x = self.dropout(x)
        return self.fc(x)
