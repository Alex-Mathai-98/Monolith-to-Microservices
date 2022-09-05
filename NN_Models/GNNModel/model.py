import torch
import torch.nn as nn
from NN_Models.GNNModel.egcn import EGCNConv,get_activation

class GNNStack(torch.nn.Module):
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs, aggr
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)
        self.activation_layer = get_activation(activation)

        # convs
        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation, aggr)

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim, edge_mode,
                     model_types, normalize_embs, activation, aggr):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0],node_input_dim,node_dim,
                                    edge_input_dim, edge_mode, normalize_embs[0], activation, aggr)
        convs.append(conv)
        #print(model_types)
        for l in range(1,len(model_types)):
            conv = self.build_conv_model(model_types[l],node_dim, node_dim,
                                    edge_dim, edge_mode, normalize_embs[l], activation, aggr)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode, normalize_emb, activation, aggr):
        
        if model_type == 'EGCNShared':
            return EGCNConv(node_in_dim,node_out_dim,edge_dim,edge_mode,activation,shared_weight=True)
        elif model_type == 'EGCNSeparate':
            return EGCNConv(node_in_dim,node_out_dim,edge_dim,edge_mode,activation,shared_weight=False)

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+edge_input_dim,edge_dim),
                get_activation(activation),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat( ((x_i+x_j)/2,edge_attr) ,dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index, edge_weight=None):

        if self.concat_states:
            concat_x = []
        
        for l,(conv_name,conv) in enumerate(zip(self.model_types,self.convs)):
            
            # self.check_input(x,edge_attr,edge_index)

            if ('EGCN' in conv_name) : 
                x = conv(x, edge_attr, edge_index, edge_weight)
            elif conv_name == 'EGSAGE':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            if self.concat_states:
                concat_x.append(x)

            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
            
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        
        return x,edge_attr

class AE_EGCN(nn.Module) :

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, model_type) :
        super(AE_EGCN, self).__init__()
        # encoders
        self.encgc1 = GNNStack(node_input_dim=input_feat_dim, edge_input_dim=input_feat_dim,
                            node_dim = hidden_dim1, edge_dim = hidden_dim1, edge_mode = 1, model_types = [model_type], 
                            dropout=0., activation='prelu', concat_states=False, node_post_mlp_hiddens=[hidden_dim1],
                            normalize_embs=[False], aggr=None)

        self.encgc2 = GNNStack(node_input_dim=hidden_dim1, edge_input_dim=hidden_dim1,
                            node_dim = hidden_dim2, edge_dim = hidden_dim2, edge_mode = 1, model_types = [model_type], 
                            dropout=0., activation='prelu', concat_states=False, node_post_mlp_hiddens=[hidden_dim2],
                            normalize_embs=[False], aggr=None)

        # decoders
        self.decgc1 = GNNStack(node_input_dim=hidden_dim2, edge_input_dim=hidden_dim2,
                            node_dim = hidden_dim1, edge_dim = hidden_dim1, edge_mode = 1, model_types = [model_type], 
                            dropout=0., activation='prelu', concat_states=False, node_post_mlp_hiddens=[hidden_dim1],
                            normalize_embs=[False], aggr=None)

        self.decgc2 = GNNStack(node_input_dim=hidden_dim1, edge_input_dim=hidden_dim1,
                            node_dim = input_feat_dim, edge_dim = input_feat_dim, edge_mode = 1, model_types = [model_type], 
                            dropout=0., activation='prelu', concat_states=False, node_post_mlp_hiddens=[input_feat_dim],
                            normalize_embs=[False], aggr=None)

    def encode(self, x, edge_features, edge_index, edge_weight):
        hidden1,edge_attr1 = self.encgc1(x,edge_features,edge_index, edge_weight=edge_weight)
        hidden2,edge_attr2 = self.encgc2(hidden1,edge_attr1,edge_index, edge_weight=edge_weight)        
        return hidden2,edge_attr2
    
    def decode(self, hidden, edge_features, edge_index, edge_weight):
        hidden1,edge_attr1 = self.decgc1(hidden, edge_features, edge_index, edge_weight=edge_weight)
        recon,edge_attr2 = self.decgc2(hidden1, edge_attr1, edge_index, edge_weight=edge_weight)        
        return recon,edge_attr2

    def forward(self, x, edge_features, edge_index, edge_weight):
        enc,enc_edge_attr = self.encode(x, edge_features, edge_index, edge_weight)
        dec,dec_edge_attr = self.decode(enc,enc_edge_attr,edge_index, edge_weight)
        return dec, enc, dec_edge_attr

class AE_EGCN_Shared(AE_EGCN) :
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2) :
        super(AE_EGCN_Shared,self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, "EGCNShared")

class AE_EGCN_Separate(AE_EGCN) :
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2) :
        super(AE_EGCN_Separate,self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, "EGCNSeparate")