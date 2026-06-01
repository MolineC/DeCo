import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.node_encoder = TSP_Encoder(**model_params)
        self.tw_encoder = TW_Encoder(**model_params)
        self.fusion=nn.Linear(256,128)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step,tw_norm = True,tw_mask = None,greedy = True):
        node_xy = state.node_xy
        node_tw=state.node_tw
        current_time=state.current_time
        if tw_norm:
            tw_end_max =  state.node_tw[:,:1,1]
            tw_start  = state.node_tw[:,:,0]  / tw_end_max
            tw_end  = state.node_tw[:,:,1]  / tw_end_max
            node_tw=torch.cat([tw_start[:,:,None],tw_end[:,:,None]],dim=-1)
            # data=torch.cat([node_xy,node_tw],dim=-1)
            tw_end_max=tw_end_max.squeeze(-1)
            current_time=state.current_time/tw_end_max
        batch_size_V = selected_node_list.shape[0]
        if self.mode == 'train':
            encoded_nodes = self.node_encoder(node_xy) 
            encoded_tw = self.tw_encoder(node_tw,current_time)
            self.encoded_nodes = self.fusion(torch.cat((encoded_nodes,encoded_tw),dim=2))
            probs = self.decoder(self.encoded_nodes, selected_node_list,tw_mask,current_time)
            selected_student = probs.argmax(dim=1)  # shape: B
            selected_teacher = solution[:, current_step]  # shape: B
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)  # shape: [B, 1]
        if self.mode == 'test':
                if current_step <= 1:
                    self.encoded_nodexy = self.node_encoder(node_xy)
                encoded_tw = self.tw_encoder(node_tw,current_time)
                self.encoded_nodes = self.fusion(torch.cat((self.encoded_nodexy,encoded_tw),dim=2))
                probs = self.decoder(self.encoded_nodes,selected_node_list,tw_mask,current_time)
                if tw_mask!=None:
                    probs_clone=probs.clone()
                    mask=tw_mask==float('-inf')
                    probs[mask]=0
                    infeasible_mask=mask.sum(dim=1)==node_xy.shape[1]
                    probs[infeasible_mask]=probs_clone[infeasible_mask]
                if greedy:
                    selected_student = probs.argmax(dim=1)
                else:
                    categorical = dist.Categorical(probs=probs)
                    selected_student = categorical.sample()
                selected_teacher = selected_student
                prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)  # shape: [B, 1]
        return selected_teacher, prob, probs, selected_student
########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(1)])



    def forward(self, data):

        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out

class TW_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        # 时间窗的4个维度
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(1)])



    def forward(self, data,current_time):
        current_time_expanded = current_time.unsqueeze(1).unsqueeze(1)  
        processed = torch.relu(data - current_time_expanded)  # 形状保持[batch, problem_nums, 2]
        result =processed
        embedded_input = self.embedding(result)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(4)])

        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)


    def _get_new_data(self, data, selected_node_list, problem_size, B_V,mask):

        list = selected_node_list
     
        new_list = torch.arange(problem_size, device=data.device)[None, :].repeat(B_V, 1)
    
        unselected_len = problem_size - list.shape[1]  # shape: [B, V-current_step]

        index_2 = list.type(torch.long)

        index_1 = torch.arange(B_V, dtype=torch.long, device=data.device)[:, None].expand(B_V, index_2.shape[1])
  
        new_list[index_1, index_2] = -2
        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, -1)
        # ----------------------------------------------------------------------------
        new_data = data
        emb_dim = data.shape[-1]
        new_data_len = unselect_list.shape[1]
        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
        index_1_ = torch.arange(B_V, dtype=torch.long, device=data.device)[:, None].expand(B_V, index_2_.shape[1])

        index_3_ = torch.arange(emb_dim, device=data.device)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_

    def _get_encoding(self,encoded_nodes, node_index_to_pick):

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes

    def forward(self,data,selected_node_list,mask = None,current_time=None):

        current_time = current_time.unsqueeze(1)
        batch_size_V = data.shape[0]  # B
        problem_size = data.shape[1]
        new_data = data

        left_encoded_node = self._get_new_data(new_data,selected_node_list, problem_size, batch_size_V,mask)

        first_and_last_node = self._get_encoding(new_data,selected_node_list[:,[0,-1]])
        embedded_first_node_ = first_and_last_node[:,0]
        embedded_last_node_ = first_and_last_node[:,1]
        #------------------------------------------------

        embedded_first_node_ = self.embedding_first_node(embedded_first_node_)

        embedded_last_node_=torch.cat((embedded_last_node_,current_time),dim=1)
        embedded_last_node_ = self.embedding_last_node(embedded_last_node_)

        out = torch.cat((embedded_first_node_.unsqueeze(1), left_encoded_node,embedded_last_node_.unsqueeze(1)), dim=1)

        layer_count=0

        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        out = self.Linear_final(out).squeeze(-1)
        out[:, [0,-1]] = out[:, [0,-1]] + float('-inf')

        props = F.softmax(out, dim=-1)

        props = props[:, 1:-1]
        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype, device=props_clone.device)  # prevent the probability from being too small
        props = props_clone

        All_props = torch.zeros(batch_size_V, problem_size, device=data.device)

        index_1_ = torch.arange(batch_size_V, dtype=torch.long, device=data.device)[:, None].expand(batch_size_V, selected_node_list.shape[1])  # shape: [B*(V-1), n]
        index_2_ = selected_node_list.type(torch.long)  
        All_props[index_1_, index_2_] = -2
        
        index=All_props!=-2
        All_props[index] = props.ravel()
        All_props[index_1_,index_2_]=0
        return All_props

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        model_params['embedding_dim']=embedding_dim
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v):
    """
    Flash-Attention-compatible multi-head attention.

    Input shape:  q/k/v = [B, H, N, D]
    Output shape:       [B, N, H*D]

    torch.nn.functional.scaled_dot_product_attention will automatically
    dispatch to Flash Attention / memory-efficient attention / math backend
    depending on PyTorch version, device, dtype, head_dim, and mask settings.
    """

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    # SDPA internally performs: softmax(q @ k^T / sqrt(D)) @ v
    # On CUDA with fp16/bf16 and supported shapes, this can use Flash Attention.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False
    )  # [B, H, N, D]

    out_transposed = out.transpose(1, 2).contiguous()  # [B, N, H, D]
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # [B, N, H*D]

    return out_concat



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
 