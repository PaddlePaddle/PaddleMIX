import numpy as np
import torch
import paddle

import sys; sys.path.append("~/Desktop/PaddleMIX/paddlemix/models/vits-svc")
from speaker.utils.io import load_fsspec_torch



class LSTMWithProjection_torch(torch.nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, proj_size, bias=False)

    def forward(self, x):
        # self.lstm.flatten_parameters()
        o, (_, _) = self.lstm(x)
        return self.linear(o)



class LSTMWithProjection(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.lstm = paddle.nn.LSTM(input_size, hidden_size) # batch_first=True
        self.linear = paddle.nn.Linear(hidden_size, proj_size, bias_attr=False)

    def forward(self, x):
        # self.lstm.flatten_parameters()
        o, (_, _) = self.lstm(x)
        return self.linear(o)



def LSTMWithProjection_torch2paddle(lstm_paddle, lstm_torch):


    # pd_model_state_dict = lstm_paddle.state_dict()
    pd_model_state_dict = {}
    tc_model_state_dict = lstm_torch.state_dict()

    # print(
    #     pd_model_state_dict['lstm.weight_ih_l0'] is pd_model_state_dict['lstm.0.cell.weight_ih'],
    #     pd_model_state_dict['lstm.weight_hh_l0'] is pd_model_state_dict['lstm.0.cell.weight_hh'],
    #     pd_model_state_dict['lstm.bias_ih_l0'] is pd_model_state_dict['lstm.0.cell.bias_ih'],
    #     pd_model_state_dict['lstm.bias_hh_l0'] is pd_model_state_dict['lstm.0.cell.bias_hh']
    # )

    pd_model_state_dict['lstm.weight_ih_l0'] = paddle.to_tensor(
        tc_model_state_dict['lstm.weight_ih_l0'].detach().cpu().numpy()
    )
    pd_model_state_dict['lstm.weight_hh_l0'] = paddle.to_tensor(
        tc_model_state_dict['lstm.weight_hh_l0'].detach().cpu().numpy()
    )
    pd_model_state_dict['lstm.bias_ih_l0'] = paddle.to_tensor(
        tc_model_state_dict['lstm.bias_ih_l0'].detach().cpu().numpy()
    )
    pd_model_state_dict['lstm.bias_hh_l0'] = paddle.to_tensor(
        tc_model_state_dict['lstm.bias_hh_l0'].detach().cpu().numpy()
    )

    # # -------------------------------------------
    pd_model_state_dict['lstm.0.cell.weight_ih'] = paddle.to_tensor(
        tc_model_state_dict['lstm.weight_ih_l0'].detach().cpu().numpy()
    )
    pd_model_state_dict['lstm.0.cell.weight_hh'] = paddle.to_tensor(
        tc_model_state_dict['lstm.weight_hh_l0'].detach().cpu().numpy()
    )
    pd_model_state_dict['lstm.0.cell.bias_ih'] = paddle.to_tensor(
        tc_model_state_dict['lstm.bias_ih_l0'].detach().cpu().numpy()
    )
    pd_model_state_dict['lstm.0.cell.bias_hh'] = paddle.to_tensor(
        tc_model_state_dict['lstm.bias_hh_l0'].detach().cpu().numpy()
    )

    lstm_paddle.load_dict(pd_model_state_dict)

    lstm_paddle.linear.weight.set_value(
        paddle.to_tensor( lstm_torch.linear.weight.data.cpu().numpy().T )
    )

    return lstm_paddle 


# if __name__ == "__main__":

#     # ---------- 测试结果 ----------
#     input_size, hidden_size, proj_size = 80, 768, 256

#     # lstm 模型
#     lstm_paddle = LSTMWithProjection(input_size, hidden_size, proj_size)
#     lstm_torch  = LSTMWithProjection_torch(input_size, hidden_size, proj_size).cuda()

#     # lstm 参数传递
#     lstm_paddle = LSTMWithProjection_torch2paddle(lstm_paddle, lstm_torch)

#     # 输入参数
#     x = np.random.rand(10, 250, 80).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)

#     lstm_paddle.lstm.could_use_cudnn = False

#     y_pd, (_, _) = lstm_paddle.lstm(x_pd)
#     y_tc, (_, _) = lstm_torch.lstm(x_tc)

#     y_pd = y_pd.numpy()
#     y_tc = y_tc.detach().cpu().numpy()

#     print(
#         abs(
#             y_pd - y_tc
#         ).max()
#     )

#     y_pd = lstm_paddle(x_pd)
#     y_tc = lstm_torch(x_tc)

#     y_pd = y_pd.numpy()
#     y_tc = y_tc.detach().cpu().numpy()

#     print(
#         abs(
#             y_pd - y_tc
#         ).max(),
#         f"mean: {y_pd.mean() - y_tc.mean()}",
#         f"std : {y_pd.std() - y_tc.std()}",
#     )




# class LSTMWithoutProjection(torch.nn.Module):
#     def __init__(self, input_dim, lstm_dim, proj_dim, num_lstm_layers):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim, num_layers=num_lstm_layers, batch_first=True)
#         self.linear = nn.Linear(lstm_dim, proj_dim, bias=True)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         _, (hidden, _) = self.lstm(x)
#         return self.relu(self.linear(hidden[-1]))


class LSTMSpeakerEncoder_torch(torch.nn.Module):
    def __init__(self, input_dim, proj_dim=256, lstm_dim=768, num_lstm_layers=3, use_lstm_with_projection=True):
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection
        layers = []
        # choise LSTM layer
        if use_lstm_with_projection:
            layers.append(LSTMWithProjection_torch(input_dim, lstm_dim, proj_dim))
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection_torch(proj_dim, lstm_dim, proj_dim))
            self.layers = torch.nn.Sequential(*layers)
        else:
            # self.layers = LSTMWithoutProjection(input_dim, lstm_dim, proj_dim, num_lstm_layers)
            pass

        self._init_layers()

    def _init_layers(self):
        for name, param in self.layers.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_normal_(param)

    # def forward(self, x):
    #     # TODO: implement state passing for lstms
    #     d = self.layers(x)
    #     if self.use_lstm_with_projection:
    #         d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
    #     else:
    #         d = torch.nn.functional.normalize(d, p=2, dim=1)
    #     return d

    @torch.no_grad()
    def inference(self, x): #
        print("torch", x.mean().item(), x.std().item())
        d = self.layers.forward(x)
        if self.use_lstm_with_projection:
            d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        else:
            d = torch.nn.functional.normalize(d, p=2, dim=1)
        return d

    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True): # 
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)

        return embeddings

    # def batch_compute_embedding(self, x, seq_lens, num_frames=160, overlap=0.5):
    #     """
    #     Generate embeddings for a batch of utterances
    #     x: BxTxD
    #     """
    #     num_overlap = num_frames * overlap
    #     max_len = x.shape[1]
    #     embed = None
    #     num_iters = seq_lens / (num_frames - num_overlap)
    #     cur_iter = 0
    #     for offset in range(0, max_len, num_frames - num_overlap):
    #         cur_iter += 1
    #         end_offset = min(x.shape[1], offset + num_frames)
    #         frames = x[:, offset:end_offset]
    #         if embed is None:
    #             embed = self.inference(frames)
    #         else:
    #             embed[cur_iter <= num_iters, :] += self.inference(frames[cur_iter <= num_iters, :, :])
    #     return embed / num_iters

    # pylint: disable=unused-argument, redefined-builtin
    def load_checkpoint(self, checkpoint_path: str, eval: bool = False, use_cuda: bool = False):
        state = load_fsspec_torch(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if use_cuda:
            self.cuda()
        if eval:
            self.eval()
            assert not self.training


class LSTMSpeakerEncoder(paddle.nn.Layer):
    def __init__(self, input_dim, proj_dim=256, lstm_dim=768, num_lstm_layers=3, use_lstm_with_projection=True):
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection
        layers = []
        # choise LSTM layer
        if use_lstm_with_projection:
            layers.append(LSTMWithProjection(input_dim, lstm_dim, proj_dim))
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection(proj_dim, lstm_dim, proj_dim))
            self.layers = paddle.nn.Sequential(*layers)
        else:
            # self.layers = LSTMWithoutProjection(input_dim, lstm_dim, proj_dim, num_lstm_layers)
            raise NotImplementedError()

        self._init_layers()

    def _init_layers(self):
        for name, param in self.layers.named_parameters():
            if 'bias' in name:
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(param)
            elif 'weight' in name:
                init_XavierNormal = paddle.nn.initializer.XavierNormal()
                init_XavierNormal(param)



    # def forward(self, x):
    #     # TODO: implement state passing for lstms
    #     d = self.layers(x)
    #     if self.use_lstm_with_projection:
    #         d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
    #     else:
    #         d = torch.nn.functional.normalize(d, p=2, dim=1)
    #     return d

    @paddle.no_grad()
    def inference(self, x): #
        print("paddle", x.mean().item(), x.std().item())
        d = self.layers.forward(x)
        if self.use_lstm_with_projection:
            d = paddle.nn.functional.normalize(d[:, -1], p=2, axis=1)
        else:
            d = paddle.nn.functional.normalize(d, p=2, axis=1)
        return d

    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True): # 
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        # frames_batch = torch.cat(frames_batch, dim=0)
        frames_batch = paddle.concat(frames_batch, axis=0)
        embeddings = self.inference(frames_batch)

        if return_mean:
            # embeddings = torch.mean(embeddings, dim=0, keepdim=True)
            embeddings = paddle.mean(embeddings, axis=0, keepdim=True)

        return embeddings

    # def batch_compute_embedding(self, x, seq_lens, num_frames=160, overlap=0.5):
    #     """
    #     Generate embeddings for a batch of utterances
    #     x: BxTxD
    #     """
    #     num_overlap = num_frames * overlap
    #     max_len = x.shape[1]
    #     embed = None
    #     num_iters = seq_lens / (num_frames - num_overlap)
    #     cur_iter = 0
    #     for offset in range(0, max_len, num_frames - num_overlap):
    #         cur_iter += 1
    #         end_offset = min(x.shape[1], offset + num_frames)
    #         frames = x[:, offset:end_offset]
    #         if embed is None:
    #             embed = self.inference(frames)
    #         else:
    #             embed[cur_iter <= num_iters, :] += self.inference(frames[cur_iter <= num_iters, :, :])
    #     return embed / num_iters

    # pylint: disable=unused-argument, redefined-builtin

    def load_checkpoint(self, checkpoint_path: str, eval: bool = False):
        # state = load_fsspec(checkpoint_path)
        ckpt = paddle.load( checkpoint_path )
        self.set_state_dict( ckpt )
        if eval:
            self.eval()
            assert not self.training

        # TODO: https://github.com/PaddlePaddle/Paddle/issues/64989 
        for pd_layer in self.layers:
            pd_layer.lstm.could_use_cudnn = False


# if __name__ == "__main__":

#     tc_model = LSTMSpeakerEncoder_torch(80, 256, 768, 3).cuda()
#     pd_model = LSTMSpeakerEncoder(80, 256, 768, 3)


#     model_path = "speaker_pretrain/best_model.pth.tar"
#     tc_model.load_checkpoint(model_path, eval=True, use_cuda=True)

#     # model_path = "speaker_pretrain/best_model.pdparam"
#     # pd_model.load_checkpoint(model_path, eval=True)

#     x = np.random.randn(1, 212, 80).astype("float32")
#     x_tc = torch.from_numpy(x).cuda()
#     x_pd = paddle.to_tensor(x)


#     for pd_layer, tc_layer in zip(pd_model.layers, tc_model.layers):
#         pd_layer.lstm.could_use_cudnn = False
#         LSTMWithProjection_torch2paddle(pd_layer, tc_layer)


#     y_tc = tc_model.compute_embedding(x_tc).detach().cpu().numpy()
#     y_pd = pd_model.compute_embedding(x_pd).detach().cpu().numpy()

#     print(
#         abs(y_tc - y_pd).max(),
#         f"{y_tc.mean().item()} {y_pd.mean().item()}",
#         f"{y_tc.std().item()} {y_pd.std().item()}",
#     )

#     paddle.save(
#         pd_model.state_dict(),
#         "speaker_pretrain/best_model.pdparam"
#     )


if __name__ == "__main__":

    tc_model = LSTMSpeakerEncoder_torch(80, 256, 768, 3).cuda()
    pd_model = LSTMSpeakerEncoder(80, 256, 768, 3)


    model_path = "speaker_pretrain/best_model.pth.tar"
    tc_model.load_checkpoint(model_path, eval=True, use_cuda=True)

    model_path = "speaker_pretrain/best_model.pdparam"
    pd_model.load_checkpoint(model_path, eval=True)

    x = np.random.randn(1, 212, 80).astype("float32")
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)


    # for pd_layer, tc_layer in zip(pd_model.layers, tc_model.layers):
    #     # TODO: https://github.com/PaddlePaddle/Paddle/issues/64989
    #     pd_layer.lstm.could_use_cudnn = False
    #     # LSTMWithProjection_torch2paddle(pd_layer, tc_layer)


    y_tc = tc_model.compute_embedding(x_tc).detach().cpu().numpy()
    y_pd = pd_model.compute_embedding(x_pd).detach().cpu().numpy()

    print(
        abs(y_tc - y_pd).max(),
        f"\nmean: {y_tc.mean().item()} {y_pd.mean().item()}",
        f"\nstd:  {y_tc.std().item()} {y_pd.std().item()}",
    )

    # paddle.save(
    #     pd_model.state_dict(),
    #     "speaker_pretrain/best_model.pdparam"
    # )
    
    print(
        pd_model_state_dict['lstm.weight_ih_l0'] is pd_model_state_dict['lstm.0.cell.weight_ih'],
        pd_model_state_dict['lstm.weight_hh_l0'] is pd_model_state_dict['lstm.0.cell.weight_hh'],
        pd_model_state_dict['lstm.bias_ih_l0'] is pd_model_state_dict['lstm.0.cell.bias_ih'],
        pd_model_state_dict['lstm.bias_hh_l0'] is pd_model_state_dict['lstm.0.cell.bias_hh']
    )