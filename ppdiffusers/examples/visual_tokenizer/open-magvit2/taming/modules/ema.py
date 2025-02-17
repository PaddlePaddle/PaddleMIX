import paddle
"""
Refer to 
https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/ema.py
"""


class LitEma(paddle.nn.Layer):

    def __init__(self, model, decay=0.999, use_num_upates=False):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.m_name2s_name = {}
        self.register_buffer(name='decay', tensor=paddle.to_tensor(data=
            decay, dtype='float32'))
        self.register_buffer(name='num_updates', tensor=paddle.to_tensor(
            data=0, dtype='int32') if use_num_upates else paddle.to_tensor(
            data=-1, dtype='int32'))
        for name, p in model.named_parameters():
            if not p.stop_gradient:
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(name=s_name, tensor=p.clone().detach()
                    .data)
        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer(name='num_updates', tensor=paddle.to_tensor(
            data=0, dtype='int32'))

    def forward(self, model):
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.
                num_updates))
        one_minus_decay = 1.0 - decay
        with paddle.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            for key in m_param:
                if '_layers' in key:
                    m_key = key.replace('_layers.', '')
                else:
                    m_key = key
                if not m_param[key].stop_gradient:
                    sname = self.m_name2s_name[m_key]
                    shadow_params[sname] = shadow_params[sname].astype(dtype
                        =m_param[key].dtype)
                    shadow_params[sname].subtract_(y=paddle.to_tensor(
                        one_minus_decay * (shadow_params[sname] - m_param[
                        key])))
                else:
                    assert not m_key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
   
        for key in m_param:
            if '_layers' in key:
                m_key = key.replace('_layers.', '')
            else:
                m_key = key
            if not m_param[key].stop_gradient:
                paddle.assign(x=shadow_params[self.m_name2s_name[m_key]],output=m_param[key])
            else:
                assert not m_key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            paddle.assign(x=c_param,output=param)

