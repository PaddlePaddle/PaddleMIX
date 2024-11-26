import torch
import paddle

ckpt_path = '/root/paddlejob/workspace/env_run/zhouhao21/code/Open-MAGVIT2/init.ckpt'

sd = torch.load(ckpt_path, map_location="cpu")['state_dict']

# torch_ckpt_path = '/root/paddlejob/workspace/env_run/zhouhao21/code/Open-MAGVIT2/1_init.ckpt'

# torch.save({'state_dict': sd}, torch_ckpt_path)
# exit()

new_dict={}
skip_keys = ['transformer.pre_emb.weight','transformer.post_emb.weight','transformer.class_emb.embedding_table.weight',]
for k, v in sd.items():
    print(k,v.shape)
    value = v.numpy()
    if len(v.shape)==2:
       
        value = value.T

    new_dict[k] = value

#paddle.save(new_dict, '/root/paddlejob/workspace/env_run/zhouhao21/code/pp-Open-MAGVIT2/taming/modules/autoencoder/lpips/vgg.pdparams')
paddle.save(new_dict,"./init.pdparams")
        