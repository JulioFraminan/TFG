import torch, os, h5py
p='output/unet_ae_model.pt'
if not os.path.isfile(p):
    p='/home/j.framinan/TFG_repo/unet_hibrido/output/unet_ae_model.pt'
cp=torch.load(p,map_location='cpu')
print('keys=', list(cp.keys()))
for k in ['tl_min','tl_max','angle_min','angle_max']:
    print(k, cp.get(k, 'MISSING'))

vf='input/validation'
if os.path.isdir(vf):
    files=[f for f in sorted(os.listdir(vf)) if f.endswith('.mat') and 'PlaneAngle' in f]
    if files:
        fpath=os.path.join(vf,files[0])
        print('validation sample file:', files[0])
        try:
            with h5py.File(fpath,'r') as fh:
                def sg(f, keys):
                    for kk in keys:
                        if kk in f:
                            return f[kk][:]
                    raise KeyError(keys)
                tl=sg(fh, ['tl','TL','tL','tl_block'])
                print('sample tl shape, min, max:', tl.T.shape, float(tl.T.min()), float(tl.T.max()))
        except Exception as e:
            print('error reading sample mat:', e)
else:
    print('validation folder missing')
