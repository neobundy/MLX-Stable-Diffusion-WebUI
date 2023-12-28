
# torch way => mlx way
# x = np.numpy([1,])
# x1 = torch.tensor(x)
# x2 = mx.array(x)

torch.torch() => mx.array()
reshape() => x2.reshape()
randn() => mx.random.normal()
to_numpy() => np.array(x2.tolist())
unsqueeze(0) => mx.expand_dims(x2, axis=0)
transpose(0,1) => mx.transpose(x2, axes=(1,0))
concatenate() => mx.concatenate()
broadcast_to() => mx.broadcast_to()
arange() => mx.arange()
cumprod() => mx.cumprod()
numel() => mx.prod(mx.size(y))