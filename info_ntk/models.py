from neural_tangents import stax

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False, W_std=1, b_std=0.):
  Main = stax.serial(
      stax.Relu(), stax.Conv(channels, (3, 3),  W_std=W_std,b_std=b_std, strides = strides, padding='SAME'),
      stax.Relu(), stax.Conv(channels, (3, 3),  W_std=W_std,b_std=b_std, padding='SAME'))
  Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
      channels, (3, 3), strides, W_std=W_std,b_std=b_std,  padding='SAME')
  return stax.serial(stax.FanOut(2),
                     stax.parallel(Main, Shortcut),
                     stax.FanInSum())

def WideResnetGroup(n, channels, strides=(1, 1), W_std=1., b_std=0.):
  blocks = []
  blocks += [WideResnetBlock(channels, strides, channel_mismatch=True, W_std=W_std, b_std=b_std)]
  for _ in range(n - 1):
    blocks += [WideResnetBlock(channels, (1, 1), W_std=W_std, b_std=b_std)]
  return stax.serial(*blocks)

def WideResnet(block_size, k, num_classes, W_std = 1., b_std = 0.):
  return stax.serial(
      stax.Conv(16, (3, 3),  W_std=W_std,b_std=b_std, padding='SAME'),
      WideResnetGroup(block_size, int(16 * k), W_std=W_std, b_std=b_std),
      WideResnetGroup(block_size, int(32 * k), (2, 2), W_std=W_std, b_std=b_std),
      WideResnetGroup(block_size, int(64 * k), (2, 2), W_std=W_std, b_std=b_std),
      stax.AvgPool((7, 7)),
      stax.Flatten(),
      stax.Dense(num_classes, W_std=W_std, b_std = b_std))

