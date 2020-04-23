import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc_pose.abcnet.spectral_norm import SpectralNorm
from edflow.custom_logging import get_logger
from torch.nn import Conv2d
from torch.nn.utils import weight_norm

_valid_norms = [None, "bn", "in", "none"]
_valid_pads = ["zeros", "reflect"]


# TODO: Use interpolate
# TODO: Add conv and conv_transposed
class Resampler(nn.Module):
    """Doubles or halves the spacial extent of a 4D tensor using nearest
    neighbour sampling."""

    def __init__(self, kind=None, size=None, conv_args={}):
        """Args:
            kind (str): Can be `up`, `down`, `conv`, `conv_transposed` or
                `None`. Determines the resampling method used:
                    - `up`: Nearest Neighbour upsampling.
                    - `down`: NN Downsampling.
                    - `conv`: Using a strided convolution with stride 2.
                    - `conv_transposed`: Using a transposed convolution.
                    - `None`: No resampling
            size (int or list): Overwrites the bahviour of :class:`Resampler`
                when :attr:`kind` is given to a scaling operation resulting
                in a tensor with spatial size :attr:`size`.A
            conv_args (dict): Arguments passed to the convolution or transposed
                convolution, should it be used. Should at least contain
                ``in_channels`` and ``out_channels``. By default it will
                contain ``kernel_size=1`` and ``stride=2``.
        """
        super().__init__()

        self.kind = kind
        self.size = size
        self.cargs = conv_args

        if size is None:
            if self.kind is None:
                self.sampler = lambda x: x
            elif self.kind in ["up", "down"]:
                self.sampler = F.interpolate
            elif self.kind == "conv":
                if "stride" not in conv_args:
                    conv_args["stride"] = 2
                if "kernel_size" not in conv_args:
                    conv_args["kernel_size"] = 1
                self.sampler = nn.Conv2d(**conv_args)
            elif self.kind == "conv_transposed":
                if "stride" not in conv_args:
                    conv_args["stride"] = 2
                if "kernel_size" not in conv_args:
                    conv_args["kernel_size"] = 1
                self.sampler = nn.ConvTranspose2d(**conv_args)
            else:
                raise NotImplementedError(
                    "kind must be one of `None`, `up`, "
                    "`down` `conv` or `conv_transposed` "
                    "but is "
                    "{}".format(self.kind)
                )
        else:
            self.sampler = F.interpolate

    def __str__(self):
        ca = ", ".join(["{}={}".format(k, v) for k, v in self.cargs.items()])
        if len(ca) > 0:
            ca = " " + ca
        return "Resampler(kind={}, size={}{})".format(self.kind, self.size, ca)

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        r"""Applies resampling to tensor x.

        Arguments:
            x (torch.Tensor): 4D Tensor of shape $[N, C, D_x, D_y]$, with batch
                size $N$, Number of channels $C$ and spacial extents $D_{x,y}$.
        Returns:
            torch.Tensor: resampled tensor.of shape $[N, C, f*D_x, f*D_y]$
                    with $f = \begin{cases} 2 \; \text{if kind = up}  \\
                    0.5 \; \text{if kind = down} \\
                    1 \; \text{if kind = `None`} \end{cases}$
        """
        if self.size is not None:
            return self.sampler(x, size=self.size, mode="nearest")
        else:
            if self.kind in ["up", "down"]:
                sf = 2 if self.kind == "up" else 0.5
                return self.sampler(x, scale_factor=sf, mode="nearest")
            elif self.kind is None:
                return self.sampler(x)
            else:
                return self.sampler(x)

    @property
    def scale(self):
        return 1 if self.kind is None else 0.5 if self.kind == "down" else 2


class NoiseInject(nn.Module):
    """Add random detail, as ins StyleGAN.

    .. codeblock:: python

        Input ----------------------------------> Addition -> Output
                                                  /
        Noise -> Weighting by learned parameters /

    """

    def __init__(self, n_channels):
        """Args:
            n_channels (int): The number of channels of the input features.
        """

        super().__init__()

        self.channel_weights = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        nn.init.kaiming_normal_(self.channel_weights, nonlinearity="linear")

    def forward(self, input_features, **noise_kwargs):
        """Args:
            input_features (torch.tensor): Some feature tensor of shape
                ``[B, C, H, W]``.
            noise_kwargs (kwargs): Arguemnts passed to the noise sampler
                :function:`torch.randn`.

        Returns:
            torch.tensor: The blurred version of :attr:`input_features` with
                shape ``[B, C, H, W]``.
        """

        # Noise with same dtype and device and shape as input_features.
        noise = torch.randn_like(input_features, **noise_kwargs)

        # Weighting along channel dimenion 1 -> this is learned
        weighted_noise = self.channel_weights * noise

        blurred_input = input_features + weighted_noise

        return blurred_input


class StyleAdaIn(nn.Module):
    r"""As in StyleGan: AdaIn Normalization from some style vector.

    .. codeblock:: python

        Stype_Input -> (FC -> Activation) * 3 -> Bias -------,
                                              `-> Scale       \
             ,----- Std --------------,                \       \
            /                          \                \       \
           /    -- Mean --,             \                \       \
          /    /           \             \                \       \
        Input_Features -> Substract -> Divide -----> Multiply -> Add -> Output
    """

    def __init__(self, style_size, feature_channels, activation=nn.ReLU, hidden=256):
        """Args:
            style_size (int): Number of elements in the style vector.
            feature_channels (int): Number of features in the input layer.
            activation (Callable): Activation function applied after all
                FC Layers.
            hidden (int): Number of hidden units of the fully connected net
                calculating the Bias and Scale of the AdaIn operation.
        """

        super().__init__()

        self.fc1_1 = nn.Linear(style_size, feature_channels)
        self.fc1_2 = nn.Linear(style_size, feature_channels)

        self.act = activation()

        self.logger = get_logger(self)

    def forward(self, style, features):
        """Args:
            style (torch.tensor): Style vector of shape ``[B, Z]``.
            features (torch.tensor): Feature Tensor to normalize of shape
                ``[B, C, H, W]``.

        Returns:
            torch.tensor: Adaptively instance normalized features of shape
                ``[B, C, H, W]``.
        """

        mean = features.mean(0).unsqueeze(0)
        std = features.std(0).unsqueeze(0)

        self.logger.debug("Style: {}".format(style.size()))
        scale = self.fc1_1(style)
        scale = self.act(scale).unsqueeze(2).unsqueeze(3)

        bias = self.fc1_2(style)
        bias = self.act(bias).unsqueeze(2).unsqueeze(3)

        self.logger.debug("scale: {}".format(scale.size()))
        self.logger.debug("bias: {}".format(bias.size()))
        self.logger.debug("mean: {}".format(mean.size()))
        self.logger.debug("std: {}".format(std.size()))
        self.logger.debug("features: {}".format(features.size()))

        return scale * (features - mean) / std + bias


class AdaNoise(torch.nn.Module):
    """Combines noise inject and Adaptive instance norm as one layer."""

    def __init__(self, style_size, n_feature_channels):
        super().__init__()

        self.ada_in = StyleAdaIn(style_size, n_feature_channels)
        self.noise_inj = NoiseInject(n_feature_channels)

    def forward(self, features, style):
        """Args:
            features (torch.tensor): Shape ``[B, C, H, W]``.
            style (torch.tensor): Shape ``[B, Z]``

        Returns:
            torch.tensor: Blurred and normalized features of shape
                ``[B, C, H, W]``.
        """

        blurred = self.noise_inj(features)
        normalized = self.ada_in(style, blurred)

        return normalized


class ResidualBlock(nn.Module):
    r"""Implementation as the one from Patrick. Skip connection is later.

    Applies the following operations to an input tensor:

    .. codeblock:: python

                    [Ingest Tensor]
                                  \
        Conv -> [Resampling ->] [Concat ->] Norm. -> Conv -> Norm. -> Addition
                               \                                     /
                                -------------------------------------
                                              Residual

    ``Ingest Tensor`` is an optional tensor, which is concatenated to the
    processed input tensor after resampling.
    ``Residual`` is the resampled input tensor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resample=None,
        latent_dim=None,
        normalization=nn.BatchNorm2d,
        f_size=3,
        padding=1,
        activation=torch.nn.ReLU(),
        ingest_channels=0,
        stride=1,
        sn=False,
    ):
        r"""
        Args:
            in_channels (int) : Number of channels of the input tensor.
            out_channels (int): Number of channels of the output tensor.
            resample (str): One of ``up``, ``down`` of ``None``.
                Resampling method applied.
                See :class:`.Resampler`. Default = `None`.
            latent_dim (int): Number of entries in the latent encodings. Only
                used if normalization is :class:`AdaNoise`.
            f_size (int): Kernel size of the convolutions. Default = 3.
            padding (int): Padding of the convolutions. Default = 1.
            activation (Callable): Callable activation function. Ignored if
                None. Default = None.
            ingest_channels (int): Number of channels of the ingested tensor.
            normalization (Callable): Callable normalization function. Ignored
                if None. Default ``= torch.nn.BatchNorm2d``.
            stride (int): Stride of the convolutions. Default = 1.
            sn (bool): add spectral norm to the conv layers.
        """
        super(ResidualBlock, self).__init__()

        assert resample in [None, "up", "down"]
        self.resample = resample

        self.conv1 = nn.Conv2d(in_channels, out_channels, f_size, stride, padding)
        if normalization.__name__ != AdaNoise.__name__:
            self.bn1 = normalization(out_channels + ingest_channels)
        else:
            self.bn1 = normalization(latent_dim, out_channels + ingest_channels)
        self.conv2 = nn.Conv2d(
            out_channels + ingest_channels, out_channels, f_size, stride, padding
        )
        if normalization.__name__ != AdaNoise.__name__:
            self.bn2 = normalization(out_channels)
        else:
            self.bn2 = normalization(latent_dim, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, f_size, stride, padding)

        if sn:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)
            self.conv3 = SpectralNorm(self.conv3)

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout2d(p=0.0)
        self.resample = Resampler(resample)

        self.out_channels = out_channels
        self.feature_channels = [out_channels + ingest_channels, out_channels]

    def forward(self, x, ingest=None, style=None):
        r"""
        Args:
            x (torch.Tensor): 4D Tensor of shape :math:`[N, C, D_x, D_y]`,
                with batch size :math:`N`,
                Number of channels :math:`C` and spatial extents
                :math:`D_{x,y}`.
            ingest (torch.Tensor): 4D Tensor of similar shape as x.
                Spatial dimension is determined by the resampling applied to x.

        Return:
            torch.Tensor: 4D Tensor of shape
                :math:`[N, \text{out_channels}, D_x^\prime, D_y^\prime]`
                with :math:`D_i^\prime = \begin{cases}
                2D_i &\text{if resample = up} \\
                \left\lfloor 0.5 \cdot D_i\right\rfloor &\text{if resample =
                down} \\
                D_i &\text{if resample = None} \end{cases}
                \forall i \in {x, y}`.

        Attributes:
            features (list of torch.Tensor): Activations from both intra
                blocks.
        """
        self.features = []

        res = out = self.conv1(x)
        res = out = self.resample(out)

        if ingest is not None:
            out = torch.add(out, ingest)

        if isinstance(self.bn1, AdaNoise):
            out = self.bn1(out, style)
        else:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        self.features += [out]

        out = self.conv2(out)
        if isinstance(self.bn2, AdaNoise):
            out = self.bn2(out, style)
        else:
            out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv3(out)

        out += res

        self.features += [out]

        return out

    def out_shape(self, input_size):
        """Calculates the shape of the output tensor."""

        if isinstance(input_size, int):
            input_size = [input_size] * 2

        scale = self.resample.scale
        out_size = [self.out_channels] + [int(i * scale) for i in input_size]

        return out_size

    @property
    def scale(self):
        """The scale factor, which describes the relation between the size
        of the input to the size of the output of the Block."""

        return self.resample.scale


class ResNet(nn.Module):
    """Concatenates a number of residual blocks (see `ResidualBlock`)."""

    def __init__(self, config, sn=False, **grow_kwargs):
        r"""
        Args:
            config (list): List of Lists with entries passed as arguments to
                the created :class:`ResidualBlock` instances.
            sn (bool): Add SpectralNorm to all layers.
        """
        super(ResNet, self).__init__()

        self.logger = get_logger(self)

        self.blocks = []
        self.feature_channels = []
        for i, args in enumerate(config):
            name = "block_{}".format(i)
            self.add_module(name, ResidualBlock(*args, sn=sn))
            self.blocks += [getattr(self, name)]
            self.feature_channels += getattr(self, name).feature_channels

    def forward(self, x, ingests=None, layers=None, style=None):
        r"""
        Args:
            x (torch.Tensor): 4D Tensor of shape $[N, C, D_x, D_y]$, with batch
                size $N$,
                Number of channels $C$ and spacial extents $D_{x,y}$.
            ingests (list): List of 4D tensors (or None) ingested to each
                residual block. See :class:`ResidualBlock`.
            layers (list): List of ints defining, which blocks to use for
                computation.
            style (torch.tensor): Style representations used if :class:`AdaIn`
                or :class:`AdaNoise` Normalizations are used.

        Returns:
            torch.Tensor: 4D Tensor with shape $[N, C^\prime, D_x^\prime,
                D_y^\prime]$, with $C^\prime$ being determined by the last
                residual block and the spacial dimension $D_{x,y}$ being
                determined by all applied resampling operations.

        Attributes:
            features (list of torch.Tensor): All features from all
                :class:`ResidualBlock`s.
        """
        # Run all blocks and collect features
        self.features = []

        # Expand ingests
        if ingests is not None:
            while len(ingests) < len(self.blocks):
                ingests.append(None)
            assert len(ingests) == len(self.blocks)
        else:
            ingests = [None] * len(self.blocks)

        # Filter blocks
        iterator = zip(self.blocks, ingests)
        if layers is not None:
            iterator = list(iterator)
            _iterator = []
            for l in layers:
                _iterator += [iterator[l]]
            iterator = _iterator

        for block, ingest in iterator:
            x = block(x, ingest=ingest, style=style)
            self.features += block.features

        return x

    def __len__(self):
        return len(self.blocks)

    def out_shape(self, input_size):
        """Given the spatial size of the input to the :class:`ResNet`, this
        computes the shape of the tensor produced from the input.

        Args:
            input_size (int or list): x and y size of the input tensor.

        Returns:
            int: Number of elements in the output tensor.
        """

        out_size = input_size
        for block in self.blocks:
            out_size = block.out_shape(out_size)[1:]

        return [self.blocks[-1].out_channels] + out_size


class BaseEncoder(nn.Module):
    r"""Maps an input image to a latent vector representation using the
    following operations:

    .. codeblock:: python

    -> Conv -> ResNet -> FC ->


    """

    def __init__(
        self, c_in, res_conf, activation, z_dim, normalization, in_size, variational, **kwargs
    ):
        r"""
        Arguments:
            c_in (int): Number of input channels.
            res_conf (list): Config List of the `ResNet` part of the Encoder
            activation (Callable): Activation function.
            z_dim (int): Size of the latent representation
            normalization (Callable): Normalization function.
            in_size (int or list(int)): Spatial size of the input image
            variational (bool): If True samples around the output of the
                encoder.
        """
        super(BaseEncoder, self).__init__()

        self.logger = get_logger(self)

        nf = res_conf[0][0]

        self.conv = nn.Conv2d(c_in, nf, 3, padding=1)

        self.resNet = ResNet(res_conf, mode="encoding", ref_size=in_size)
        self.activation = activation

        self.logger.debug("res_out: {}".format(self.resNet.out_shape(in_size)))
        self.f_in = np.prod(self.resNet.out_shape(in_size))
        self.logger.debug("f_in: {}".format(self.f_in))
        self.fc1 = nn.Linear(self.f_in, z_dim)

        if normalization is None:
            self.bn = lambda x: x
        else:
            self.bn = normalization(z_dim)

        self.is_variational = variational
        if self.is_variational:
            self.normal = torch.distributions.Normal(0, 1)

    def forward(self, x, skip=False, depth=None, alpha=None):
        r"""
        Args:
            x (torch.Tensor): 4D Tensor of shape $[N, C, D_x, D_y]$, with batch
                size $N$, Number of channels $C$ and spacial extents $D_{x,y}$.
            skip (boolean): If True, we need to output the features of the ResNet to use them in the decoder.
                            Long skip connection.
            depth (int): When training the models using the growing method of
                Karras et al. specifies the number of :class:`ResNet` blocks
                to use during generation and discrimination.
                If None the models is completely unrolled.
            alpha (float): Value between 0 and 1 used to blend in layers
                during during blending. Ignored if None or depth is None.

        Returns:
            torch.Tensor: 2D Tensor with shape $[N, Z]$
        """
        out = self.conv(x)
        out = self.activation(out)

        layers = None
        if depth is not None:
            layers = list(range(len(self.resNet)))
            if alpha is not None and alpha < 1:
                layers = layers[: depth + 1]
            else:
                layers = layers[:depth]

        out = self.resNet(out, layers=layers)
        out = self.activation(out)

        self.logger.debug("out: {}".format(out.size()))
        out = out.view(-1, self.f_in)
        self.logger.debug("out: {}".format(out.size()))
        out = self.fc1(out)
        self.logger.debug("out: {}".format(out.size()))

        out = self.bn(out)
        self.logger.debug("out: {}".format(out.size()))

        if self.is_variational:
            out += torch.randn_like(out)
            self.logger.debug("out: {}".format(out.size()))

        if skip:
            return out, self.resNet.features
        return out


class Decoder(nn.Module):
    r"""Generates an image from two latent encodings, one encoding the pose,
    one the appearance.

    .. codeblock:: python

    z_app    \
    z_pose    }-> ResNet -> BN -> Activation -> Image
    z_motion /

    """

    def __init__(self, z_dim, z_spatial, res_conf, activation, normalization, c_out, **kwargs):
        """
        Args:
            res_conf (list): :class:`ResNet` config/arguments passed to
                its constructor.
            z_dim (int): Number of elements in the latent encodings.
            z_spatial (int or list(int)): Initial spatial tensor fed to the
                ResNet block.
            activation (Callable): Activation function.
            normalization (Callable): Normalization function.
            c_out (int): Number of channels in the output image.
        """
        super(Decoder, self).__init__()

        self.logger = get_logger(self)

        # Prepare inputs for resnet part
        if isinstance(z_spatial, int):
            z_spatial = [z_spatial] * 2

        self.pre_resNet_shape = z_spatial
        self.logger.debug("pre_resnet: {}".format(self.pre_resNet_shape))
        self.logger.debug("spatial: {}".format(z_spatial))

        n_ch = np.prod(self.pre_resNet_shape)
        self.conv0 = nn.Conv2d(z_dim * 2, n_ch, 1)

        # Resnet part
        self.resNet = ResNet(res_conf, mode="generative", ref_size=z_spatial[1:])
        res_out_shape = self.resNet.out_shape(z_spatial[1:])
        self.logger.debug("res out: {}".format(res_out_shape))

        # Make it an image!
        self.bn = normalization(res_out_shape[0])
        self.activation = activation
        self.conv1 = nn.Conv2d(res_out_shape[0], c_out, 1)
        self.image_act = nn.Sigmoid()

    def forward(self, z_app, z_pose, skips=None, depth=None, alpha=None):
        """
        Args:
            z_app (torch.Tensor): Appearance encoding
            z_pose (torch.Tensor): Pose encoding
            z_motion (torch.Tensor): Motion encoding
            depth (int): When training the models using the growing method of
                Karras et al. specifies the number of :class:`ResNet` blocks
                to use during generation and discrimination.
                If None the models is completely unrolled.
            alpha (float): Value between 0 and 1 used to blend in layers
                during during blending. Ignored if None or depth is None.
        """

        # Concatenate the encodings, then convert to a 3D tensor with
        # shape [Z, 1, 1]
        z_concat = torch.cat([z_app, z_pose], dim=1)  # [Z*3]

        z_spatial = z_concat.unsqueeze(-1).unsqueeze(-1)  # [Z*3, 1, 1]

        z_spatial = self.conv0(z_spatial)  # [256*4*4, 1, 1]
        z_spatial = z_spatial.view(-1, *self.pre_resNet_shape)  # [256, 4, 4]

        layers = None
        if depth is not None:
            layers = list(range(len(self.resNet)))
            if alpha is not None and alpha < 1:
                layers = layers[depth + 1 :]
            else:
                layers = layers[depth:]

        pre_image = self.resNet(z_spatial, ingests=skips, layers=layers, style=z_app)

        self.logger.debug("pre im {}".format(pre_image.size()))
        pre_image = self.bn(pre_image)
        pre_image = self.activation(pre_image)
        self.logger.debug("pre im {}".format(pre_image.size()))

        image = self.conv1(pre_image)
        # image = self.image_act(image)
        self.logger.debug("im {}".format(image.size()))

        return image


class EncBasic(nn.Module):
    """
    Basic encoder, maps any input of size (*, nc, ipt_size, ipt_size) to an embedding of size (*, latent_dim, 1, 1)
    """

    def __init__(
        self,
        ipt_size=64,
        complexity=64,
        nc=3,
        latent_dim=256,
        activation="tanh",
        norm="bn",
        use_bias=False,
        normalize_encodings=False,
        use_coord_conv=False,
        vae=False,
    ):

        super(EncBasic, self).__init__()

        assert norm in _valid_norms, "The entered norm {} is not valid, need one of {}".format(
            norm, _valid_norms
        )

        if norm is not None:
            if norm == "bn":
                norm = nn.BatchNorm2d
                use_bias = False
            elif norm == "in":  # Instance Norm
                norm = nn.InstanceNorm2d
                use_bias = True
            else:
                norm = None

        self.vae_mode = vae
        self.activation = activation
        self.normalize_encodings = normalize_encodings
        if normalize_encodings:
            self.activation = None

        self.n_blocks = int(np.log2(ipt_size) - 1)

        self.main = nn.Sequential()
        # BLOCK 0
        if not use_coord_conv:
            self.main.add_module(
                "b00", Conv2d(nc, complexity, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )
            if norm is not None:
                self.main.add_module("b01", norm(complexity))
            self.main.add_module("b02", nn.LeakyReLU(0.2, inplace=True))
        else:
            self.main.add_module("b00", CoordConv(nc, nc))
            self.main.add_module(
                "b01", Conv2d(nc, complexity, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )
            if norm is not None:
                self.main.add_module("b02", norm(complexity))
            self.main.add_module("b03", nn.LeakyReLU(0.2, inplace=True))

        # BLOCKS 1 - N-1
        for b in range(1, self.n_blocks - 1):
            c_in = complexity * 2 ** (b - 1)
            c_out = complexity * 2 ** b
            n = "b" + str(b)
            self.main.add_module(
                n + "0", Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )
            if norm is not None:
                self.main.add_module(n + "1", norm(c_out))
            self.main.add_module(n + "2", nn.LeakyReLU(0.2, inplace=True))

        if not vae:
            # BLOCK N: 4 --> 1
            n = "b" + str(self.n_blocks - 1) + "0"
            self.main.add_module(
                n, Conv2d(c_out, latent_dim, kernel_size=4, stride=1, padding=0, bias=True)
            )
        else:
            self.mu_layer = Conv2d(c_out, latent_dim, kernel_size=4, stride=1, padding=0, bias=True)
            self.var_layer = Conv2d(
                c_out, latent_dim, kernel_size=4, stride=1, padding=0, bias=True
            )

    def forward(self, x, skip=False):
        if self.vae_mode:
            x = self.main(x).squeeze()
            mu = self.mu_layer(x)
            logvar = self.var_layer(x)
            x = self.reparameterize(mu, logvar)
            return x, mu, logvar
        else:
            x = self.main(x)
            if self.normalize_encodings:
                x = x / torch.norm(x.squeeze(), p=2, dim=1)
                return x
            x = torch.tanh(x)
            return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def extract_features(self, x, layer_idx):
        idx = 0
        for layer in self.main:
            x = layer(x)
            if idx == layer_idx:
                return x
            idx += 1

    def extract_multiple_features(self, x):
        features = list()
        for ix in self.skip_ids:
            features.append(self.extract_features(x, ix))
        return features


class GenBasic(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        ipt_size=64,
        complexity=64,
        nc_out=3,
        norm="in",
        upsample_layer=True,
        use_bias=True,
        use_coord_conv=False,
        skips=None,
    ):

        super(GenBasic, self).__init__()

        assert norm in _valid_norms, "The entered norm {} is not valid, need one of {}".format(
            norm, _valid_norms
        )

        if norm is not None:
            if norm == "bn":
                norm = nn.BatchNorm2d
                use_bias = False
            elif norm == "in":  # Instance Norm
                norm = nn.InstanceNorm2d
                use_bias = True
            else:
                norm = None

        from torch.nn import ConvTranspose2d as CT

        self.n_blocks = int(np.log2(ipt_size) - 1)
        self.main = nn.Sequential()
        self.latent_size = latent_dim
        # BLOCK 0
        # bs x lat x 1 x 1 --> bs x cout x 4 x 4
        c_out = complexity * 2 ** (self.n_blocks - 2)
        self.main.add_module("b00", CT(latent_dim, c_out, 4, 1, 0, bias=use_bias))
        if norm is not None and norm != "adain":
            self.main.add_module("b01", norm(c_out))
        self.main.add_module("b02", nn.LeakyReLU(0.15, inplace=True))

        # as karras seems to use adain after activation, let's do this here too
        if norm == "adain":
            self.main.add_module("b01_post", AdaIn(latent_dim, c_out))

        kernel_size = 4
        stride = 2
        if upsample_layer:
            CT = UpsampleConvLayer
            kernel_size = 3
            stride = 1

        # BLOCKS 1 - N-1
        for i, b in enumerate(reversed(range(1, self.n_blocks - 1))):
            c_in = complexity * 2 ** (b)
            c_out = complexity * 2 ** (b - 1)
            n = "b" + str(b)
            self.main.add_module(
                n + "0",
                CT(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=1, bias=use_bias),
            )
            if norm is not None and norm != "adain":
                self.main.add_module(n + "1", norm(c_out))
            self.main.add_module(n + "2", nn.LeakyReLU(0.15, inplace=True))

            if norm == "adain":
                self.main.add_module(b + "_post", AdaIn(latent_dim, c_out))

        # BLOCK N: 4 --> 1
        n = "b" + str(self.n_blocks - 1)
        s0, s1 = "0", "1"
        if use_coord_conv:
            self.main.add_module(n + "0", CoordConv(complexity, complexity))
            s0, s1 = "1", "2"
        self.main.add_module(
            n + s0,
            CT(
                complexity, nc_out, kernel_size=kernel_size, stride=stride, padding=1, bias=use_bias
            ),
        )
        self.main.add_module(n + s1, nn.Tanh())

    def forward(self, z_app, z_beta, skips=None):
        z = torch.cat((z_app, z_beta), dim=1)
        z = self.main(z)
        return z


class GenBasicAdaIn(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        ipt_size=64,
        complexity=64,
        nc_out=3,
        upsample_layer=False,
        use_bias=True,
        cpa=False,
    ):

        super().__init__()
        from torch.nn import ConvTranspose2d as CT

        self.n_blocks = int(np.log2(ipt_size) - 1)
        self.cts = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.acts = nn.ModuleDict()
        self.use_pose_and_app = cpa
        if self.use_pose_and_app:
            adain_dim = latent_dim
        else:
            adain_dim = latent_dim // 2

        self.latent_size = latent_dim
        # BLOCK 0
        # bs x lat x 1 x 1 --> bs x cout x 4 x 4
        c_out = complexity * 2 ** (self.n_blocks - 2)

        self.cts.update({"c0": CT(latent_dim, c_out, 4, 1, 0, bias=use_bias)})
        self.acts.update({"a0": nn.LeakyReLU(0.15)})
        self.norms.update({"n0": AdaIn(nf_latent=adain_dim, nfn=c_out)})

        kernel_size = 4
        stride = 2
        if upsample_layer:
            CT = UpsampleConvLayer
            kernel_size = 3
            stride = 1

        # BLOCKS 1 - N-1
        for i, b in enumerate(reversed(range(1, self.n_blocks - 1))):
            c_in = complexity * 2 ** (b)
            c_out = complexity * 2 ** (b - 1)

            self.cts.update(
                {
                    f"c{b}": CT(
                        c_in,
                        c_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1,
                        bias=use_bias,
                    )
                }
            )
            self.acts.update({f"a{b}": nn.LeakyReLU(0.15)})
            # only use half of the latents as only the pose information is used to condition the output
            self.norms.update({f"n{b}": AdaIn(adain_dim, c_out)})

        self.cts.update(
            {
                "c_out": CT(
                    complexity,
                    nc_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=use_bias,
                )
            }
        )
        self.acts.update({f"a_out": nn.Tanh()})

    def forward(self, z_app, z_beta):
        z = torch.cat((z_app, z_beta), dim=1)

        # only use shape and pose information if enabled by user
        latents = torch.cat((z_app, z_beta), dim=1) if self.use_pose_and_app else z_beta
        latents = latents.permute([0, 2, 3, 1])

        # first layer
        out = self.cts["c0"](z)
        out = self.acts["a0"](out)
        out = self.norms["n0"](out, latents)

        for i in reversed(range(1, self.n_blocks - 1)):

            out = self.cts[f"c{i}"](out)
            out = self.acts[f"a{i}"](out)
            out = self.norms[f"n{i}"](out, latents)

        out = self.cts["c_out"](out)
        out = self.acts["a_out"](out)
        return out


class Interpolater(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Interpolater, self).__init__()
        self.input_layer = nn.Linear(z_dim * 3, hidden_dim)
        self.activation_layer = nn.LeakyReLU()
        ###
        self.hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        ###
        self.output_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_previous, z_next, t):
        x = torch.cat([z_previous, z_next, t], dim=1)
        x = self.input_layer(x)
        x = self.activation_layer(x)
        ###
        x = self.hidden_1(x)
        x = self.activation_layer(x)
        ###
        x = self.output_layer(x)
        return x


class ReconstructionDiscriminator(nn.Module):
    """Classifies images as real or fake."""

    def __init__(self, c_in, res_conf, in_size, activation=torch.nn.ReLU(), **kwargs):
        """
        Args:
            in_channels (int) : Number of channels of the input tensor.
            out_channels (int): Number of channels of the output tensor.
            res_conf (list): List of arguments passed to the :class:`ResNet`
                part of the Encoder.
            activation (Callable): Callable activation function. Ignored if
                ``None`` (Default).
        """
        super().__init__()

        self.logger = get_logger(self)

        nf = res_conf[0][0]
        self.conv1 = SpectralNorm(nn.Conv2d(c_in, nf, 3, padding=1))

        self.resNet = ResNet(res_conf, sn=True, mode="encoding", ref_size=in_size)

        # self.resNet = ResNet(res_conf, sn=True)
        self.activation = activation
        self.f_in = f_in = np.prod(self.resNet.out_shape(in_size))
        self.fc1 = SpectralNorm(nn.Linear(f_in, 1))
        self.out_act = torch.nn.Sigmoid()

        self.features = []
        self.feature_channels = self.resNet.feature_channels

    def forward(self, x, is_target=False, depth=None, alpha=None):
        """
        Args:
            x (torch.Tensor): Image to be classified as real or fake
            is_target (bool): Used to determine if the conv net features are
                detached from the graph

        Returns:
            torch.Tensor: Probability that images real of fake.
        """
        p_orig = self.conv1(x)

        p_orig = self.resNet(p_orig)
        p_orig = self.activation(p_orig)

        self.features = self.resNet.features
        if is_target:
            for i, f in enumerate(self.features):
                self.features[i] = f.detach()

        self.logger.debug("p_orig: {}".format(p_orig.size()))
        p_orig = p_orig.view(-1, self.f_in)  # sum(3).sum(2)
        self.logger.debug("p_orig: {}".format(p_orig.size()))
        p_orig = self.fc1(p_orig)
        self.logger.debug("p_orig: {}".format(p_orig.size()))
        p_orig = self.out_act(p_orig)
        self.logger.debug("p_orig: {}".format(p_orig.size()))

        return p_orig

    def increase_alpha(self, step=0.01):
        return self.resNet.step_alpha(step)


class EmbeddingDiscriminator(nn.Module):
    """Used to enforce the disentanglement between Pose and Appearance during
    training, by comparing pairs of pose and appearance encodings, which
    consist of either pose and appearance from the same person or pose and
    appearance each from a different person.
    """

    def __init__(self, embedding_size, n_embeddings, hidden, activation, **kwargs):
        """
        Args:
            embedding_size (int): Number of elementes in the embedding vector
            n_embeddings (int): Number of embeddings to compare
            hidden (int): Size of the hidden layers of the discriminator
            activation (Callable): activation function
        """
        super().__init__()

        self.activation = activation
        self.fc1 = SpectralNorm(nn.Linear((n_embeddings + 1) * embedding_size, hidden))
        self.fc2 = SpectralNorm(nn.Linear(hidden, hidden))
        self.fc3 = SpectralNorm(nn.Linear(hidden, 1))

    def forward(self, embeddings):
        """
        Args:
            embeddings (list(torch.Tensor)): Appearance and Pose encoding pair

        Returns:
            torch.Tensor: Probability, that the embedding come from different
                persons.
        """
        e_combo = torch.cat(embeddings, dim=1)
        prod = torch.prod(torch.stack(embeddings, dim=1), dim=1)
        e_combo = torch.cat([e_combo, prod], dim=-1)

        p_diff = self.fc1(e_combo)
        p_diff = self.activation(p_diff)

        p_diff = self.fc2(p_diff)
        p_diff = self.activation(p_diff)

        p_diff = self.fc3(p_diff)

        return p_diff


class FCNet(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_layers: int,
        hidden_dimensions: list,
        output_dimension: int,
        activation=nn.ReLU,
    ):
        """
        Create a fully connected network given the following hyperparameters. The input and output is supposed to have
        a one-dimensional shape.
        Parameters
        ----------
        input_dimension: Number of dimensions of the input features.
        hidden_layers: Number of hidden layers.
        hidden_dimensions: Number of dimensions each layer has.
        output_dimension: Number of dimensions the output has.
        """
        super(FCNet, self).__init__()
        self.hidden_layers = hidden_layers
        if len(hidden_dimensions) != hidden_layers and len(hidden_dimensions) != 1:
            raise ValueError(
                "Number of hidden layers does not match the number of hidden dimension values provided."
            )
        former_dim = input_dimension
        self.features = nn.Sequential()
        for i, hidden_dim in enumerate(hidden_dimensions):
            self.features.add_module("hidden_" + str(i), nn.Linear(former_dim, hidden_dim))
            self.features.add_module("activation_" + str(i), activation())
            former_dim = hidden_dim
        self.output = nn.Linear(former_dim, output_dimension)

    def forward(self, inpt):
        features = self.features(inpt)
        output = self.output(features)
        return output


class EmbeddingMemory(nn.Module):
    """Updates an embedding given the sequence of embeddings that came before.
    Also does the bookkeeping of hidden states etc.
    """

    def __init__(self, embedding_size):
        """
        Arguments:
            embedding_size: Number of elements in the embedding vector
        """
        super().__init__()

        self.tanh = nn.Tanh()
        self.rnn_cell = nn.LSTMCell(embedding_size, embedding_size)

    def forward(self, embedding, ts=0):
        """
        Arguemts:
            embedding: torch.Tensor of size embedding_size
            ts: current time step or index of sequence element. If ts=0 rnn
                    cell is initialized with the input, else input is updated
                    to incorporate information from previous time steps.
                    In any case a Tanh activation is applied to the embedding,
                    to match the output support of the lstm cell.
        Returns:
            updated_embedding: embedding also containing inforation from
                    possible previous time steps.
        """

        updated_embedding = self.tanh(embedding)
        if ts == 0:
            self.h = torch.zeros_like(embedding)
            self.c = torch.zeros_like(embedding)

        self.h, self.c = self.rnn_cell(updated_embedding, (self.h, self.c))

        return self.h


class CoordConv(nn.Conv2d):
    """Thin wrapper around the conv layer, which adds pixel coordinates as
    feature channels. Drop in for existing conv layers, without the need to
    update the number of :attr:`in_channels`.
    """

    def __init__(self, in_channels, *args, **kwargs):
        self._coords = None

        super().__init__(in_channels + 2, *args, **kwargs)

    def coords(self, input, dimx, dimy, bs):
        """Coordinate in the range [-1, 1]"""
        if self._coords is None:
            I = torch.linspace(-1, 1, dimx)
            J = torch.linspace(-1, 1, dimx)
            i, j = torch.meshgrid(I, J)
            i = i.to(input)
            j = j.to(input)

            i = i[None, None, :, :].repeat_interleave(bs, dim=0)
            j = j[None, None, :, :].repeat_interleave(bs, dim=0)

            self._coords = [i, j]
        return self._coords

    def forward(self, input, *args, **kwargs):

        bs, ch, dimx, dimy = input.size()
        i, j = self.coords(input, dimx, dimy, bs)
        input = torch.cat([input, i, j], dim=-3)

        return super().forward(input)


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
        --> adapted from
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class AddCoords(nn.Module):
    """
    CoordConv by Uber, see https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    and the original paper https://arxiv.org/pdf/1807.03247.pdf
    An alternative implementation for PyTorch with auto-infering the x-y dimensions.
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)],
            dim=1,
        )

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size=kernel_size, **kwargs)
        self.weight = self.conv.weight

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/

    Note: Adapted a little bit to adjust to dual use here.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0.0,
        upsample=2.0,
        bias=True,
        upsample_mode="nearest",
    ):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.weight = self.conv2d.weight  # added to make weight init possible
        self.up_mode = upsample_mode

    def forward(self, x):
        x_in = x
        if self.upsample is not None:
            x_in = torch.nn.functional.interpolate(
                x_in, mode=self.up_mode, scale_factor=self.upsample
            )
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


def get_n_layers(config):
    img_size = config.img_size
    pw = config.patch_width
    if img_size not in [128, 256]:
        if config.dataset_name == "mnist":
            return 2  # half width
        else:
            raise NotImplementedError("Images of size {} not yet supported".format(img_size))
    if pw == "half":
        n_layers_patch_dis = 4 if img_size == 256 else 3
    elif pw == "fourth":
        n_layers_patch_dis = 3 if img_size == 256 else 2
    elif pw == "eighth":
        n_layers_patch_dis = 2 if img_size == 256 else 1
    else:
        raise NotImplementedError("patch_width {} not (yet) supported.".format(pw))
    return n_layers_patch_dis


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x


class IDAct(nn.Module):
    def forward(self, input):
        return input


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # required for the beta and gamma params initializers to be placed on gpu
        self.register_buffer("beta_init", torch.zeros([1, out_channels, 1, 1], dtype=torch.float32))
        self.register_buffer("gamma_init", torch.ones([1, out_channels, 1, 1], dtype=torch.float32))
        self.beta = nn.Parameter(self.beta_init)
        self.gamma = nn.Parameter(self.gamma_init)
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), name="weight"
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = conv_layer(channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.op2 = IDAct()

    def forward(self, x):
        out = self.up(x)
        out = self.op2(out)
        return out


class VUnetResnetBlock(nn.Module):
    """
    Resnet Block as utilized in the vunet publication
    """

    def __init__(
        self,
        out_channels,
        use_skip=False,
        kernel_size=3,
        activate=True,
        conv_layer=NormConv2d,
        dropout_prob=0.0,
        final_act=False,
    ):
        """

        :param n_channels: The number of output filters
        :param process_skip: the factor between output and input nr of filters
        :param kernel_size:
        :param activate:
        """
        super().__init__()
        self.dout = nn.Dropout(p=dropout_prob)
        self.use_skip = use_skip
        if self.use_skip:
            self.conv2d = conv_layer(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.pre = conv_layer(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1
            )
        else:
            self.conv2d = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        if activate:
            self.act_fn = nn.LeakyReLU() if final_act else nn.ELU()
        else:
            self.act_fn = IDAct()

    def forward(self, x, a=None):
        x_prc = x

        if self.use_skip:
            assert a is not None
            a = self.act_fn(a)

            a = self.pre(a)
            x_prc = torch.cat([x_prc, a], dim=1)

        x_prc = self.act_fn(x_prc)
        x_prc = self.dout(x_prc)
        x_prc = self.conv2d(x_prc)

        return x + x_prc


class AdaIn(nn.Module):
    def __init__(self, nf_latent, nfn, norm=nn.InstanceNorm2d):
        super().__init__()
        self.norm = norm(nfn)
        self.nfn = nfn
        self.linear = nn.Linear(in_features=nf_latent, out_features=2 * self.nfn)

    def forward(self, x, latents):

        out = self.norm(x)

        # mapping from latents to scale and shift components
        y = self.linear(latents)

        # splt y in y_scale and y_shift
        y = y.reshape([-1, 2, self.nfn])
        scale = y[:, 0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        shift = y[:, 1].unsqueeze(dim=-1).unsqueeze(dim=-1)

        return torch.add(torch.mul(out, torch.add(scale, 1.0)), shift)
