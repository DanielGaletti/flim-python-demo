"""
U-NetFLIM: FLIM frozen encoder + U-Net decoder trainável com skip connections.
Encoder FLIM capturado via forward hooks (evita incompatibilidades de grad).
"""
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=1, p=0):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False),   # inplace=False para segurança com autograd
        )


class UNetFLIM(nn.Module):
    """
    FLIM encoder (frozen) + U-Net decoder (trainável).
    Features do encoder capturadas via hooks — sem manipulação manual do grafo.
    """

    def __init__(self, flim_model, channel_list, n_input_channels: int = 3):
        super().__init__()
        self.n_blocks = len(channel_list)
        self.n_input_channels = n_input_channels

        # Guarda referência (NÃO registra como submodule — evita duplicar params)
        object.__setattr__(self, '_flim', flim_model)

        # Congela encoder
        for p in flim_model.parameters():
            p.requires_grad = False

        # Projeções de skip: converte canais FLIM → 32 (trainável, cria grad_fn)
        self.skip_projs = nn.ModuleList([
            ConvBnRelu(c, 32) for c in channel_list
        ])

        # Bottleneck: projeta feature mais profunda (initial d)
        self.bottleneck = ConvBnRelu(channel_list[-1], 32)

        # Decoder blocks: recebe (32 de cima + 32 do skip) → 32
        self.dec_blocks = nn.ModuleList([
            ConvBnRelu(64, 32, k=3, p=1) for _ in range(self.n_blocks - 1)
        ])

        self.head = nn.Conv2d(32, 1, 1)

    def _encode(self, x):
        """Executa encoder FLIM e captura features via hooks."""
        flim = object.__getattribute__(self, '_flim')
        features = []
        hooks = []

        # Registra hook em cada bloco para capturar saída após pooling
        def make_hook(lst):
            def hook(module, inp, out):
                lst.append(out.detach())   # detach: sem grad pelo encoder
            return hook

        # Hook no pool de cada camada
        for layer in flim.layers:
            h = layer.pool.register_forward_hook(make_hook(features))
            hooks.append(h)

        # Detecta canais esperados pelo encoder via norm_parameters do primeiro layer
        try:
            expected_ch = len(flim.layers[0].normalization_parameters["mean"])
        except Exception:
            expected_ch = x.shape[1]
        if expected_ch == 1 and x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        with torch.no_grad():
            flim(x)   # forward completo (usa decoder interno do FLIM, ignoramos)

        for h in hooks:
            h.remove()

        # Garante que grad está ativo após sair do no_grad (pyflim pode deixar desabilitado)
        torch.set_grad_enabled(True)

        return features[:self.n_blocks]

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Encoder (frozen) — features sem grad, mas skip_projs criarão grad
        enc_feats = self._encode(x)

        # Bottleneck: deepest feature → 32ch (trainável → cria grad_fn)
        d = self.bottleneck(enc_feats[-1])

        # Decoder U-Net: do mais profundo ao mais raso
        # skip_projs[i] corresponde a enc_feats[i] (índice crescente = mais raso)
        for i in range(self.n_blocks - 2, -1, -1):
            skip = self.skip_projs[i](enc_feats[i])   # trainável → grad ✓
            d = F.interpolate(d, size=skip.shape[2:], mode='bilinear', align_corners=False)
            d = self.dec_blocks[self.n_blocks - 2 - i](torch.cat([d, skip], dim=1))

        return F.interpolate(self.head(d), (H, W), mode='bilinear', align_corners=False)


def build_unet_flim(arch_file, flim_model):
    """Constrói U-NetFLIM a partir de arch_file + modelo FLIM treinado.
    Detecta canais reais via probe forward pass — k-means pode produzir
    número diferente de kernels do que noutput_channels no arch."""
    import json as _json
    with open(arch_file) as f:
        cfg = _json.load(f)
    n_layers = cfg["nlayers"]
    n_input_cfg = cfg.get("layer1", {}).get("conv", {}).get("nInput_channels", 3)

    # Probe: forward pass com imagem dummy para capturar shapes reais
    try:
        expected_ch = len(flim_model.layers[0].normalization_parameters["mean"])
    except Exception:
        expected_ch = n_input_cfg

    probe = torch.zeros(1, expected_ch, 256, 256)
    feats = []
    hooks = []

    def _hook(lst):
        def h(m, inp, out):
            lst.append(out.shape[1])   # número de canais
        return h

    for layer in flim_model.layers:
        hooks.append(layer.pool.register_forward_hook(_hook(feats)))

    with torch.no_grad():
        flim_model(probe)

    for h in hooks:
        h.remove()
    torch.set_grad_enabled(True)

    channels = feats[:n_layers]
    # Fallback por layer caso o probe não retorne todos
    for i in range(len(channels), n_layers):
        channels.append(cfg[f"layer{i+1}"]["conv"]["noutput_channels"])

    n_input = expected_ch
    return UNetFLIM(flim_model, channels, n_input_channels=n_input)
