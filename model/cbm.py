import json
import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import utils as data_utils
from glm_saga.elasticnet import glm_saga


class CBM_model(torch.nn.Module):
    def __init__(
        self,
        backbone_name,
        W_c,
        W_g,
        b_g,
        proj_mean,
        proj_std,
        device="cuda",
        use_clip_penultimate: bool = False,
    ):
        super().__init__()
        if "clip" in backbone_name:
            clip_backbone = BackboneCLIP(
                backbone_name,
                use_penultimate=use_clip_penultimate,
                device=device,
            )
            self.backbone = clip_backbone
            self.preprocess = clip_backbone.preprocess
        elif backbone_name == "resnet18_cub":
            model, preprocess = data_utils.get_target_model(backbone_name, device)
            self.preprocess = preprocess
            self.backbone = lambda x: model.features(x)
        elif "cub" in backbone_name:
            model, preprocess = data_utils.get_target_model(backbone_name, device)
            self.preprocess = preprocess
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            model, preprocess = data_utils.get_target_model(backbone_name, device)
            self.preprocess = preprocess
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

        self.proj_layer = torch.nn.Linear(
            in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False
        ).to(device)
        self.proj_layer.load_state_dict({"weight": W_c})

        self.proj_mean = proj_mean
        self.proj_std = proj_std

        self.final = torch.nn.Linear(
            in_features=W_g.shape[1], out_features=W_g.shape[0]
        ).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})
        self.concepts = None

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x - self.proj_mean) / self.proj_std
        x = self.final(proj_c)
        return x, proj_c


class standard_model(torch.nn.Module):
    def __init__(
        self,
        backbone_name,
        W_g,
        b_g,
        proj_mean,
        proj_std,
        device="cuda",
        use_clip_penultimate: bool = False,
    ):
        super().__init__()
        if "clip" in backbone_name:
            clip_backbone = BackboneCLIP(
                backbone_name,
                use_penultimate=use_clip_penultimate,
                device=device,
            )
            self.backbone = clip_backbone
            self.preprocess = clip_backbone.preprocess
        elif backbone_name == "resnet18_cub":
            model, preprocess = data_utils.get_target_model(backbone_name, device)
            self.preprocess = preprocess
            self.backbone = lambda x: model.features(x)
        elif "cub" in backbone_name:
            model, preprocess = data_utils.get_target_model(backbone_name, device)
            self.preprocess = preprocess
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            model, preprocess = data_utils.get_target_model(backbone_name, device)
            self.preprocess = preprocess
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

        self.proj_mean = proj_mean
        self.proj_std = proj_std

        self.final = torch.nn.Linear(
            in_features=W_g.shape[1], out_features=W_g.shape[0]
        ).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})
        self.concepts = None

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        proj_c = (x - self.proj_mean) / self.proj_std
        x = self.final(proj_c)
        return x, proj_c


class Backbone(nn.Module):
    # store intermediate feature values from backbone
    feature_vals = {}

    def __init__(self, backbone_name: str, feature_layer: str, device: str = "cuda"):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_layer = feature_layer
        target_model, target_preprocess = data_utils.get_target_model(
            backbone_name, device
        )

        # hook into feature layer
        def hook(module, input, output):
            self.feature_vals[output.device] = output

        command = "target_model.{}.register_forward_hook(hook)".format(feature_layer)
        eval(command)

        # assign backbone and preprocess
        self.backbone = target_model
        self.preprocess = target_preprocess
        self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[backbone_name]

    def forward(self, x):
        out = self.backbone(x)
        return self.feature_vals[out.device].mean(dim=[2, 3])

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        model = cls.from_args(load_path, device)
        model.backbone.load_state_dict(
            torch.load(os.path.join(load_path, "backbone.pt"))
        )
        return model

    @classmethod
    def from_args(cls, load_dir: str, device: str = "cuda"):
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            args = json.load(f)
        return cls(args["backbone"], args["feature_layer"], device)


class BackboneCLIP(nn.Module):
    def __init__(
        self, backbone_name: str, use_penultimate: bool = True, device: str = "cuda"
    ):
        super().__init__()
        import clip

        target_model, target_preprocess = clip.load(backbone_name[5:], device=device)
        if use_penultimate:
            logger.info("Using penultimate layer of CLIP")
            target_model = target_model.visual
            N = target_model.attnpool.c_proj.in_features
            identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
            nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            target_model.attnpool.c_proj = identity
            self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{backbone_name}_penultimate"
            ]
        else:
            logger.info("Using final layer of CLIP")
            target_model = target_model.visual
            self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[backbone_name]

        # assign backbone and preprocess
        self.backbone = target_model.float()
        self.preprocess = target_preprocess

    def forward(self, x):
        output = self.backbone(x).float()
        return output

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))

    @classmethod
    def from_args(cls, load_dir: str, device: str = "cuda"):
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            args = json.load(f)
        return cls(args["backbone"], args["use_clip_penultimate"], device)

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        model = cls.from_args(load_path, device)
        model.backbone.load_state_dict(
            torch.load(os.path.join(load_path, "backbone.pt"))
        )
        return model


class ConceptLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        model = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(num_hidden):
            model.append(nn.ReLU())
            model.append(nn.Linear(out_features, out_features, bias=bias))

        self.model = nn.Sequential(*model).to(device)
        self.out_features = out_features
        logger.info(self.model)

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_dir):
        # save model
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        num_hidden = args["cbl_hidden_layers"]
        if args["use_clip_penultimate"] and args["backbone"].startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))

        # create model
        model = cls(encoder_dim, num_concepts, num_hidden=num_hidden, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "cbl.pt")))
        return model


class CosineSimilarityConceptLayer(nn.Module):
    """Concept bottleneck via cosine similarity between L2-normalized features and prototypes.

    Each concept c has a prototype p_c. The logit is:
        logit_c = exp(log_tau) * cos(x, p_c)
    where both x and p_c are L2-normalized before the dot product. A learned
    log-temperature is initialized to log(tau) so the logit scale is similar to
    the linear baseline.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 20.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(tau))))
        self.out_features = out_features
        self.to(device)
        logger.info(
            "CosineSimilarityConceptLayer: in={} out={} tau_init={:.2f}",
            in_features,
            out_features,
            tau,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x.float(), dim=-1)
        p_n = F.normalize(self.prototypes.float(), dim=-1)
        return self.log_tau.exp() * F.linear(x_n, p_n)

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        tau = float(args.get("cbl_tau", 20.0))
        model = cls(encoder_dim, num_concepts, tau=tau, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "cbl.pt")))
        return model


class LinearResidualRefinerCBL(nn.Module):
    """Linear CBL with a small gated residual refinement branch.

    Main path: standard linear projection (identical to linear baseline).
    Residual branch: in → ReLU(hidden) → out, initialized at zero.
    Gate: learned per-concept scalar sigmoid, biased toward closed (-4 → ≈ 0.018).

    At init: output ≈ linear(x) so the baseline is fully preserved.
    Training: gate and residual may open to correct hard concepts.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.res_down = nn.Linear(in_features, hidden_dim, bias=True)
        self.res_up = nn.Linear(hidden_dim, out_features, bias=False)
        self.gate_bias = nn.Parameter(torch.full((out_features,), -4.0))

        nn.init.normal_(self.res_down.weight, std=0.01)
        nn.init.zeros_(self.res_down.bias)
        nn.init.zeros_(self.res_up.weight)

        self.to(device)
        logger.info(
            "LinearResidualRefinerCBL: in={} out={} hidden_dim={}",
            in_features,
            out_features,
            hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        h = torch.relu(self.res_down(x))
        correction = self.res_up(h)
        gate = torch.sigmoid(self.gate_bias)
        return linear_out + gate * correction

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt")) as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        hidden_dim = int(args.get("cbl_residual_hidden_dim", 64))
        model = cls(encoder_dim, num_concepts, hidden_dim=hidden_dim, device=device)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "cbl.pt"), map_location=device)
        )
        return model


class InputGatedRefinerCBL(nn.Module):
    """Linear CBL with a residual branch gated by the input features.

    Main path: standard linear projection (identical to linear baseline).
    Residual branch: in → ReLU(hidden) → out, initialized at zero.
    Gate: per-input sigmoid from a linear gate network (weight=0, bias=-4 at init).

    At init: gate_net(x)=-4 for all x → gate≈0.018, output≈linear(x).
    Training: gate_net.weight learns to open the gate selectively per sample+concept.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.res_down = nn.Linear(in_features, hidden_dim, bias=True)
        self.res_up = nn.Linear(hidden_dim, out_features, bias=False)
        self.gate_net = nn.Linear(in_features, out_features, bias=True)

        nn.init.normal_(self.res_down.weight, std=0.01)
        nn.init.zeros_(self.res_down.bias)
        nn.init.zeros_(self.res_up.weight)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, -4.0)

        self.to(device)
        logger.info(
            "InputGatedRefinerCBL: in={} out={} hidden_dim={}",
            in_features,
            out_features,
            hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        h = torch.relu(self.res_down(x))
        correction = self.res_up(h)
        gate = torch.sigmoid(self.gate_net(x))
        return linear_out + gate * correction

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt")) as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        hidden_dim = int(args.get("cbl_residual_hidden_dim", 64))
        model = cls(encoder_dim, num_concepts, hidden_dim=hidden_dim, device=device)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "cbl.pt"), map_location=device)
        )
        return model


class ConceptCorrRefinerCBL(nn.Module):
    """Linear CBL with a low-rank concept-space correction.

    Main path: standard linear projection (identical to linear baseline).
    Concept correlation path: z → ReLU(rank-dim) → correction in concept space.
    Captures attribute co-occurrence: if linear misses one correlated concept,
    the correction learns to boost it from co-activated sibling concepts.
    Gate: learned per-concept scalar sigmoid, biased toward closed (-4 → ≈ 0.018).

    At init: corr_up.weight = 0 → correction = 0 → output = linear(x).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        corr_rank: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.out_features = out_features
        self.corr_rank = corr_rank

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.corr_down = nn.Linear(out_features, corr_rank, bias=True)
        self.corr_up = nn.Linear(corr_rank, out_features, bias=False)
        self.gate_bias = nn.Parameter(torch.full((out_features,), -4.0))

        nn.init.normal_(self.corr_down.weight, std=0.01)
        nn.init.zeros_(self.corr_down.bias)
        nn.init.zeros_(self.corr_up.weight)

        self.to(device)
        logger.info(
            "ConceptCorrRefinerCBL: in={} concepts={} rank={}",
            in_features,
            out_features,
            corr_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        h = torch.relu(self.corr_down(z))
        correction = self.corr_up(h)
        gate = torch.sigmoid(self.gate_bias)
        return z + gate * correction

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt")) as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        corr_rank = int(args.get("cbl_corr_rank", 16))
        model = cls(encoder_dim, num_concepts, corr_rank=corr_rank, device=device)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "cbl.pt"), map_location=device)
        )
        return model


class ConceptCorrGatedRefinerCBL(nn.Module):
    """Linear CBL with low-rank concept-space correction and bottleneck-adaptive gating.

    Extends ConceptCorrRefinerCBL: the per-concept gate is conditioned on the bottleneck
    h = ReLU(corr_down(z)) rather than being fixed per-concept. This allows the gate to
    open wider when concept co-occurrence evidence is strong (large h values) and fall
    back to the baseline gate when evidence is weak (h ≈ 0 via ReLU).

    Architecture:
        z = linear(x)                            # 512 → concepts (baseline path)
        h = ReLU(corr_down(z))                   # concepts → rank (bottleneck)
        correction = corr_up(h)                  # rank → concepts (zero-init)
        gate_input = gate_from_h(h)              # rank → concepts (zero-init)
        gate = sigmoid(gate_bias + gate_input)   # per-concept, adaptive
        output = z + gate * correction

    At init: corr_up.weight = 0, gate_from_h.weight = 0 → output = z (pure linear).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        corr_rank: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.out_features = out_features
        self.corr_rank = corr_rank

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.corr_down = nn.Linear(out_features, corr_rank, bias=True)
        self.corr_up = nn.Linear(corr_rank, out_features, bias=False)
        self.gate_bias = nn.Parameter(torch.full((out_features,), -4.0))
        self.gate_from_h = nn.Linear(corr_rank, out_features, bias=False)

        nn.init.normal_(self.corr_down.weight, std=0.01)
        nn.init.zeros_(self.corr_down.bias)
        nn.init.zeros_(self.corr_up.weight)
        nn.init.zeros_(self.gate_from_h.weight)

        self.to(device)
        logger.info(
            "ConceptCorrGatedRefinerCBL: in={} concepts={} rank={}",
            in_features,
            out_features,
            corr_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        h = torch.relu(self.corr_down(z))
        correction = self.corr_up(h)
        gate = torch.sigmoid(self.gate_bias + self.gate_from_h(h))
        return z + gate * correction

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt")) as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        corr_rank = int(args.get("cbl_corr_rank", 16))
        model = cls(encoder_dim, num_concepts, corr_rank=corr_rank, device=device)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "cbl.pt"), map_location=device)
        )
        return model


class ConceptCorrCenteredRefinerCBL(nn.Module):
    """Linear CBL with a low-rank concept-space correction on centered concept logits.

    Identical to ConceptCorrRefinerCBL except that the concept logits are mean-centered
    before feeding into the low-rank bottleneck:
        z_c = z - z.mean(dim=-1, keepdim=True)
        h = ReLU(corr_down(z_c))
        correction = corr_up(h)
        output = z + gate * correction

    Motivation: ConceptCorrRefinerCBL collapses to a rank-1 correction where the
    right singular vector of corr_up is nearly uniform (std=0.002, range/mean=0.06).
    This means the learned correction is a global offset (mean concept activity), not
    a concept co-occurrence correction. Centering z removes the trivially learnable
    global-mean direction, forcing the bottleneck to find genuine concept covariance.

    At init: corr_up.weight = 0 → correction = 0 → output = z. Identical to baseline.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        corr_rank: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.out_features = out_features
        self.corr_rank = corr_rank

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.corr_down = nn.Linear(out_features, corr_rank, bias=True)
        self.corr_up = nn.Linear(corr_rank, out_features, bias=False)
        self.gate_bias = nn.Parameter(torch.full((out_features,), -4.0))

        nn.init.normal_(self.corr_down.weight, std=0.01)
        nn.init.zeros_(self.corr_down.bias)
        nn.init.zeros_(self.corr_up.weight)

        self.to(device)
        logger.info(
            "ConceptCorrCenteredRefinerCBL: in={} concepts={} rank={}",
            in_features,
            out_features,
            corr_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        z_c = z - z.mean(dim=-1, keepdim=True)
        h = torch.relu(self.corr_down(z_c))
        correction = self.corr_up(h)
        gate = torch.sigmoid(self.gate_bias)
        return z + gate * correction

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt")) as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        corr_rank = int(args.get("cbl_corr_rank", 16))
        model = cls(encoder_dim, num_concepts, corr_rank=corr_rank, device=device)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "cbl.pt"), map_location=device)
        )
        return model


class CombinedRefinerCBL(nn.Module):
    """Linear CBL with both a feature-space residual and a concept-space correction.

    Main path: standard linear projection (identical to linear baseline).
    Feature-space branch: x → ReLU(feat_down) → feat_up → gated correction.
      Captures non-linear feature combinations the linear layer misses.
    Concept-space branch: z → ReLU(corr_down) → corr_up → gated correction.
      Captures concept co-occurrence; boosts under-activated correlated concepts.
    Both gates are per-concept scalars biased to -4 (≈ 0.018) at init.

    At init: both corrections = 0, output = linear(x).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        corr_rank: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.corr_rank = corr_rank

        self.linear = nn.Linear(in_features, out_features, bias=True)

        # feature-space residual branch
        self.feat_down = nn.Linear(in_features, hidden_dim, bias=True)
        self.feat_up = nn.Linear(hidden_dim, out_features, bias=False)
        self.gate_feat = nn.Parameter(torch.full((out_features,), -4.0))

        # concept-space correction branch
        self.corr_down = nn.Linear(out_features, corr_rank, bias=True)
        self.corr_up = nn.Linear(corr_rank, out_features, bias=False)
        self.gate_conc = nn.Parameter(torch.full((out_features,), -4.0))

        nn.init.normal_(self.feat_down.weight, std=0.01)
        nn.init.zeros_(self.feat_down.bias)
        nn.init.zeros_(self.feat_up.weight)
        nn.init.normal_(self.corr_down.weight, std=0.01)
        nn.init.zeros_(self.corr_down.bias)
        nn.init.zeros_(self.corr_up.weight)

        self.to(device)
        logger.info(
            "CombinedRefinerCBL: in={} concepts={} hidden_dim={} corr_rank={}",
            in_features,
            out_features,
            hidden_dim,
            corr_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        h_feat = torch.relu(self.feat_down(x))
        feat_corr = self.feat_up(h_feat)
        h_conc = torch.relu(self.corr_down(z))
        conc_corr = self.corr_up(h_conc)
        gate_feat = torch.sigmoid(self.gate_feat)
        gate_conc = torch.sigmoid(self.gate_conc)
        return z + gate_feat * feat_corr + gate_conc * conc_corr

    def save_model(self, save_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        with open(os.path.join(load_path, "args.txt")) as f:
            args = json.load(f)
        if args.get("use_clip_penultimate") and args.get("backbone", "").startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        hidden_dim = int(args.get("cbl_residual_hidden_dim", 64))
        corr_rank = int(args.get("cbl_corr_rank", 16))
        model = cls(encoder_dim, num_concepts, hidden_dim=hidden_dim, corr_rank=corr_rank, device=device)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "cbl.pt"), map_location=device)
        )
        return model


class NormalizationLayer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

    def save_model(self, save_dir):
        # save model
        torch.save(self.mean, os.path.join(save_dir, "train_concept_features_mean.pt"))
        torch.save(self.std, os.path.join(save_dir, "train_concept_features_std.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        mean = torch.load(
            os.path.join(load_path, "train_concept_features_mean.pt"),
            map_location=device,
        )
        std = torch.load(
            os.path.join(load_path, "train_concept_features_std.pt"),
            map_location=device,
        )
        normalization_layer = cls(mean, std, device=device)
        return normalization_layer


class FinalLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device: str = "cuda"):
        super().__init__(in_features, out_features, bias=True)
        self.to(device)

    def forward(self, x):
        return super().forward(x)

    def save_model(self, save_dir):
        # save model
        torch.save(self.state_dict(), os.path.join(save_dir, "final.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        num_classes = len(data_utils.get_classes(args["dataset"]))

        # create model
        model = cls(num_concepts, num_classes, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "final.pt")))
        return model


def load_cbm(load_dir, device):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)

    W_c = torch.load(os.path.join(load_dir, "W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(
        args["backbone"],
        W_c,
        W_g,
        b_g,
        proj_mean,
        proj_std,
        device,
        use_clip_penultimate=args.get("use_clip_penultimate", False),
    )
    return model


def load_std(load_dir, device):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)

    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = standard_model(
        args["backbone"],
        W_g,
        b_g,
        proj_mean,
        proj_std,
        device,
        use_clip_penultimate=args.get("use_clip_penultimate", False),
    )
    return model


def per_class_accuracy(
    model: torch.nn.Module, loader: DataLoader, classes: List[str], device: str = "cuda"
) -> Dict[str, float]:
    correct = torch.zeros(len(classes)).to(device)
    total = torch.zeros(len(classes)).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for features, _, targets in tqdm(loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            for pred, target in zip(preds, targets):
                total[target] += 1
                if pred == target:
                    correct[target] += 1

    per_class_accuracy = correct / total
    total_accuracy = correct.sum() / total.sum()
    total_datapoints = total.sum()

    # return a dictionary of class names and accuracies, and total accuracy
    return {
        "Per class accuracy": {
            classes[i]: f"{per_class_accuracy[i]*100.0:.2f}"
            for i in range(len(classes))
        },
        "Overall accuracy": f"{total_accuracy*100.0:.2f}",
        "Datapoints": f"{total_datapoints}",
    }


def validate_cbl(
    backbone: Backbone,
    cbl: ConceptLayer,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str = "cuda",
    cached_embeddings: Optional[torch.Tensor] = None,
    cached_concepts: Optional[torch.Tensor] = None,
    cache_batch_size: int = 512,
):
    val_loss = 0.0
    with torch.no_grad():
        logger.info("Running CBL validation")
        if cached_embeddings is not None and cached_concepts is not None:
            n_batches = 0
            for start_idx in tqdm(
                range(0, len(cached_embeddings), cache_batch_size),
                total=max(1, (len(cached_embeddings) + cache_batch_size - 1) // cache_batch_size),
            ):
                end_idx = start_idx + cache_batch_size
                embeddings = cached_embeddings[start_idx:end_idx].to(device)
                concept_one_hot = cached_concepts[start_idx:end_idx].to(device)
                concept_logits = cbl(embeddings)
                batch_loss = loss_fn(concept_logits, concept_one_hot)
                val_loss += batch_loss.item()
                n_batches += 1
            val_loss = val_loss / max(1, n_batches)
        else:
            for features, concept_one_hot, _ in tqdm(val_loader):
                features = features.to(device)
                concept_one_hot = concept_one_hot.to(device)

                # forward pass
                concept_logits = cbl(backbone(features))

                # calculate loss
                batch_loss = loss_fn(concept_logits, concept_one_hot)
                val_loss += batch_loss.item()

            # finalize metrics and update model
            val_loss = val_loss / len(val_loader)

    return val_loss


def train_cbl(
    backbone: Backbone,
    cbl: ConceptLayer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    loss_fn: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    concepts: Optional[List[str]] = None,
    tb_writer=None,
    device: str = "cuda",
    finetune: bool = False,
    optimizer: str = "sgd",
    scheduler: str = None,
    backbone_lr: float = 1e-3,
    data_parallel=False,
    cached_val_embeddings: Optional[torch.Tensor] = None,
    cached_val_concepts: Optional[torch.Tensor] = None,
    early_stop_patience: int = 0,
    min_delta: float = 0.0,
    min_epochs: int = 0,
):
    # setup optimizer
    base_optimizer_cls = None
    if optimizer == "sgd":
        base_optimizer_cls = torch.optim.SGD
        optimizer_kwargs = dict(lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == "adam":
        base_optimizer_cls = torch.optim.Adam
        optimizer_kwargs = dict(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError
    optimizer = base_optimizer_cls(cbl.parameters(), **optimizer_kwargs)
    if finetune:
        optimizer.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})

    # setup schedular
    if scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_val_loss_epoch = None
    best_model_state = None
    best_backbone_state = None
    epochs_without_improvement = 0
    if data_parallel:
        backbone = torch.nn.DataParallel(backbone)
        cbl = torch.nn.DataParallel(cbl)
    for epoch in range(epochs):
        train_loss = 0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Running CBL training for Epoch: {epoch}")
        its = tqdm(total=len(train_loader), position=0, leave=True)
        for batch_idx, (features, concept_one_hot, _) in enumerate(train_loader):
            features = features.to(device)  # (batch_size, feature_dim)
            concept_one_hot = concept_one_hot.to(device)  # (batch_size, n_concepts)

            def compute_batch_loss():
                if finetune:
                    backbone.train()
                    embeddings = backbone(features)
                else:
                    with torch.no_grad():
                        embeddings = backbone(features)
                concept_logits = cbl(embeddings)
                return loss_fn(concept_logits, concept_one_hot)

            batch_loss = compute_batch_loss()
            train_loss += batch_loss.item()

            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # print batch stats
            if (batch_idx + 1) % 1000 == 0:
                its.update(1000)
                print(
                    "Epoch: {} | Batch: {} | Loss: {:.6f}".format(
                        epoch, batch_idx, batch_loss.item()
                    )
                )

                # exit if loss is nan
                if torch.isnan(batch_loss):
                    # Exit process if loss is nan
                    logger.error(f"Loss is nan at epoch {epoch} and batch {batch_idx}")
                    sys.exit(1)
        backbone.eval()
        # finalize metrics and update model
        its.close()
        train_loss = train_loss / len(train_loader)
        # train_per_concept_roc = train_per_concept_roc.compute()

        # evaluate on validation set
        logger.info(f"Running CBL validation for Epoch: {epoch}")
        val_loss = validate_cbl(
            backbone,
            cbl.module if data_parallel else cbl,
            val_loader,
            loss_fn=loss_fn,
            device=device,
            cached_embeddings=cached_val_embeddings,
            cached_concepts=cached_val_concepts,
        )
        improved = val_loss < best_val_loss - min_delta
        if improved:
            logger.info(f"Updating best val loss at epoch: {epoch}")
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            best_backbone_state = backbone.state_dict()
            best_model_state = cbl.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("lr", lr, epoch)

        # print epoch stats
        logger.info(
            f"Epoch: {epoch} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        if (
            early_stop_patience
            and epoch + 1 >= min_epochs
            and epochs_without_improvement >= early_stop_patience
        ):
            logger.info(
                f"Early stopping CBL at epoch {epoch}: best val loss "
                f"{best_val_loss:.6f} at epoch {best_val_loss_epoch}, "
                f"no improvement greater than {min_delta:.6f} for "
                f"{epochs_without_improvement} epochs"
            )
            break

    # return best model based on validation loss
    logger.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")
    cbl.load_state_dict(best_model_state)
    backbone.load_state_dict(best_backbone_state)
    if data_parallel:
        cbl = cbl.module
        backbone = backbone.module
    return cbl, backbone


def test_model(
    loader: DataLoader,
    backbone: Backbone,
    cbl: ConceptLayer,
    normalization: NormalizationLayer,
    final_layer: FinalLayer,
    device: str = "cuda",
):
    acc_mean = 0.0
    for features, concept_one_hot, targets in tqdm(loader):
        features = features.to(device)
        concept_one_hot = concept_one_hot.to(device)
        targets = targets.to(device)

        # forward pass
        with torch.no_grad():
            embeddings = backbone(features)
            concept_logits = cbl(embeddings)
            concept_probs = normalization(concept_logits)
            logits = final_layer(concept_probs)

        # calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).sum().item()
        acc_mean += accuracy

    return acc_mean / len(loader.dataset)


def train_sparse_final(
    linear,
    indexed_train_loader,
    val_loader,
    n_iters,
    lam,
    step_size=0.1,
    device="cuda",
):
    # zero initialize
    num_classes = linear.weight.shape[0]
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = lam

    # Solve the GLM path
    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        step_size,
        n_iters,
        ALPHA,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(indexed_train_loader.dataset),
        n_classes=num_classes,
        verbose=True,
    )

    return output_proj


def train_dense_final(
    model,
    indexed_train_loader,
    val_loader,
    n_iters,
    lr=0.001,
    device="cuda",
):
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # setup schedular
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # setup loss
    ce_loss = torch.nn.CrossEntropyLoss()

    # train
    for epoch in range(n_iters):
        train_loss = 0
        val_loss = 0
        val_accuracy = 0

        # train
        for inputs, targets, _ in tqdm(indexed_train_loader, desc="Train"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward
            logits = model(inputs)

            # calculate loss
            batch_loss = ce_loss(logits, targets)
            train_loss += batch_loss.item()

            # optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss = train_loss / len(indexed_train_loader)

        # validation
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward
                logits = model(inputs)

                # calculate loss
                batch_loss = ce_loss(logits, targets)
                val_loss += batch_loss.item()

                # calculate metrics
                classes = torch.argmax(logits, dim=1)
                val_accuracy += (classes == targets).sum().item()
        val_accuracy = val_accuracy / len(val_loader.dataset) * 100.0
        val_loss = val_loss / len(val_loader)

        # print stats
        print(
            f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_accuracy}, lr: {optimizer.param_groups[0]['lr']}"
        )

        # Step the scheduler
        scheduler.step()

    output_proj = {}
    output_proj["path"] = [{}]
    output_proj["path"][0]["weight"] = model.weight
    output_proj["path"][0]["bias"] = model.bias
    output_proj["path"][0]["lr"] = lr
    for key in ("lam", "alpha", "time"):
        output_proj["path"][0][key] = -1.0
    output_proj["path"][0]["metrics"] = {"val_accuracy": val_accuracy}
    return output_proj
