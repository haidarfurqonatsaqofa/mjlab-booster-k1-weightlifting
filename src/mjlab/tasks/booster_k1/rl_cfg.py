from dataclasses import dataclass, field

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class BoosterK1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "booster_k1_approach_walk"
    logger: str = "tensorboard"
    clip_actions: float | None = 4.0

    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("policy",), "critic": ("policy",)}
    )

    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        )
    )

    critic: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
        )
    )

    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
        )
    )