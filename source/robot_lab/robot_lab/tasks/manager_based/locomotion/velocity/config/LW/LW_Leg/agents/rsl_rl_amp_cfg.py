from importlib.resources import files

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlSymmetryCfg
from robot_lab.utils.wrappers.rsl_rl.rl_amp_cfg import (
    RslRlAlgorithmAmpCfg,
    RslRlOnPolicyRunnerAmpCfg
)


@configclass
class LWRoughAmpRunnerCfg(RslRlOnPolicyRunnerAmpCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "LW_leg_rough_amp"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlAlgorithmAmpCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func="robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry.lw_leg:compute_symmetric_states"
        ),
    )
    amp_history_length=10
    amp_discr_hidden_dims=[1024, 512, 256]
    amp_motion_files=["source/robot_lab/robot_lab/datasets/LW_Leg/motion_amp_expert/motion_*.txt"]
    amp_num_preload_transitions=200000
    amp_replay_buffer_size=100000
    disc_learning_rate=1.0e-4
    amp_reward_coef=0.3
    amp_task_reward_lerp=0.7

@configclass
class LWFlatAmpRunnerCfg(LWRoughAmpRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 50000
        self.experiment_name = "LW_leg_flat_amp"


@configclass
class LWFlatCommandAmpRunnerCfg(LWFlatAmpRunnerCfg):
    amp_dataset_version: str = "lw_grid153_125hz_tracking_002_heading_velocity"
    amp_stand_threshold: float = 0.1
    amp_reward_coef=2.0
    amp_task_reward_lerp=0.3
    amp_num_preload_transitions = 0  # Windows are gathered from the selected clip during each AMP update.

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "LW_leg_flat_command_amp"
        self.algorithm.class_name = "CommandConditionedAMPPPO"
        self.algorithm.symmetry_cfg.data_augmentation_func = (
            "robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry.lw_leg:compute_flat_amp_symmetric_states"
        )
        release = files("robot_lab").joinpath("datasets", "LW", self.amp_dataset_version)
        self.amp_motion_files = [str(release.joinpath("motion_amp_expert", "motion_*.txt"))]
