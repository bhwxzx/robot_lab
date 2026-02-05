from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
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
    )
    amp_discr_hidden_dims=[1024, 512, 256]
    amp_motion_files=["source/robot_lab/robot_lab/datasets/LW_Leg/motion_amp_expert/motion_.*.txt"]
    amp_num_preload_transitions=200000
    amp_replay_buffer_size=100000
    amp_reward_coef=0.3
    amp_task_reward_lerp=0.7

@configclass
class LWFlatAmpRunnerCfg(LWRoughAmpRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 50000
        self.experiment_name = "LW_leg_flat_amp"