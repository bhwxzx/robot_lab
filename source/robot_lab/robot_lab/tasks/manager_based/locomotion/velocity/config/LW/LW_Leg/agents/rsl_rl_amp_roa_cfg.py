from isaaclab.utils import configclass

from robot_lab.utils.wrappers.rsl_rl.rl_amp_roa_cfg import (
    RslRlOnPolicyRunnerAmpRoaCfg,
    RslRlActorCriticAmpRoaCfg,
    RslRlAlgorithmAmpRoaCfg
)

@configclass
class LWRoughAmpRoaRunnerCfg(RslRlOnPolicyRunnerAmpRoaCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "LW_leg_rough_amp_roa"
    obs_groups = {
        "policy": ["policy"],       
        "critic": ["critic"],
        "amp": ["amp"],
        "privileged": ["privileged"]  # 专供特权编码器使用的纯物理参数组
    }
    policy = RslRlActorCriticAmpRoaCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        priv_encoder_dims=[256, 128, 20],
        vel_offset=41,  # 41
    )
    algorithm = RslRlAlgorithmAmpRoaCfg(
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
        priv_reg_coef_schedule=[0.0, 0.1, 2000, 3000],
        priv_reg_coef_schedule_resume=[0.0, 0.1, 0, 1],
        dagger_update_freq=20,
        vel_loss_coef=1.0,
    )
    amp_history_length=10
    amp_discr_hidden_dims=[1024, 512]
    amp_motion_files=["source/robot_lab/robot_lab/datasets/LW/motion_amp_expert/motion_*.txt"]
    amp_num_preload_transitions=200000
    amp_replay_buffer_size=100000
    disc_learning_rate=1.0e-4
    amp_reward_coef=3.0   # 2.0
    amp_task_reward_lerp=0.3

@configclass
class LWFlatAmpRoaRunnerCfg(LWRoughAmpRoaRunnerCfg):
    max_iterations = 50000
    experiment_name = "LW_leg_flat_amp_roa"
    policy = RslRlActorCriticAmpRoaCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        priv_encoder_dims=[64, 20],
        vel_offset=41,
    )
    amp_reward_coef=2.0
    amp_task_reward_lerp=0.7
