from isaaclab.utils import configclass

from robot_lab.utils.wrappers.rsl_rl.rl_dwaq_cfg import (
    RslRlOnPolicyRunnerDwaqCfg,
    RslRlActorCriticDwaqCfg,
    RslRlAlgorithmDwaqCfg
)

@configclass
class LWRoughDwaqRunnerCfg(RslRlOnPolicyRunnerDwaqCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "LW_leg_rough_dwaq"
    obs_groups = {
        "policy": ["policy"],       
        "critic": ["critic"]     
    }
    policy = RslRlActorCriticDwaqCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        vae_hidden_dims=[128, 64],
        latent_dim=16, # 环境参数隐变量
        velocity_dim=3, # 显示监督速度维度
    )
    algorithm = RslRlAlgorithmDwaqCfg(
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
        vae_beta=1.0, # vae系数
        obs_dim=41    # 本体感知的观测维度
    )

@configclass
class LWFlatDwaqRunnerCfg(LWRoughDwaqRunnerCfg):

    max_iterations = 50000
    experiment_name = "LW_leg_flat_dwaq"