from isaaclab.utils import configclass

from robot_lab.utils.wrappers.rsl_rl.rl_roa_cfg import (
    RslRlOnPolicyRunnerRoaCfg,
    RslRlActorCriticRoaCfg,
    RslRlAlgorithmRoaCfg
)

@configclass
class LWWheelRoughRoaRunnerCfg(RslRlOnPolicyRunnerRoaCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "LW_wheel_rough_roa"
    obs_groups = {
        "policy": ["policy"],       
        "critic": ["critic"],
        "privileged": ["privileged"]  # 专供特权编码器使用的纯物理参数组
    }
    policy = RslRlActorCriticRoaCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        priv_encoder_dims=[64, 20]
    )
    algorithm = RslRlAlgorithmRoaCfg(
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
        dagger_update_freq=20,
    )

@configclass
class LWWheelFlatRoaRunnerCfg(LWWheelRoughRoaRunnerCfg):
    max_iterations = 50000
    experiment_name = "LW_wheel_flat_roa"
