from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
)

@configclass
class LWRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "LW_wheel_rough_teacher"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        state_dependent_std=False,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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

@configclass
class LWRoughDistillationRunnerCfg(RslRlDistillationRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 20000  
    save_interval = 1000
    experiment_name = "LW_wheel_rough_teacher"  # 会去这里寻找load_run
    obs_groups = {
        "policy": ["policy"],       # student inputs
        "teacher": ["teacher"],     # teacher inputs (same as teacher training)
    }
    
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,      # 每个迭代更新学习多少次
        learning_rate=1e-4,         
        optimizer="adam",
        loss_type="mse",            # 均方误差 (模仿 Teacher 动作)
        gradient_length=1,          # 对于 MLP (非递归)，梯度长度通常设为 1
        max_grad_norm=1.0,
    )

    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        
        student_hidden_dims=[512, 256, 128],  # 学生网络结构 
        student_obs_normalization=True,      # 开启学生输入归一化
        init_noise_std=0.1,                  # 初始噪声 (虽然蒸馏主要看 MSE，但保留很小的噪声)
        
        # --- 教师网络配置 (用于加载预训练权重) ---
        # 这里的结构必须与训练好的 PPO Teacher 完全一致！
        teacher_hidden_dims=[512, 256, 128], 
        teacher_obs_normalization=True,
        
        activation="elu",
    )
