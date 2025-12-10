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
    experiment_name = "LW_leg_rough_teacher"
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
    max_iterations = 10000  
    save_interval = 1000
    experiment_name = "LW_leg_rough_student"
    obs_groups = {
        "policy": ["policy"],       # student inputs
        "teacher": ["critic"],     # teacher inputs (same as teacher training)
    }
    
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=2,      # 每个迭代更新多少次
        learning_rate=1e-3,         
        optimizer="adam",
        loss_type="mse",            # 均方误差 (模仿 Teacher 动作)
        gradient_length=15,          # 对于 MLP (非递归)，梯度长度通常设为 1
        max_grad_norm=1.0,
    )

    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        
        student_hidden_dims=[512, 256, 128],  # 学生网络结构 (可以比 Teacher 小)
        student_obs_normalization=True,      # 开启学生输入归一化
        init_noise_std=1.0,                  # 初始噪声 (虽然蒸馏主要看 MSE，但保留探索机制)
        
        # --- 教师网络配置 (用于加载预训练权重) ---
        # 这里的结构必须与训练好的 PPO Teacher 完全一致！
        teacher_hidden_dims=[512, 256, 128], 
        teacher_obs_normalization=True,
        
        activation="elu",
    )
    
    # 指定 Teacher 模型路径
    # load_run = "my_ppo_teacher_run"  # Teacher 的 run_name
    # load_checkpoint = "model_20000.pt" # Teacher 的具体 checkpoint