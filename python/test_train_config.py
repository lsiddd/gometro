from train import (
    FRAME_STACK,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_TARGET_KL,
)


def test_ppo_hyperparameters_are_consistent_for_frame_stack_training():
    assert FRAME_STACK == 4
    assert PPO_N_STEPS % PPO_BATCH_SIZE == 0
    assert PPO_BATCH_SIZE <= PPO_N_STEPS
    assert PPO_N_EPOCHS >= 1
    assert 0.99 <= PPO_GAMMA < 1.0
    assert 0.9 <= PPO_GAE_LAMBDA <= 1.0
    assert 0.05 <= PPO_CLIP_RANGE <= 0.3
    assert PPO_TARGET_KL > 0
