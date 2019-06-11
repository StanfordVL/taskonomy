ckpt_string = 'model_checkpoint_path: "/home/ubuntu/s3/model_log_final/{task}/logs/slim-train/time/model.ckpt-{step}"\nall_model_checkpoint_paths: "/home/ubuntu/s3/model_log_final/{task}/logs/slim-train/time/model.ckpt-{step}"'
with open("checkpoint", "w") as text_file:
    print(ckpt_string.format(task="keypoint3d", step="112830"), file=text_file)
