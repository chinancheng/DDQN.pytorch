class Config(object):
    lr = 1e-4
    discount_factor = 0.99
    reply_buffer_size = 50000
    total_episode = 300000
    update_target_frequency = 5
    initial_epsilon = 0.1
    min_epsilon = 0.0001
    epsilon_discount_rate = 1e-5
    save_video_frequency = 500
    save_logs_frequency = 500
    show_loss_frequency = 10
    batch_size = 32
    initial_observe_episode = 20
    screen_width = 84
    screen_height = 84
