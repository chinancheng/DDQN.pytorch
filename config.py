class Config(object):
    lr = 1e-6
    discount_factor = 0.99
    reply_buffer_size = 50000
    total_episode = 200000
    update_target_frequency = 2000
    initial_epsilon = 0.1
    min_epsilon = 0.0001
    epsilon_discount_rate = 0.95
    update_epsilon_frequency = 100
    save_film_frequency = 10000
    save_model_frequency = 1000
    batch_size = 32
    initial_observe_episode = 20
    screen_width = 84
    screen_height = 84
