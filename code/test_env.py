import os
from pr_classes import World
from constants import DATASET_PATH, WORLD_FILE, N

#define the world
env = World()
env_input_file = os.path.join(DATASET_PATH, WORLD_FILE)
env.setup(env_input_file)

# test world settings
print(f'These are the first {N} landmarks in the world:')
test_landmarks = env.landmarks[:N]
for i in range(N):
    lm = test_landmarks[i]
    print('Landmark ID {} has coordinates x={}, y={}, z={}'.format(lm.id, lm.x, lm.y, lm.z))