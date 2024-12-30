import h5py

from exputils.drls.buffer import TransitionBuffer
from exputils.drls.env import get_env_info, make_env, reset_env_fn

env_id = "CartPole-v1"
seed = 10
env = make_env(env_id)
env_info = get_env_info(env)
print(env_info)

buffer = TransitionBuffer(
    state_shape=env_info["state_shape"],
    action_shape=env_info["action_shape"],
    action_dtype=env_info["action_dtype"],
    device="cpu",
    buffer_size=8000,
)

print("Cur size buffer:", buffer.size)

next_state, _ = reset_env_fn(env, seed)

for t in range(5):
    state = next_state
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    print("state:", state, "action:", action, "reward:", reward)
    buffer.insert_transition(state, action, next_state, reward, float(terminated))

print("states in buffer:", buffer.buffers[0])
print("actions in buffer:", buffer.buffers[1].squeeze())
print("reward in buffer", buffer.buffers[3].squeeze())


buffer.save_buffer(save_dir=".", file_name="buffer")

# load h5 file
loaded_dataset = h5py.File("buffer.hdf5", "r")

buffer.insert_dataset(loaded_dataset)
print("states in buffer:", buffer.buffers[0])
print("actions in buffer:", buffer.buffers[1].squeeze())
print("reward in buffer", buffer.buffers[3].squeeze())
