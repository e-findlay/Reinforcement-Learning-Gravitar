# this is a Deep Q Learning (DQN) agent including replay memory and a target network 
# you can write a brief 8-10 line abstract detailing your submission and experiments here
# This code implemented a noisy recurrent dqn with prioritised replay and the episodic memory
# and rnd exploration methods from the paper Never Give Up: Learning Directed Exploration
# Strategies. I found that a beta value of 0.3 works best for combining extrinsic and intrinsic 
# rewards. Most of the other hyperparameters were taken directly from the paper. Also from the NGU
# paper, I inject a vector of the previous beta value, intrinsic reward, reward and action to the 
# lstm layer in the network. The logs and video were generated using my own machine as the episodic reward is
# very slow to generate on colab.
# the code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py, which is released under the MIT licesne
# make sure you reference any code you have studied as above, with one comment line per reference


# imports
import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import cv2

# hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 30000
episode_limit = 5000
batch_size    = 32
video_every   = 10
print_every   = 5
L = 5
eps = 0.03
sm = 8
cluster_distance = 0.008
kernel_c = 0.001
beta = 0.3


# this code is based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/lib/dqn_extra.py
class PrioritizedReplayBuffer():
  def __init__(self, alpha=0.6, beta=0.4):
    self.buffer = []
    self.alpha = alpha
    self.priorities = np.zeros((buffer_limit,), dtype=np.float32)
    self.position = 0
    self.max_len = buffer_limit
    self.beta = beta
    self.beta_frames = 100000

  def __len__(self):
    return len(self.buffer)
  
  def update_beta(self, idx):
    value = 0.4 + idx * (1.0 - 0.4) / self.beta_frames
    self.beta = min(1.0, value)


  def put(self, transition):
    max_priority = self.priorities.max() if self.buffer else 1.0
    if len(self.buffer) < self.max_len:
      self.buffer.append(transition)
    else:
      self.buffer[self.position] = transition
    self.priorities[self.position] = max_priority
    self.position = (self.position + 1) % self.max_len

  def update_priorities(self, idxs, priorities):
    for idx, p in zip(idxs, priorities):
      self.priorities[idx] = p

  def sample(self, n):
    if len(self.buffer) == self.max_len:
      priorities = self.priorities
    else:
      priorities = self.priorities[:self.position]
    probs = priorities ** self.alpha
    probs /= probs.sum()
    indices = np.random.choice(len(self.buffer), n, p=probs)
    samples = [self.buffer[idx] for idx in indices]
    total = len(self.buffer)
    weights = (total * probs[indices]) ** (-self.beta)
    s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, h_lst, c_lst, vector = [], [], [], [], [], [], [], []
    s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, h_lst, c_lst, vector = zip(*samples)
    return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst), torch.tensor(h_lst), torch.tensor(c_lst), indices, torch.tensor(weights, dtype=torch.float), torch.tensor(vector)




class EpisodicMemory():
  def __init__(self):
      self.buffer = []
      self.length = 0
  
  def store_state(self, embedding):
    self.length += 1
    if self.length >= episode_limit:
      self.length -= 1
      self.buffer.pop(0)
    self.buffer.append(embedding)

  
  def __len__(self):
      return self.length
  
  def clear(self):
    self.buffer = []
    self.length = 0

  def KNN(self, obs, k=10):
      distances = []
      if self.length < k:
        for i in self.buffer:
          distances.append(np.linalg.norm(i - obs))
        return self.buffer, self.length, distances

      buffer = np.array(self.buffer)
      for i in buffer:
        distances.append((i, np.linalg.norm(i - obs)))
      distances.sort(key=lambda x: x[1])
      N = []
      d = []
      for i in range(k):
        N.append(distances[i][0])
        d.append(distances[i][1])
      return N, k, d


def K(d):
    return eps / (d + eps)


# this code is based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/lib/dqn_extra.py
class FactorizedGaussianNoisyLinear(nn.Linear):
    def __init__(self, n_in, n_out, sigma=0.4, bias=True):
        super(FactorizedGaussianNoisyLinear, self).__init__(n_in, n_out, bias=bias)
        self.sigma = sigma / math.sqrt(n_in)
        w = torch.full((n_out, n_in), self.sigma)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(n_out, n_in)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(n_out, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
          w = torch.full((n_out,), sigma)
          self.sigma_bias = nn.Parameter(w)

    def forward(self, X):
      self.epsilon_input.normal_()
      self.epsilon_output.normal_()
      func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
      eps_input = func(self.epsilon_input.data)
      eps_output = func(self.epsilon_output.data)

      bias = self.bias
      if bias is not None:
        bias = bias + self.sigma_bias * eps_output.t()
      noise = torch.mul(eps_input, eps_output)
      v = self.weight + self.sigma_weight * noise
      return F.linear(X, v, bias)

      

# _get_conv_out_shape code from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/lib/dqn_model.py
class DRQN(nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out = self._get_conv_out_shape(env.observation_space.shape)
        self.lstm = nn.LSTMCell(conv_out + 4, hidden_size=512)

        self.adv = nn.Sequential(
          FactorizedGaussianNoisyLinear(512, 512),
          nn.ReLU(),
          FactorizedGaussianNoisyLinear(512, env.action_space.n)
        )
        self.val = nn.Sequential(
          FactorizedGaussianNoisyLinear(512, 512),
          nn.ReLU(),
          FactorizedGaussianNoisyLinear(512, 1)
        )
    def _get_conv_out_shape(self, input):
      out = self.conv_layers(torch.zeros(1, *input))
      return int(np.prod(out.size()))


    def forward(self, x, h, c, vector, batch_size=1):
        x = x.float() / 256
        x = self.conv_layers(x).view(x.size()[0],-1)
        x = torch.cat((x, vector), 1)
        h, c = self.lstm(x, (h, c))
        adv = self.adv(h)
        val = self.val(h)
        return val + adv - adv.mean(1).unsqueeze(1).expand(batch_size, env.action_space.n), h, c



class RNDNetwork(nn.Module):
    def __init__(self):
      super(RNDNetwork, self).__init__()
      self.conv_layers = nn.Sequential(
          nn.Conv2d(env.observation_space.shape[0],32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(32,64,kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.ReLU()
      )
      conv_out = self._get_conv_out_shape(env.observation_space.shape)

      self.linear_layers = nn.Sequential(
          nn.Linear(conv_out, 128),
          nn.Softmax(dim=1)
      )
    def _get_conv_out_shape(self,input):
      out = self.conv_layers(torch.zeros(1, *input))
      return int(np.prod(out.size()))

    def forward(self, X):
      X = self.conv_layers(X).view(X.size()[0],-1)
      target = self.linear_layers(X)
      return target
      

class RND(nn.Module):
    def __init__(self):
      super(RND, self).__init__()
      self.target = RNDNetwork()
      self.predictor = RNDNetwork()

      for param in self.target.parameters():
        param.requires_grad = False
    
    def forward(self, X):
      X = X.float()
      target = self.target(X)
      prediction = self.predictor(X)
      return target, prediction


class EmbeddingNetwork(nn.Module):
  def __init__(self):
    super(EmbeddingNetwork, self).__init__()
    self.conv_layers = nn.Sequential(
          nn.Conv2d(env.observation_space.shape[0],32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(32,64,kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.ReLU()
    )
    out = self._get_conv_out_shape(env.observation_space.shape)

    self.linear_layers = nn.Sequential(
          nn.Linear(out, 32),
          nn.ReLU()
    )
  def _get_conv_out_shape(self, input):
    out = self.conv_layers(torch.zeros(1, *input))
    return int(np.prod(out.size()))
  
  def forward(self, X):
    X = X.float()
    X = self.conv_layers(X).view(X.size()[0],-1)
    X = self.linear_layers(X)
    return X
  
  
class SiameseEmbeddingNetwork(nn.Module):
  def  __init__(self):
    super(SiameseEmbeddingNetwork, self).__init__()
    self.s_net = EmbeddingNetwork()
    self.s_prime_net = EmbeddingNetwork()

    self.joint_layers = nn.Sequential(
          nn.Linear(64, 128),
          nn.ReLU(),
          nn.Linear(128,env.action_space.n)
    )
  
  def forward(self, X1, X2):
    X1 = self.s_net(X1)
    X2 = self.s_prime_net(X2)
    out = torch.cat((X1, X2), 1)
    out = self.joint_layers(out)
    return F.softmax(out, dim=1)


class Agent():
  def __init__(self, env):
    self.env = env
    self.memory = PrioritizedReplayBuffer()
    self.episodic_memory = EpisodicMemory()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = DRQN().to(self.device)
    self.target = DRQN().to(self.device)
    self.RND = RND().to(self.device)
    self.embedding_net = SiameseEmbeddingNetwork().to(self.device)
    self.target.load_state_dict(self.model.state_dict())
    self.optimizer = optim.Adam(self.model.parameters(), learning_rate)
    self.rnd_optimizer = optim.Adam(self.RND.predictor.parameters(), learning_rate)
    self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), learning_rate)
    self.loss = nn.SmoothL1Loss()
    self.embedding_loss = nn.NLLLoss()

  
  def sample_action(self, obs, hidden, vector):
    obs = obs.to(self.device)
    h, c = hidden
    h = h.to(self.device)
    c = c.to(self.device)
    vector = vector.to(self.device)
    out, new_h, new_c = self.model(obs, h, c, vector)
    return out.argmax().item(), new_h.detach().cpu(), new_c.detach().cpu()
  
  def intrinsic_reward(self, obs):
    obs = obs.to(self.device)
    target, prediction = self.RND(obs)
    intrinsic_reward = (prediction - target).pow(2).sum(1) / 2
    return intrinsic_reward.cpu()
  
  def episodic_reward(self, obs):
    obs = obs.cpu().numpy()
    s = 0
    N, k, distances = self.episodic_memory.KNN(obs)
    distances = np.array(distances)
    moving_average = 0
    for i in range(k):
      moving_average = (moving_average * i + distances[i]) / (i+1)
      distances[i] = np.divide(distances[i], moving_average, out=np.zeros_like(distances[i]), where=moving_average!=0)
      distances[i] = np.maximum(distances[i] - cluster_distance, 0)
    for i in range(k):
      s += K(distances[i])
    s = np.sqrt(s) + kernel_c
    if s > sm:
      return 0
    else:
      
      return 1 / s.item()
    
  
  def reset_env(self):
    state = self.env.reset()
    return state
  
  def encode(self, state, next_state):
    state, next_state = state.to(self.device), next_state.to(self.device)
    prob_action = self.embedding_net(state, next_state)
    return prob_action

  
  def update(self):
    s, a, r, s_prime, done_mask, h, c, indices, weights, vectors = self.memory.sample(batch_size)
    s = s.to(self.device)
    a = a.to(self.device)
    r = r.to(self.device)
    h = h.to(self.device)
    c = c.to(self.device)
    vectors = vectors.to(self.device)
    s_prime = s_prime.to(self.device)
    done_mask = done_mask.to(self.device)
    weights = weights.to(self.device)
    h = h.view(batch_size, 512)
    c = c.view(batch_size, 512)
    vectors = vectors.view(batch_size, 4)
    q_values, _, _ = self.model(s, h, c, vectors, batch_size)
    q_values = q_values.gather(1,a.unsqueeze(1)).squeeze(1)
    max_q_prime, _, _ = self.target(s_prime, h, c, vectors, batch_size)
    max_q_prime = max_q_prime.max(1)[0].detach()
    target = r + gamma * max_q_prime * done_mask
    loss = (q_values - target) ** 2
    loss = loss * weights
    priorities = loss + 1e-5
    loss = loss.mean()
    self.memory.update_priorities(indices, priorities.data.cpu().numpy())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    intrinsic_reward = self.intrinsic_reward(s[-5:])
    self.rnd_optimizer.zero_grad()
    intrinsic_reward.sum().backward()
    self.rnd_optimizer.step()
    embeddings = self.embedding_net(s[-5:], s_prime[-5:])
    embedding_loss = self.embedding_loss(embeddings, a[-5:])
    self.embedding_optimizer.zero_grad()
    embedding_loss.backward()
    self.embedding_optimizer.step()

# these wrappers are based on https://github.com/openai/baselines/tree/master/baselines

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxSkipEnv(gym.Wrapper):
  def __init__(self, env=None, skip=4):
    super(MaxSkipEnv, self).__init__(env)
    self.obs_buffer = collections.deque(maxlen=2)
    self.skip = skip

  def step(self, action):
    total = 0.0
    done = None
    for _ in range(self.skip):
      obs, reward, done, info = self.env.step(action)
      total += reward
      if done:
        break
    max_frame = np.max(np.stack(self.obs_buffer), axis=0)
    return max_frame, total, done, info
  
  def reset(self):
    self.obs_buffer.clear()
    obs = self.env.reset()
    self.obs_buffer.append(obs)
    return obs

class to84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(to84x84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return to84x84.process(obs)

    @staticmethod
    def process(frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class NormalizeImage(gym.ObservationWrapper):
  def observation(self, obs):
    return np.array(obs).astype(np.float32) / 255.0

    
def make_env(env_name):
  env = gym.make(env_name)
  env = MaxSkipEnv(env)
  env = FireResetEnv(env)
  env = to84x84(env)
  env = ImageToPyTorch(env)
  env = FrameStack(env, 4)
  env = NormalizeImage(env)
  return env

# this code is based on https://github.com/liyanage/python-modules/blob/master/running_stats.py
class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
    
    def clear(self):
        self.n = 0
        
    def push(self, x):
        self.n += 1
        
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0
    
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
        
    def standard_deviation(self):
        return math.sqrt(self.variance())

  # setup the Gravitar ram environment, and record a video every 50 episodes. You can use the non-ram version here if you prefer
env = make_env('Gravitar-v0')
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0,force=True)

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)
agent = Agent(env)
rnd_running = RunningStats()
steps = 0

score    = 0.0
marking  = []

for n_episode in range(int(1e32)):
    s = agent.reset_env()
    done = False
    score = 0.0
    if n_episode % 5 == 0:
      agent.episodic_memory.clear()
    h = torch.zeros(1,512)
    c = torch.zeros(1,512)
    hidden = (h,c)
    vector = torch.zeros(1,4)

    while True:
        steps += 1
        a, new_h, new_c = agent.sample_action(torch.from_numpy(s).float().unsqueeze(0), hidden, vector)
        s_prime, r, done, info = agent.env.step(a)
        state_encoding = agent.encode(torch.from_numpy(s).float().unsqueeze(0), torch.from_numpy(s_prime).float().unsqueeze(0)).detach()
        r_i = agent.episodic_reward(state_encoding)
        alpha = agent.intrinsic_reward(torch.from_numpy(s).float().unsqueeze(0)).detach().clamp(-1.0, 1.0).item()
        rnd_running.push(alpha)
        rnd_mean, rnd_std = rnd_running.mean(), rnd_running.standard_deviation()
        if rnd_std > 0:
          alpha = 1 + (alpha - rnd_mean) / rnd_std
        else:
          alpha = 1 + (alpha - rnd_mean)
        curiosity_reward = r_i * min(max(1,alpha),L)
        combined_reward = r + beta * curiosity_reward
        combined_reward = combined_reward / 100.0
        agent.episodic_memory.store_state(state_encoding.cpu().numpy())
        done_mask = 0.0 if done else 1.0
        new_vector = torch.tensor([[a, r/100.0, curiosity_reward, beta]])
        agent.memory.put((s,a,combined_reward,s_prime, done_mask, h.numpy(), c.numpy(), vector.numpy()))
        vector = new_vector
        s = s_prime
        h = new_h
        c = new_c
        score += r
        if steps % 3000 == 0:
          agent.memory.update_beta(steps // 3000)
        if done:
            agent.episodic_memory.clear()
            break
        
    if len(agent.memory)>2000:
        agent.update()

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    if n_episode%100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []
        

    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode%print_every==0 and n_episode!=0:
        agent.target.load_state_dict(agent.model.state_dict())
        print("episode: {}, score: {:.1f}".format(n_episode, score))
