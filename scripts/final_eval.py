from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import random
import sys
from collections import Counter
from av_irl import calculate_safe_distance


expert = PPO.load('model/expert_ppo_mlt_h1_m_h2') # 22.15 22.16 12.775
# learner = PPO.load("model/gail_learner_august_e8k_ts100000_ts100000_mlt_old") # reward:20.15 22 12.45
learner = PPO.load("model/expert_ppo_mlt_h1_m_h2")

cal_cor = False

if sys.argv[1] == "e":
    model = expert
elif sys.argv[1] == "l":
    model = learner
elif sys.argv[1] == "c":
    cal_cor = True
    model = expert

if sys.argv[2] == "r":
    env = gym.make('roundabout-v0', render_mode='rgb_array')
elif sys.argv[2] == "i":
    env = gym.make('intersection-v1', render_mode='rgb_array')
elif sys.argv[2] == "h2":
    env = gym.make('highway-v0', render_mode='rgb_array')
elif sys.argv[2] == "h":
    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    # env = gym.make('highway-v0', render_mode='rgb_array')
    env.config["vehicles_count"] = 30
    env.config["vehicles_density"] = 1.5
elif sys.argv[2] == "h3":
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.config["vehicles_count"] = 60
    env.config["vehicles_density"] = 1.2
    env.config["lanes_count"] = 5
elif sys.argv[2] == "m":
    env = gym.make('merge-v0', render_mode='rgb_array')


s = [3,12,15,22,26,33,37,41,48,50,56,63,67,71,76,85,88,92,97,137,210]
rnd = 2


def generate_seeds(n):
    return [random.randint(0, 4294967295) for _ in range(n)]

if len(sys.argv) == 4:
    arg_seed = sys.argv[3]
    s = generate_seeds(int(arg_seed))

s = [3693441017, 1379311007, 345399021, 3866570930, 1440486917, 433119437, 941627344, 1392760376, 592577554, 1264255133, 741198289, 85302750, 1410076489, 211663658, 989135185, 1570779053, 507304143, 3495408044, 1103807189, 1596059065, 42805350, 521199278, 3329126507, 2708077417, 3106557841, 1064915112, 376899085, 2323231202, 657042672, 2162942099, 2170917990, 1832670571, 174373067, 2024159871, 2910350503, 3654751374, 3083785470, 32320730, 4258597804, 913536181, 1610817831, 3026220863, 945250660, 571319771, 271808935, 306855096, 4053625423, 739177828, 2960126609, 3657911354, 870113469, 3261250014, 3604468301, 696297419, 2865412639, 1168047477, 1445881685, 3788576699, 3863289598, 303639681, 3238666515, 1256008546, 2918054877, 1196236626, 445641807, 3220468212, 3696816633, 2661084908, 1106403187, 28265912, 4072718319, 1657821793, 1892593215, 2596180305, 2272646799, 3761416187, 2994571515, 1023992424, 3244354341, 667574573, 147697477, 902146094, 1340136995, 2400939166, 3673613530, 455156064, 1303219402, 1080440186, 1213997960, 3841494078, 1089289909, 1395852106, 887900089, 2140352646, 3875150970, 3302560521, 2600555133, 80910033, 3341670772, 2324072196]

rnd = len(s)


r = []
dl = []
rl = []
crash = 0
safe = []

for i in range(rnd):
    obs, info = env.reset(seed=s[i])
    #obs, info = env.reset(seed=48)
    #obs, info = env.reset()
    done = truncated = False
    score = 0
    acts = []
    rs = []
    n = 0
    data = []
    data_r = []
    pen = 0.0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        acts.append(action)
        obs, reward, done, truncated, info = env.step(action)
        rs.append(reward)
        pen += calculate_safe_distance(info)
        # print(info)
        tmp = (n,int(action))
        score += reward
        tmp2 = (n, reward)
        env.render()
        n += 1
        data.append(tmp)
        data_r.append(tmp2)
    r.append(score)
    safe.append(pen)

    # print(f"score: {score}, seed:{s[i]}")
    # print(acts)
    dl.append(data)
    rl.append(data_r)
    print(acts)
    print(rs)

    if sys.argv[2] == "m":
        if len(acts) != 17:
            crash += 1
    else:
        if len(acts) != 30:
            crash += 1



    # print(data_r)
    # print(i, len(acts))

#print(dl)
#print(rl)

print(f"{sys.argv[1]} mean score: {np.mean(r)}")
print(f"{sys.argv[1]} median score: {np.median(r)}")
print(f"mean safe distance penalty: {np.mean(safe)}")

if cal_cor:
    crash2 = 0
    r = []
    dl2 = []
    rl2 = []
    safe2 = []
    model = learner
    for i in range(rnd):
        obs, info = env.reset(seed=s[i])

        #obs, info = env.reset(seed=48)
        #obs, info = env.reset()
        done = truncated = False
        score = 0
        acts = []
        n = 0
        data = []
        data_r = []
        rs = []
        pen = 0.0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            acts.append(action)
            obs, reward, done, truncated, info = env.step(action)
            rs.append(reward)
            pen += calculate_safe_distance(info)
            #print(info)
            tmp = (n,int(action))
            score += reward
            tmp2 = (n, reward)
            env.render()
            n += 1
            data.append(tmp)
            data_r.append(tmp2)
        r.append(score)
        safe2.append(pen)
        # print(f"score: {score}, seed:{s[i]}")
        # print(acts)
        dl2.append(data)
        rl2.append(data_r)
        # print(data)
        # print(data_r)
        # print(i, len(acts))
        print(acts)
        print(rs)
        if sys.argv[2] == "m":
            if len(acts) != 17:
                crash2 += 1
        else:
            if len(acts) != 30:
                crash2 += 1

    print(f"{sys.argv[1]} mean score: {np.mean(r)}")
    print(f"{sys.argv[1]} median score: {np.median(r)}")
    print(f"mean safe distance penalty: {np.mean(safe2)}")

    ca, cr = [], []
    ns = []
    for i in range(len(dl2)):
        if len(dl2[i]) != len(dl[i]):
            print(f"seed {s[i]}, len(e): {len(dl[i])}, len(l): {len(dl2[i])}")
            continue
        x1, y1 = zip(*dl[i])
        x2, y2 = zip(*dl2[i])

        if np.var(y1) == 0 or np.var(y2) == 0:
            continue

        correlation_coefficient = round(np.corrcoef(y1, y2)[0, 1], 4)
        ca.append(correlation_coefficient)
        ns.append(s[i])

        x1, y1 = zip(*rl[i])
        x2, y2 = zip(*rl2[i])
        correlation_coefficient = round(np.corrcoef(y1, y2)[0, 1], 4)
        cr.append(correlation_coefficient)
    if len(ca) != 0:
        max_a = max(ca)
        min_a = min(ca)
        max_r = max(cr)
        min_r = min(cr)

        print(ca)
        print(cr)
        for ii in range(len(ca)):
            print(ca[ii],cr[ii],ns[ii])

        print(max_a, min_a, max_r, min_r)

        idx_xa = ca.index(max_a)
        idx_xr = cr.index(max_r)

        max_a_seed = s[idx_xa]
        max_r_seed = s[idx_xr]

        print(f"max action cor: {max_a:.2f}, reward: {cr[idx_xa]}, seed: {max_a_seed}, min action cor: {min_a:.2f}")
        print(f"max reward cor: {max_r:.2f}, action: {ca[idx_xr]}, seed: {max_r_seed}, min reward cor: {min_r:.2f}")

        print(f"Action's correlation between the expert and the learner: {np.median(ca):.2f}")
        print(f"Reward's correlation between the expert and the learner: {np.median(cr):.2f}")

print(f"seeds: {s}")
