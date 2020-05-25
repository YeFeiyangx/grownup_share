#%%
import numpy as np
np.zeros((3,3))

# %%

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    ## 哈希状态。这个是什么神仙玩意。。。啊噗噗噗。。。。
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')


def get_all_states_impl(current_state, current_symbol, all_states):
    global Cal_List
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                Cal_List[0] += 1
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    Cal_List[0] += 1
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)


def get_all_states():
    global Cal_List
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    Cal_List[0] += 1
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states

Cal_List = [0]
# all possible board configurations
all_states = get_all_states()
print(Cal_List[0])
print(len(all_states))
#%%
import numpy as np
a = np.zeros((4, 4))
print(a)
for i in range(3):
    a[i, i] = 1
    a[i, 3 - i] = -1
    
print(a)
# %%
a = np.zeros((4, 4))
print(a)
for i in range(1,4):
    a[i, i] = 1
    a[i, 3 - i] = -1
print(a)

# %%
a = np.zeros((4, 4))
print(a)
for i in range(3):
    a[i+1, i] = 1
    a[i+1, 3 - i] = -1
print(a)

# %%
a = np.zeros((4, 4))
print(a)
for i in range(3):
    a[i, i+1] = 1
    a[i, 2 - i] = -1
print(a)

# %%
import pickle
def save_policy(input_file):
    with open('all_state%s.bin' % '_44', 'wb') as f:
        pickle.dump(input_file, f)

def load_policy():
    with open('all_state%s.bin' % '_44', 'rb') as f:
        output_file = pickle.load(f)
    return output_file

a = {'A':123,'B':234}
save_policy(a)
b = load_policy()
print(b)

# %%

def load_policy():
    with open('policy_%s.bin' % 'first', 'rb') as f:
        output_file = pickle.load(f)
    return output_file
a = load_policy()
for i in list(a.keys())[:10]:
    print(i)
    print(a[i])
    print('------------')

# %%
from scipy.stats import poisson
import pandas as pd
import numpy as np

def poisson_probability(n, lam):
    value = poisson.pmf(n, lam)
    return pd.DataFrame(data=[[lam,n,value]], columns=['lam', 'n', 'value'])

display_df = pd.DataFrame(data=None, columns=['lam', 'n', 'value'])

for _lam in range(1, 5):
    for i in range(1, 10):
        display_df = display_df.append(poisson_probability(i, _lam),ignore_index=True)

display_df

# %%
import pandas as pd
pd.DataFrame(data=[[1,2,3]], columns=['lam', 'n', 'value'])

# %%
import numpy as np
value = np.zeros((3, 3))
value

# %%
value = value+3

# %%
import numpy as np
actions = np.arange(-5, 5 + 1)
inverse_actions = {el: ind[0] for ind, el in np.ndenumerate(actions)}
inverse_actions

# %%
import itertools
all_states = ((i, j) for i, j in itertools.product(list(range(4)), list(range(4))))
dir(all_states)

# %%
for i,j in itertools.product(list(range(4)), list(range(4))):
    print(i,j)

# %%
all_states.throw(1)

# %%
next(all_states)

# %%
np.zeros((4, 4, np.size(np.arange(-4, 4 + 1))))

# %%
import numpy as np
np.round([0.113131,0.6356564], 5)

# %%
card = 1
usable_ace_player = True
usable_ace_player |= (1 == card)
usable_ace_player

# %%
while True:
    a = 2
    print('hello world')
    assert a==1
    if 1 == 1:
        break

# %%
from collections import defaultdict
_Q = defaultdict(lambda: defaultdict(int))


# %%
_Q4 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
_Q3 = defaultdict(lambda: defaultdict(int))
V = 100
for i in range(3):
    for j in range(2):
        for k in range(2):
            _Q4[i][j][k]=V
_Q4

# %%
V = 100
for i in range(3):
    for j in range(2):
        _Q3[i][j]=V
_Q3

# %%
max(_Q3[0],key=_Q3[0].get)

# %%
_Q3[4][5]-=0
_Q3

# %%
