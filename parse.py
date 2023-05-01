import requests
import warnings
import pickle
import time
import json
from tqdm import tqdm
from functools import wraps
from datetime import datetime
from typing import Dict, List, Union, Callable, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
import math
from tqdm import tqdm
import pandas as pd'


class VKMutual:
    """
    Get friends and their mutual friends
    """
    def __init__(self, token, v):
        self.token = token
        self.v = v
        self.friends = self.get_friends()

    def get_req(self, method, parameters):
        return requests.get('https://api.vk.com/method/%s?%s&access_token=%s' % (method, '&'.join(parameters), \
                                                                                 self.token)).json()

    def get_friends(self):
        method = 'friends.get'
        data = self.get_req(method, ['fields=sex', f'v={self.v}'])
        friends = data['response']['items']
        return friends

    def get_mutual_friends(self):
        mutual = {}
        for i in self.friends:
            x = (self.get_req('friends.getMutual', ['order=hints', \
                                                    'target_uid=%s' % i['id'], f'v={self.v}']))
            try:
                mutual[i['id']] = x['response']
            except:
                continue

            time.sleep(0.5)

        return mutual