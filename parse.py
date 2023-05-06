import requests
import time
import os
import logging

logger = logging.getLogger(__name__)


class VKMutual:
    def __init__(self, token, api_version='5.130', max_request_per_sec=3):
        self.token = token
        self.api_version = api_version
        self.max_request_per_sec = max_request_per_sec
        self.last_request = 0

    def request_delay(self):
        delay = (self.last_request+1/self.max_request_per_sec) - time.time()
        if delay > 0:
            time.sleep(delay)
            logger.debug("Delay {}".format(delay))
        self.last_request = time.time()

    def default_request(self, api_method, query_params):
        self.request_delay()
        base_url = "https://api.vk.com/"
        address = "method/" + api_method
        base_query_params = {
            "access_token": self.token,
            "v": self.api_version
        }
        base_query_params.update(query_params)
        data = requests.get(base_url+address, base_query_params).json()
        if "error" in data:
            logger.error("Error code = {}: {}".format(data["error"]["error_code"], data["error"]["error_msg"]))
            raise Exception(data["error"]["error_msg"])
        return data

    def get_friends(self):
        method = 'friends.get'
        params = {
            "fields": "sex"
        }
        data = self.default_request(method, params)
        friends = data['response']['items']
        logging.debug("Get friends:{}".format(friends))
        return friends

    def get_mutual_friends(self, friends=None):
        mutual = {}
        method = 'friends.getMutual'
        if friends is None:
            friends = self.get_friends()
        number_friends = len(friends)
        for index, friend in enumerate(friends):
            logger.debug("Try get friend {}/{}".format(index, number_friends))
            params = {
                "order": "hints",
                "target_uid": friend['id']
            }
            try:
                req = self.default_request(method, params)
            except Exception as ex:
                logger.warning(ex)
                continue
            if len(req['response']) == 0:
                continue
            mutual[friend['id']] = req['response']
        return mutual


def test():
    from prettytable import PrettyTable
    logging.basicConfig(level=logging.DEBUG)
    token = os.environ['VK_TOKEN']
    print(token)
    api_version = '5.130'
    vk_mutual = VKMutual(token, api_version)
    friends = vk_mutual.get_friends()
    table = PrettyTable(["id", "name", "last name"])
    for friend in friends:
        table.add_row([friend["id"], friend["first_name"], friend["last_name"]])
    print(table)
    mutual_friends = vk_mutual.get_mutual_friends()
    table = PrettyTable(["id", "friends"])
    for key in mutual_friends.keys():
        friends_str = "\n".join(list(map(lambda x: str(x), mutual_friends[key])))
        table.add_row([key, friends_str])
    print(table)


if __name__ == "__main__":
    test()
