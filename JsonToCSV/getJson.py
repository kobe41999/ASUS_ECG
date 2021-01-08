import requests
import config as C


def getToken():
    r = requests.post(C.getTokenUrl, json=C.User)
    jsonResponse = r.json()
    Token = jsonResponse['token']
    return Token


def getData(Token):
    r = requests.get(C.getSchemaUrl, headers={'Authorization': Token})
    jsonResponse = r.json()
    ecg_data = jsonResponse['data']
    return ecg_data
