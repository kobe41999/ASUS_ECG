getTokenUrl = "https://omnicare.asus.life/auth/user"

User = {"uid": "jimmy850419@gmail.com", "pwd": "abcd1234"}

deviceID = "VivowatchSE-SNC-0001"

startTime = "20200515000000"

endTime = "20200720000000"

schema = "omnicare_ecg"

getDataUrl = f"https://omnicare.asus.life/api/data/device/{deviceID}/{startTime}/{endTime}"

getSchemaUrl = f"https://omnicare.asus.life/api/data/device/{schema}/{startTime}/{endTime}"

trainCSV = './JsonToCSV/train.csv'

