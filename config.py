getTokenUrl = "https://omnicare.asus.life/auth/user"

User = {"uid": "jimmy850419@gmail.com", "pwd": "abcd1234"}

deviceID = "VivowatchSE-SNC-0003"

startTime = "20200815000000"

endTime = "20210630000000"

schema = "omnicare_ecg"

getDataUrl = f"https://omnicare.asus.life/api/data/device/{deviceID}/{startTime}/{endTime}"

getSchemaUrl = f"https://omnicare.asus.life/api/data/schema/{schema}/{startTime}/{endTime}"

trainCSV = './JsonToCSV/train.csv'

