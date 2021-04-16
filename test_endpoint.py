import requests
fac_info = requests.get('http://quickpass-backend.azurewebsites.net/facility/by/602ea8d423a00b4812b77ee6')
print(fac_info.json())
isCapacitySet = fac_info.headers['isCapacitySet']