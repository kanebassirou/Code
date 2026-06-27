import requests
class EarthdataSession(requests.Session):
    def rebuild_auth(self, prepared_request, response): pass
session = EarthdataSession()
session.auth = ('bassiroukane', 'eT&57igjaC&Lf@Z')
r = session.get('https://cddis.nasa.gov/archive/gnss/products/ionex/2023/300/')
for line in r.text.split('\n'):
    if 'JPL' in line or 'jpl' in line:
        print(line)
