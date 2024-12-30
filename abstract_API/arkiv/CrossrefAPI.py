
import pprint

"""
Only 3% has absracts

https://stackoverflow.com/questions/55081656/database-of-scientific-paper-abstracts

TerMINATE

"""

'''
### Test DOI
from crossref.restful import Works, Etiquette
works = Works()

test=works.doi('https://doi.org/10.1109/ACC.2012.6315468')

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(test)'''

'''
10.1590/0102-311x00133115

### not crossref examples
works.agency('10.6084/m9.figshare.1314859.v1') 
works.agency('10.5240/B1FA-0EEC-C316-3316-3A73-L')

item['title']

https://api.crossref.org/works?filter=has-full-text:true&mailto=GroovyBib@example.org
'''



 


"""
var = "IGARSS" 
url = f"https://dblp.org/search/venue/api?q={var}&format=json&h=1"

response = requests.get(url)
data = response.json()

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(data.get("result").get("hits").get("hit")[0].get("info").get("url"))"""




### https://api.crossref.org/swagger-ui/index.html