import re, unicodedata

LEGAL = {'llc', 'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'pvt', 'private', 'llp', 'lp',
         'pllc', 'pc', 'pa', 'plc', 'group', 'the', 'and', 'of',
         'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'ei', 'cie', 'et', 'de', 'des', 'du', 'la', 'le', 'les'}

def core(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = s.replace('0', 'o').replace('1', 'l').replace('8', 'b')
    toks = [t for t in re.findall(r'[a-z]+', s) if t not in LEGAL]
    return ' '.join(sorted(set(toks)))
