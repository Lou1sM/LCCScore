from LCC.discrete_lcc import discrete_lcc


with open('data/library_of_babel_Borges.txt') as f:
    text = f.read()
results = discrete_lcc(text, nits=4)
print(results['LCCScore'])

