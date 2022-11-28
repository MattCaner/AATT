from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint

rogue = ROUGEScore()

candidates = [['he','is','a','retard'],['captain','leaves','the','sinking','ship','last'],['time','to','cause','some','severe','turbulence']]

references = [[['he','is','a','retard'],['he','is','a','stupid','man']],[['captain','leaves','the','ship','last'],['captain','never','leaves','the','ship']],[['time','to','cause','some','severe','turbulence']]]

candidates_fused = [' '.join(s) for s in candidates]
references_fused = [[' '.join(s) for s in r ] for r in references]

print(candidates_fused)
print(references_fused)

print(bleu_score(candidates,references))
pprint(rogue(candidates_fused,references_fused))


candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]

print(bleu_score(candidate_corpus, references_corpus))

candidates_next = [['1','2','3','4'],['5','6']]
references_next = [[['1','2','3','4'],['1','3','3','5']], [['5','9','6']]]
print(bleu_score(candidates_next,references_next))

candidates2 = [[1,3,3,7,10],[2,3,11,29,5,5],[5,3,2,2]]
references2 = [[[1,3,7,10]],[[2,3,11,29,5]],[[5,3,2]]]

candidates_text = []

for c in candidates2:
    c = [str(cc) for cc in c]
    candidates_text.append(c)


references_text = []

for r in references2:
    newrr = []
    for rr in r:
        rr = [str(rrr) for rrr in rr]
        newrr.append(rr)
    references_text.append(newrr)


print(candidates_text)
print(references_text)


print(bleu_score(candidates_text,references_text))
