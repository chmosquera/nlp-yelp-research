import spacy, nltk, random
from nltk import Tree
from nltk.corpus import sentiwordnet, stopwords, wordnet
from nltk.tokenize.casual import casual_tokenize
from nltk.wsd import lesk
import pandas as pd
from prettytable import PrettyTable
from pprint import pprint
from sense2vec import Sense2VecComponent

# Resources for determining similarity: Spacy, sense2vec
s2v_path = "D:\\Programs\\Python37x64\\nlp_config\\s2v_reddit_2015_md"
spacy_lg_path = 'D:\\Programs\\Python37x64\\nlp_config\\venv\\Lib\\site-packages\\en_core_web_lg\\en_core_web_lg-2.2.5'
nlp = spacy.load(spacy_lg_path)
s2v = Sense2VecComponent(nlp.vocab).from_disk(s2v_path)
nlp.add_pipe(s2v)


nltk.download('sentiwordnet')
nltk.download('stopwords')

FOOD_SEEDS = "food drink"
ATMS_SEEDS = "atmosphere place environment"
SERV_SEEDS = "server management time"
PRCE_SEEDS = "money expensive"

def casualTokenize(raw_sentence, preserve_case=False):
  return casual_tokenize(raw_sentence, preserve_case=preserve_case, reduce_len=True)

def loadData(fileName='yelp-dataset/yelp_25k.csv'):
    global _df
    # Import Dataset, restaurants only

    _df = pd.read_csv(fileName)
    restaurants = _df[_df['categories'].str.contains("Restaurants")]
    foods = _df[_df['categories'].str.contains("Food")]
    _df = pd.concat([restaurants, foods])

    return _df


def getWordnetPOS(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

######################################################################
#   Isolate possible aspects by splitting sentences syntactically
######################################################################

def spacyToNltkTree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [spacyToNltkTree(child) for child in node.children])
    else:
        return node.orth_

def nltkTreeToString(root):
    S = root.label() + " "
    for node in root:
        if type(node) is nltk.Tree:
            S += nltkTreeToString(node) + " "
        else:
            S += node + " "

    return S

def analyzeNounChunks(sentence):
  TT = PrettyTable(['Chunk', 'Root', '-->', 'Head', 'Children'])
  doc = nlp(sentence)
  chunks, roots, dep_s, heads, children = [],[],[],[],[]
  for chunk in doc.noun_chunks:
      chunks.append(chunk)
      roots.append(chunk.root)
      dep_s.append(chunk.root.dep_)
      heads.append(chunk.root.head)
      children.append(chunk.root.children)
      # noun_chunks.loc[len(noun_chunks.index)] = [chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head, ", ".join([child.text for child in chunk.root.children])]
      TT.add_row([chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text, ", ".join([child.text for child in chunk.root.children])])

  nchunk_dict = {'chunk': chunks, 'root': roots, 'dep_' : dep_s, 'head': heads,'children': children}

  print(TT)
  return nchunk_dict

######################################################################
#   Word Sense Disambiguation
######################################################################
def wsdLesk(sentence, word, tree_pos=None, debug=False):
    tokenized = casualTokenize(sentence)
    wsd_syn = None

    # perform lesk without POS
    if not tree_pos:
        wsd_syn = lesk(tokenized, word)
    # OR perform lesk with POS
    else:
        wn_pos = getWordnetPOS(tree_pos)
        wsd_syn = lesk(tokenized, word, pos=wn_pos)

        # but if none found, perform without POS
        wsd_syn = lesk(tokenized, word)

    if debug and wsd_syn is not None:
        print(word, wsd_syn, wsd_syn.definition())
    return wsd_syn

######################################################################
#   Sentiment
######################################################################

def calculateSentiment(pos, neg, objv):
    # objectivity threshold
    if objv > 0.625:
        return 0.0
    return pos - neg


def sentiWordNet(sentence, debug=True):
    # Used for debugging
    LOGDATA = []
    T_SENTI = PrettyTable(['word', 'tag', 'def', '+', '-', 'objv'])

    # begin
    total_sentiment = 0

    doc = nlp(sentence)
    filtered_spacy_tokens = [tok for tok in doc if not tok.text in stopwords.words('english') and tok.text.isalnum()]
    tagged = [(tok.text, tok.tag_) for tok in filtered_spacy_tokens]
    tokens = [tok for tok, tag in tagged]

    if debug:
        print(tagged)
        LOGDATA.append(tagged)

    for tok, tag in tagged:
        wntag = getWordnetPOS(tag)

        syn = wsdLesk(sentence, tok, tag)
        defn, pos, neg, objv = None, None, None, None
        if syn:
            syn_sent = sentiwordnet.senti_synset(syn.name())
            defn = syn.definition()
            pos = syn_sent.pos_score()
            neg = syn_sent.neg_score()
            objv = syn_sent.obj_score()

            total_sentiment += calculateSentiment(pos, neg, objv)

        T_SENTI.add_row([tok, tag, defn, pos, neg, objv])
        LOGDATA.append([tok, tag, defn, pos, neg, objv])

    if debug:
        print(T_SENTI)

    return total_sentiment


def aspectSentimentCalculation(review):
    # 1. Get noun chunks and other, rebuild sentences
    subj_senti = []

    nchunks = analyzeNounChunks(review)
    root_to_head = list(zip(nchunks['root'], nchunks['head']))

    subject_sentences = [(root.text,  nltkTreeToString(spacyToNltkTree(head))) for root,head in root_to_head]

    # Calculate sentiment on each rebuilt sentence
    for subj,sentence in subject_sentences:
        # print(sentence)
        score = sentiWordNet(sentence, debug=False)
        subj_senti.append((subj,score))

    return subj_senti


##
def analyzeS2VSimilarity(compare_to, sentence, words, use_s2v=True, log_to="s2v_similarity_vectors.txt"):
    logdata = []
    logdata.append("===============================================")
    logdata.append("## Similarity Analysis (use_s2v=" + str(use_s2v) + ")")
    logdata.append("## compare_to: " + ", ".join(compare_to))
    logdata.append("## sentence: " + sentence)
    logdata.append("===============================================")

    compare_to = nlp(compare_to)
    # convert sentence to nlp spacy
    sentence = nlp(sentence)
    # convert words to spacy tokens from the sentence
    words = [tok for tok in sentence for w in words if w == tok.text]

    result = []

    TT = PrettyTable(['word'] + [tok for tok in compare_to])

    for tok2 in words:
        # Use sv2 vectors
        if use_s2v and (tok2._.in_s2v or tok2.has_vector):
            row = []
            score_dict = {}
            for tok1 in compare_to:
                # First try s2v vectors
                if tok1.has_vector and tok2._.in_s2v:
                    score = tok2._.s2v_similarity(tok1)
                    row.append(score)
                    score_dict[tok1.text] = score
                # otherwise just use standard vectors
                elif tok1.has_vector and tok2.has_vector:
                    score = tok2.similarity(tok1)
                    row.append(score)
                    score_dict[tok1.text] = score
                else:
                    row.append("None")
                    score_dict[tok1.text] = None
            TT.add_row([tok2.text] + row)
            result.append((tok2.text, score_dict))

        # Use standard vectors
        elif not use_s2v and tok2.has_vector:
            row = []
            score_dict = {}
            for tok1 in compare_to:
                if tok1.has_vector:
                    score = tok1.similarity(tok2)
                    row.append(score)
                    score_dict[tok1] = score
            TT.add_row([tok2.text] + row)
            result.append((tok2.text, score_dict))

        # no vectors exist
        else:
            similarities = ['None' for i in range(len(compare_to))]
            TT.add_row([tok2.text] + similarities)

    logdata.append(TT.get_string())

    # save file
    if log_to:
        save = open(log_to, 'a+')
        save.writelines('\n'.join(logdata) + '\n')
        save.close()

    return result
#
#
# loadData()

##
loadData()
ex = random.choice(_df['text'])
print(ex)
subj_senti = aspectSentimentCalculation(ex)
print(subj_senti)
subj = [s for s,senti in subj_senti]

food_sims = analyzeS2VSimilarity(FOOD_SEEDS, ex, subj)
atms_sims = analyzeS2VSimilarity(ATMS_SEEDS, ex, subj)