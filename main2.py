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

######################################################################
#   Data, Tools
######################################################################

def loadData(fileName='yelp-dataset/yelp_25k.csv'):
    global _df
    # Import Dataset, restaurants only

    _df = pd.read_csv(fileName)
    restaurants = _df[_df['categories'].str.contains("Restaurants")]
    foods = _df[_df['categories'].str.contains("Food")]
    _df = pd.concat([restaurants, foods])
    _df = _df.drop_duplicates()
    _df.set_index("review_id", inplace=True)

    return _df

# gets a random review (review_id,text)
def randReview():
    while True:
        try:
            review_id = random.choice(_df.index)
            break
        except KeyError:
            print("Oops!  Couldn't get a random review. Try again...")
    return (review_id, _df.loc[review_id].text)


def condenseReview(review):
    return review.replace("\n", " ")

def casualTokenize(raw_sentence, preserve_case=False):
  return casual_tokenize(raw_sentence, preserve_case=preserve_case, reduce_len=True)


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
    if type(root) is not nltk.Tree:
        return root

    print("root: ", root, type(root) is nltk.Tree)
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

# TODO not getting good sentiment scores
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


# Unresolved coreferencing issue
# In the meantime, don't include these stop nouns
STOP_NOUNS = ["i", "me", "we", "they", "it"]
def aspectSentimentCalculation(review):
    # 1. Get noun chunks and other, rebuild sentences
    subj_senti = []

    nchunks = analyzeNounChunks(review)
    root_to_head = list(zip(nchunks['root'], nchunks['head']))

    subject_sentences = [(root.text,  nltkTreeToString(spacyToNltkTree(head))) for root,head in root_to_head]

    # Calculate sentiment on each rebuilt sentence
    for subj,sentence in subject_sentences:
        if subj.lower() not in STOP_NOUNS:
            # print(sentence)
            score = sentiWordNet(sentence, debug=False)
            subj_senti.append((subj,score,sentence))

    return subj_senti


######################################################################
#   Topic Classification
######################################################################
def maxScore(food_score, atms_score, serv_score, prce_score):

    # each dict of scores contains multiple scores, depending on the # of seed words per topic
    # we want the max score for each topic
    food_max_sc = [(k, v) for k, v in sorted(food_score[1].items(), key=lambda item: item[1], reverse=True)][0][1]
    atms_max_sc = [(k, v) for k, v in sorted(atms_score[1].items(), key=lambda item: item[1], reverse=True)][0][1]
    serv_max_sc = [(k, v) for k, v in sorted(serv_score[1].items(), key=lambda item: item[1], reverse=True)][0][1]
    prce_max_sc = [(k, v) for k, v in sorted(prce_score[1].items(), key=lambda item: item[1], reverse=True)][0][1]

    max_scores = [("FOOD", food_max_sc),
              ("ATMS", atms_max_sc),
              ("SERV", serv_max_sc),
              ("PRCE", prce_max_sc)]

    # sort the list of max scores and return the topic with the highest score
    most_similar_topic = [(k, v) for k, v in sorted(max_scores, key=lambda item: item[1], reverse=True)][0]

    return most_similar_topic

def yelpSimilarities(review, nouns):
    global FOOD_SEEDS, ATMS_SEEDS, SERV_SEEDS, PRCE_SEEDS

    # calculate scores for all the nouns for each possible topic
    food_sims = analyzeS2VSimilarity(FOOD_SEEDS, review, nouns, log_to="food_similarity_vectors.txt")
    atms_sims = analyzeS2VSimilarity(ATMS_SEEDS, review, nouns, log_to="atms_similarity_vectors.txt")
    serv_sims = analyzeS2VSimilarity(SERV_SEEDS, review, nouns, log_to="serv_similarity_vectors.txt")
    prce_sims = analyzeS2VSimilarity(PRCE_SEEDS, review, nouns, log_to="prce_similarity_vectors.txt")
    assigned_topics = []
    for i in range(len(nouns)):
        most_similar_topic = maxScore(food_sims[i],atms_sims[i],serv_sims[i],prce_sims[i])
        assigned_topics.append((nouns[i], most_similar_topic))
    return assigned_topics

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


#########################
# Summarizing Results
#########################

def summarizeResults(topics_sentiments, categories=["FOOD","ATMS","SERV","PRCE"]):
    scores = {c:0 for c in categories}
    for topic,sentiment in topics_sentiments:
        scores[topic] += sentiment

    if not len(scores) == len(categories):
        print("Error Mismatch: Assigned topics don't match given categories. Extra topics include: ", list(scores)[len(categories):])
        return "Error: no summary given"

    return [(topic,score) for topic,score in scores.items()]
    # return ",".join([topic + " (" + str(score) + ")" for topic,score in scores.items()])

#########################
# Example of pipeline
#########################
def calculateAESO_one():
    _df = loadData()

    # Get a random (review_id, text)
    ex = randReview()

    LOGDATA = []
    LOGDATA.append("###########################################################")
    LOGDATA.append("review:: " + condenseReview(ex[1]))
    LOGDATA.append("###########################################################")
    TT = PrettyTable(["sentence", "subj", "sent_score", "topic", "sim_score"])

    nouns_sentiments = aspectSentimentCalculation(ex[1])
    nouns = [s for s,senti,sentence in nouns_sentiments]
    assigned_topics = yelpSimilarities(ex[1],nouns)

    temp_join = list(zip(nouns_sentiments, assigned_topics))
    subj_topic_sentiment = []   # (phrase, subject, sentiment, assigned topic, similarity score)
    summary_tup = []                # (topic, sentiment)
    for ns,at in temp_join:
        if not ns[0] == at[0]:
            print("Error Mismatch: Trying to join subject/sentiment with subject/topic. Subject '" + ns[0] + "' does not match '" + at[0] + "'")
            continue
        # add row: phrase, subject, sentiment, assigned topic, similarity score
        subj_topic_sentiment.append((ns[2], ns[0], ns[1], at[1][0], at[1][1]))
        summary_tup.append((at[1][0], ns[1]))
        TT.add_row([ns[2], ns[0], ns[1], at[1][0], at[1][1]])

    summary = summarizeResults(summary_tup)
    LOGDATA.append(topic + " (" + score + ") " for topic,score in summary)
    LOGDATA.append(TT.get_string())

    # save file
    save = open("complete_analysis.txt", 'a+')
    save.writelines('\n'.join(LOGDATA) + '\n')
    save.close()


def calculateAESO():
    # Prepare dataframe to save data
    df = {'review_id': [], 'text': [], 'food_score': [], 'atms_score': [], 'serv_score': [], 'prce_score': []}

    human_scores = loadData()

    for idx, row in human_scores.iterrows():
        ex = (row['review_id'], row['text'])
        df['text'].append(row['text'])
        df['review_id'].append(row['review_id'])

        # ########################
        # Example of pipeline
        # ########################
        LOGDATA = []
        LOGDATA.append("###########################################################")
        LOGDATA.append("review:: " + condenseReview(ex[1]))
        LOGDATA.append("###########################################################")
        TT = PrettyTable(["sentence", "subj", "sent_score", "topic", "sim_score"])

        nouns_sentiments = aspectSentimentCalculation(ex[1])
        nouns = [s for s, senti, sentence in nouns_sentiments]
        assigned_topics = yelpSimilarities(ex[1], nouns)

        temp_join = list(zip(nouns_sentiments, assigned_topics))
        subj_topic_sentiment = []  # (phrase, subject, sentiment, assigned topic, similarity score)
        summary_tup = []  # (topic, sentiment)
        for ns, at in temp_join:
            if not ns[0] == at[0]:
                print("Error Mismatch: Trying to join subject/sentiment with subject/topic. Subject '" + ns[
                    0] + "' does not match '" + at[0] + "'")
                continue
            # add row: phrase, subject, sentiment, assigned topic, similarity score
            subj_topic_sentiment.append((ns[2], ns[0], ns[1], at[1][0], at[1][1]))
            summary_tup.append((at[1][0], ns[1]))
            TT.add_row([ns[2], ns[0], ns[1], at[1][0], at[1][1]])

        summary = summarizeResults(summary_tup)

        df['food_score'].append(summary[0][1])
        df['atms_score'].append(summary[1][1])
        df['serv_score'].append(summary[2][1])
        df['prce_score'].append(summary[3][1])

        # LOGDATA.append(topic + " (" + score + ") " for topic,score in summary)
        # LOGDATA.append(TT.get_string())

        # # save file
        # save = open("complete_analysis.txt", 'a+')
        # save.writelines('\n'.join(LOGDATA) + '\n')
        # save.close()

    # Save to csv
    system_scores = pd.DataFrame(df)
    system_scores.to_csv('D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\data\\results.csv')
