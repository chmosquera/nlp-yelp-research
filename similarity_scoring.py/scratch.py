import spacy
from sense2vec import Sense2VecComponent
s2v_path = "D:\\Programs\\Python37x64\\nlp_config\\s2v_reddit_2015_md"
spacy_lg_path = 'D:\\Programs\\Python37x64\\nlp_config\\venv\\Lib\\site-packages\\en_core_web_lg\\en_core_web_lg-2.2.5'

nlp = spacy.load(spacy_lg_path)
s2v = Sense2VecComponent(nlp.vocab).from_disk(s2v_path)
nlp.add_pipe(s2v)

from nltk.tokenize.casual import casual_tokenize
from prettytable import PrettyTable

FOOD_SEEDS = "food drink"
ATMS_SEEDS = "atmosphere place environment"
SERV_SEEDS = "server management time"
PRCE_SEEDS = "money expensive"

def  casualTokenize(raw_sentence, preserve_case=False):
  return casual_tokenize(raw_sentence, preserve_case=preserve_case, reduce_len=True)


def maxSimScore(subj_similarity_scores):

    scores = []
    for subj,score_dict in subj_similarity_scores:
        sorted_scores = [(k, v) for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)]
        max = sorted_scores[0][1]   # element 0 has highest score 1
        print(subj, max)

        # similarity score threshold
        if max > 0.425:
            scores.append((subj, max))

    print(scores)
    return scores


def yelpSimilarities(review, nouns):
    global FOOD_SEEDS, ATMS_SEEDS, SERV_SEEDS, PRCE_SEEDS

    # calculate scores for all the nouns for each possible topic
    food_sims = analyzeS2VSimilarity(FOOD_SEEDS, review, nouns, log_to="food_similarity_vectors.txt")
    atms_sims = analyzeS2VSimilarity(ATMS_SEEDS, review, nouns, log_to="atms_similarity_vectors.txt")
    serv_sims = analyzeS2VSimilarity(SERV_SEEDS, review, nouns, log_to="servsimilarity_vectors.txt")
    prce_sims = analyzeS2VSimilarity(PRCE_SEEDS, review, nouns, log_to="prce_similarity_vectors.txt")
    for i in range(len(nouns)):
        scores = [("FOOD", food_sims[i]),
                  ("ATMS", atms_sims[i]),
                  ("SERV", serv_sims[i]),
                  ("PRCE", prce_sims[i])]

        # the topic with the highest avg score is the assigned topic
        sorted_scores = [(k, v) for k, v in sorted(scores, key=lambda item: item[1], reverse=True)]
        print(nouns[i], sorted_scores)


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
    print(compare_to)
    print(sentence)
    print(words)

    result = []

    TT = PrettyTable(['word'] + [tok for tok in compare_to])

    for tok2 in words:
        print("tok2: ", tok2)
        # Use sv2 vectors
        if use_s2v and (tok2._.in_s2v or tok2.has_vector):
            print("made it here")
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

# sent1 = "I had to wait such a long time for the waiter to come."
sent2 = """
Mediocre chain diner with around Denny's quality or slightly lower food.  I guess the big appeal is their hours?
Maybe I have been spoiled by the awesomeness that is Chase's Diner in Chandler, but there's really no reason for me to ever go here again.
"""
# sent3 = "This place will take a long time to serve your food - as it took them almost an hour to give us our food."
# analyzeS2VSimilarity("service management host server", sent1, casualTokenize(sent1))
save = analyzeS2VSimilarity("food drink", sent2, ['chain', 'I', 'quality', 'appeal', 'hours', 'awesomeness', 'Diner', 'Chandler', 'reason', 'me'])
# analyzeS2VSimilarity("service management host server", sent3, casualTokenize(sent3))


# ##
#
# doc = nlp("A sentence about natural language processing.")
# assert doc[3:6].text == "natural language processing"
# freq = doc[3:6]._.s2v_freq
# vector = doc[3:6]._.s2v_vec
# most_similar = doc[3:6]._.s2v_most_similar(3)
# [(('machine learning', 'NOUN'), 0.8986967),
#  (('computer vision', 'NOUN'), 0.8636297),
#  (('deep learning', 'NOUN'), 0.8573361)]